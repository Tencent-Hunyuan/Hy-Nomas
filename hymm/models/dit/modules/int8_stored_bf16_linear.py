"""INT8 stored Linear weights with BF16 compute and validated checkpoint loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

FORMAT = "nomas-dit-w8-bf16-v1"
GROUP_SIZE = 128
MANIFEST = "w8_bf16_manifest.json"
SINGLE_FILE_MANIFEST = "w8_bf16_manifest"
NOMAS_SINGLE_FILE_FORMAT = "nomas-int8-bf16-single-v1"


def resolve_w8_bf16_checkpoint(inference_config, checkpoint):
    mode = getattr(inference_config, "dit_weight_quantization", "none")
    if mode == "fp8":
        from .fp8_linear import FORMAT
        manifest = Path(checkpoint).expanduser() / "fp8_manifest.json"
        if not manifest.is_file() or json.loads(manifest.read_text()).get('format') != FORMAT:
            raise ValueError("FP8 DiT requires a converted fp8_manifest.json checkpoint directory")
        return False
    if mode not in {"none", "int8"}:
        raise ValueError("dit_weight_quantization must be none, int8, or fp8")
    path = Path(checkpoint).expanduser()
    if (path / "fp8_manifest.json").is_file():
        raise ValueError("FP8 DiT checkpoint requires inference.dit_weight_quantization=fp8")
    if path.is_file():
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            metadata = reader.metadata() or {}
            present = (SINGLE_FILE_MANIFEST in metadata
                       or metadata.get("format") == NOMAS_SINGLE_FILE_FORMAT)
        if present:
            _read_single_file_manifest(path)
    else:
        present = (path / MANIFEST).is_file()
    if mode == "int8" and not present:
        raise ValueError(
            "INT8 DiT requires a W8/BF16 manifest directory or a single-file "
            f"safetensors checkpoint with {SINGLE_FILE_MANIFEST} metadata "
            f"or format={NOMAS_SINGLE_FILE_FORMAT}"
        )
    if mode == "none" and present:
        raise ValueError("W8/BF16 checkpoint requires dit_weight_quantization=int8")
    return present


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel, per-128-input-group INT8 quantization."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise TypeError("Expected a floating-point [out_features, in_features] weight")
    if not torch.isfinite(weight).all():
        raise ValueError("Cannot quantize a weight containing NaN or Inf")
    n, k = weight.shape
    groups = (k + GROUP_SIZE - 1) // GROUP_SIZE
    padded = F.pad(weight.float(), (0, groups * GROUP_SIZE - k))
    rows = padded.reshape(n, groups, GROUP_SIZE)
    maximum = rows.abs().amax(dim=-1)
    scale = torch.where(maximum > 0, maximum / 127.0, 1.0)
    normalized = rows / scale.unsqueeze(-1)
    rounded = (normalized.abs() + 0.5).floor() * normalized.sign()
    q = rounded.clamp_(-127, 127).to(torch.int8).reshape(n, -1)[:, :k].contiguous()
    return q, scale.contiguous()


class Int8StoredBF16Linear(nn.Module):
    """Only stores INT8 weight and FP32 scales; materializes BF16 per call."""

    def __init__(self, original: nn.Linear):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.register_buffer(
            "weight_q", torch.empty((self.out_features, self.in_features),
                                    dtype=torch.int8, device="meta")
        )
        self.register_buffer(
            "weight_scale", torch.empty((self.out_features,
                                          (self.in_features + GROUP_SIZE - 1) // GROUP_SIZE),
                                         dtype=torch.float32, device="meta")
        )
        self.bias = original.bias

    def dequant_weight(self) -> torch.Tensor:
        scales = self.weight_scale.repeat_interleave(GROUP_SIZE, dim=1)
        weight = self.weight_q.float() * scales[:, :self.in_features]
        return weight.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            if (x.device.type == "cuda" and torch.is_autocast_enabled("cuda")
                    and torch.get_autocast_dtype("cuda") == torch.bfloat16):
                x = x.to(torch.bfloat16)
            else:
                raise TypeError(f"INT8 stored BF16 Linear requires BF16 input, got {x.dtype}")
        return F.linear(x, self.dequant_weight(), self.bias)


def linear_weight(module: nn.Module) -> torch.Tensor | nn.Module:
    if isinstance(module, Int8StoredBF16Linear):
        return module.dequant_weight()
    return module if getattr(module, "is_fp8_linear", False) else module.weight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linear_dtype(module):
    return torch.bfloat16 if isinstance(module, Int8StoredBF16Linear) or getattr(module, "is_fp8_linear", False) else module.weight.dtype


def _tensor_sha256(tensor):
    """Hash exact CPU tensor bytes, including BF16 and INT8 storage."""
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _read_nomas_single_file_manifest(reader, metadata, model_type=None):
    """Validate the NOMAS v1 header without reading or requantizing payloads."""
    for key, expected in (
        ("group_size", str(GROUP_SIZE)),
        ("quantization", "symmetric_int8_absmax"),
        ("compute_dtype", "bfloat16"),
        ("rounding", "half_away_from_zero"),
    ):
        if metadata.get(key) != expected:
            raise ValueError(f"NOMAS INT8 checkpoint requires {key}={expected}")
    declared_model_type = metadata.get("model_type")
    if not isinstance(declared_model_type, str) or not declared_model_type.strip():
        raise ValueError("NOMAS INT8 checkpoint requires model_type metadata")
    if model_type is not None and declared_model_type != model_type:
        raise ValueError("NOMAS INT8 model_type does not match configuration")
    try:
        layers = json.loads(metadata.get("layers", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("NOMAS INT8 layers metadata must be a JSON object") from exc
    if not isinstance(layers, dict) or not layers:
        raise ValueError("NOMAS INT8 metadata requires nonempty layers")
    for name, shape in layers.items():
        if (not isinstance(name, str) or not name
                or not isinstance(shape, list) or len(shape) != 2
                or any(type(dim) is not int or dim <= 0 for dim in shape)):
            raise ValueError(f"Invalid NOMAS INT8 layer name/shape: {name!r}")
    keys = set(reader.keys())
    for suffix in (".weight_q", ".weight_scale"):
        if {key.removesuffix(suffix) for key in keys if key.endswith(suffix)} != set(layers):
            raise ValueError(f"NOMAS INT8 layers do not match {suffix} tensor keys")
    for name, (out_features, in_features) in layers.items():
        for suffix, shape, dtype in (
            (".weight_q", [out_features, in_features], "I8"),
            (".weight_scale", [out_features, (in_features + GROUP_SIZE - 1) // GROUP_SIZE], "F32"),
        ):
            key = name + suffix
            tensor = reader.get_slice(key)
            if tensor.get_shape() != shape or tensor.get_dtype() != dtype:
                raise ValueError(f"NOMAS INT8 tensor shape/dtype mismatch: {key}")
    # NOMAS v1 records source provenance, not per-tensor checksums. Tensor
    # keys already use runtime names; original_key_map is provenance only.
    # The shared loader still checks the complete model, dtypes and values.
    return {"format": NOMAS_SINGLE_FILE_FORMAT, "group_size": GROUP_SIZE,
            "model_type": declared_model_type, "layers": layers}


def _read_single_file_manifest(path, model_type=None):
    """Read either supported single-file format without a sidecar manifest."""
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        if SINGLE_FILE_MANIFEST not in metadata:
            if metadata.get("format") == NOMAS_SINGLE_FILE_FORMAT:
                return _read_nomas_single_file_manifest(reader, metadata, model_type)
            raise ValueError("Single-file W8 checkpoint is missing supported embedded metadata")
        # An embedded W8 manifest always retains its mandatory checksum
        # validation, even if the file also has a NOMAS format marker.
        raw = metadata[SINGLE_FILE_MANIFEST]
        manifest = json.loads(raw)
        if (not isinstance(manifest, dict)
                or manifest.get("format") != FORMAT
                or manifest.get("group_size") != GROUP_SIZE
                or not isinstance(manifest.get("model_type"), str)):
            raise ValueError("Unsupported single-file W8/BF16 metadata")
        if model_type is not None and manifest["model_type"] != model_type:
            raise ValueError("W8/BF16 model_type does not match configuration")
        layers, checksums = manifest.get("layers"), manifest.get("tensor_sha256")
        if not isinstance(layers, dict) or not layers:
            raise ValueError("Single-file W8 metadata requires nonempty layers")
        if not isinstance(checksums, dict) or set(checksums) != set(reader.keys()):
            raise ValueError("Single-file W8 tensor checksum keys do not match file tensors")
        for key, checksum in checksums.items():
            if (not isinstance(checksum, str) or len(checksum) != 64
                    or any(c not in "0123456789abcdef" for c in checksum)):
                raise ValueError(f"Invalid single-file W8 tensor checksum: {key}")
    return manifest


def _read_manifest(root, model_type):
    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest.get("format") != FORMAT or manifest.get("group_size") != GROUP_SIZE:
        raise ValueError("Unsupported W8/BF16 manifest format or group_size")
    if manifest.get("model_type") != model_type:
        raise ValueError("W8/BF16 model_type does not match configuration")
    layers, shards = manifest.get("layers"), manifest.get("shards")
    if not isinstance(layers, dict) or not layers or not isinstance(shards, list) or not shards:
        raise ValueError("Manifest requires nonempty layers and shards")
    files = set()
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("Invalid shard record")
        name = shard.get("file")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Unsafe W8/BF16 shard path")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or name in files:
            raise ValueError("Duplicate or escaping W8/BF16 shard path")
        files.add(name)
        if type(shard.get("bytes")) is not int or shard["bytes"] <= 0:
            raise ValueError("Invalid shard byte count")
        checksum = shard.get("sha256")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Invalid shard checksum")
        if path.stat().st_size != shard["bytes"] or _sha256(path) != checksum:
            raise ValueError(f"W8/BF16 shard checksum mismatch: {name}")
    return manifest


def _restore_derived_buffers(model):
    """Rebuild camera RoPE tables omitted from the checkpoint after meta construction."""
    from hymm.models.camera.action_module import ActionModule

    for module in model.modules():
        if isinstance(module, ActionModule) and module.freqs_cos.is_meta:
            with torch.device("cpu"):
                cos, sin = module.get_rotary_pos_embed(
                    7500, module.patch_size[1], module.patch_size[2], module.rope_dim_list,
                )
            module.freqs_cos = cos.to(torch.bfloat16)
            module.freqs_sin = sin.to(torch.bfloat16)
    missing = [name for name, value in model.named_buffers() if value.is_meta]
    if missing:
        raise ValueError(f"Unmaterialized model buffers: {missing[:5]}")


def load_w8_bf16_model(model_factory, model_type, checkpoint, device):
    """Load validated INT8 storage from a manifest directory or a single file."""
    from .w8_checkpoint_cache import cached_checkpoint
    path = Path(checkpoint).expanduser().resolve()
    single_file = path.is_file()
    if single_file:
        manifest = _read_single_file_manifest(path, model_type)
        files = [path]
    else:
        root = cached_checkpoint(path)
        manifest = _read_manifest(root, model_type)
        files = [root / shard["file"] for shard in manifest["shards"]]
    checksums = manifest.get("tensor_sha256") if single_file else None
    with torch.device("meta"):
        model = model_factory()
    if any(not tensor.is_meta for tensor in model.parameters()):
        raise ValueError("W8/BF16 model_factory must construct parameters on meta")
    layers = {name: list(module.weight.shape) for name, module in model.named_modules()
              if isinstance(module, nn.Linear)}
    if any(name not in layers or shape != layers[name]
           for name, shape in manifest["layers"].items()):
        raise ValueError("W8/BF16 layer names/shapes do not match model Linear modules")
    # Converted checkpoints may keep small camera projections in BF16.
    # Every omitted Linear remains an ordinary, strictly checked tensor.
    for name in manifest["layers"]:
        original = model.get_submodule(name)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, Int8StoredBF16Linear(original))
    expected = {key: (tuple(t.shape), t.dtype) for key, t in model.state_dict().items()}
    seen = set()
    for checkpoint_file in files:
        tensors = load_file(str(checkpoint_file), device="cpu")
        for key, tensor in tensors.items():
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate W8/BF16 tensor: {key}")
            shape, dtype = expected[key]
            ordinary_bf16 = (
                not key.endswith((".weight_q", ".weight_scale"))
                and dtype.is_floating_point
                and tensor.dtype == torch.bfloat16
            )
            if tuple(tensor.shape) != shape or (tensor.dtype != dtype and not ordinary_bf16):
                raise ValueError(f"W8/BF16 tensor shape/dtype mismatch: {key}")
            if key.endswith(".weight_scale") and (
                not torch.isfinite(tensor).all() or not torch.all(tensor > 0)
            ):
                raise ValueError(f"Invalid quantization scales: {key}")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite W8/BF16 tensor: {key}")
            if checksums is not None and _tensor_sha256(tensor) != checksums[key]:
                raise ValueError(f"Single-file W8 tensor checksum mismatch: {key}")
        seen.update(tensors)
        model.load_state_dict(tensors, strict=False, assign=True)
        del tensors
    missing = expected.keys() - seen
    if missing:
        raise ValueError(f"Missing W8/BF16 tensors: {sorted(missing)[:5]}")
    _restore_derived_buffers(model)
    model.eval().requires_grad_(False)
    model.to(device=device)
    return model


def load_w8_bf16_transformer(model_type, checkpoint, args, device, logger=None):
    from hymm.models.dit.modules import load_model

    model = load_w8_bf16_model(
        lambda: load_model(model_type, args, {"device": torch.device("meta"), "dtype": torch.bfloat16}),
        model_type, checkpoint, device,
    )
    if logger is not None:
        logger.info(f"Loaded W8/BF16 DiT from {checkpoint}; backend={args.dit.seg_ar_attention_backend}")
    return model
