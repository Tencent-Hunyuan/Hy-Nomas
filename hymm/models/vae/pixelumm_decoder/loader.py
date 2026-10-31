"""Load PixelUMM when selected by model.vae.name=pixelumm.

All checkpoint, geometry, sampling and optional acceleration settings live in
model.vae. DeepGEMM, CUTLASS and CuTe are imported only when explicitly enabled.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _get(config, key, default=None):
    value = getattr(config, key, default)
    return default if value is None else value


def _install_inference_optimizations(module, config, logger) -> None:
    """Apply the opt-in optimizations to the bare ``DCAEVPixelUMMDecoder`` (eval, no grad).

    Order matters: FP8 and CuTe replace submodules / forwards and must precede compilation.
    """
    if bool(_get(config, "deepgemm_fp8", False)):
        from .fp8 import enable_deepgemm_fp8, find_fp8_checkpoint

        if getattr(module, "_prequantized_fp8", False):
            if module._prequantized_fp8_backend != "deepgemm":
                raise ValueError("Serialized PixelUMM backend does not match deepgemm_fp8")
            count = module._prequantized_fp8_layers
            sidecar = "single-file serialized weights (no startup weight quantization)"
        else:
            sidecar = find_fp8_checkpoint(config.checkpoint)
            count = enable_deepgemm_fp8(module, checkpoint_path=sidecar)
        logger.info(
            f"PixelUMM DeepGEMM FP8: {count} MLP/Q/O linears, K/V stay BF16; "
            f"weights {sidecar or 'quantized from BF16 at startup'}"
        )
    if bool(_get(config, "cutlass_fp8", False)):
        from .cutlass_fp8 import enable_cutlass_fp8

        if getattr(module, "_prequantized_fp8", False):
            if module._prequantized_fp8_backend != "cutlass":
                raise ValueError("Serialized PixelUMM backend does not match cutlass_fp8")
            count = module._prequantized_fp8_layers
            logger.info("PixelUMM: loaded serialized FP8 weights/scales; no startup weight quantization")
        else:
            count = enable_cutlass_fp8(module)
        logger.info(f"PixelUMM CUTLASS SM89 FP8: {count} MLP/Q/O linears, 128x128 scales")
    if bool(_get(config, "cute_attention", False)):
        from .cute_attention import enable_cute_attention

        count = enable_cute_attention(module)
        logger.info(f"PixelUMM CuTe QK RMSNorm/RoPE: {count} attention layers")
    if bool(_get(config, "compile_decoder", False)):
        module.forward = torch.compile(module.forward, mode="default", fullgraph=True, dynamic=False)
        logger.info("PixelUMM decoder forward compiled (fullgraph, static shapes)")


def maybe_load_pixelumm_decoder(args, *, device, logger):
    """Load the optional PixelUMM decoder, or return ``None`` another VAE is selected."""
    config = getattr(args, "vae", None)
    if getattr(config, "name", None) != "pixelumm":
        return None

    if bool(_get(config, "deepgemm_fp8", False)) and bool(_get(config, "cutlass_fp8", False)):
        raise ValueError("Select only one FP8 backend: model.vae.deepgemm_fp8 or model.vae.cutlass_fp8")

    from hymm.distributed.parallel_states import get_model_parallel_info

    from .autoencoder import DMD2_DENOISING_STEPS, PixelUMMDCAELatentDecoder

    steps = _get(config, "denoising_step_list", DMD2_DENOISING_STEPS)
    decoder = PixelUMMDCAELatentDecoder.from_checkpoint(
        str(config.checkpoint),
        dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
            str(_get(config, "vae_precision", "bf16"))
        ],
        device=device,
        fp8_backend=("cutlass" if _get(config, "cutlass_fp8", False) else
                     "deepgemm" if _get(config, "deepgemm_fp8", False) else None),
        local_attention_tile_pixels=int(_get(config, "local_attention_tile_pixels", 128)),
        denoising_step_list=[float(v) for v in steps],
        num_steps=int(_get(config, "num_steps", 1)),
        seed=int(_get(config, "seed", 42)),
        tail_hold_latents=int(_get(config, "tail_hold_latents", 1)),
        # CPU offload also keeps the final decoded video off the GPU.
        decode_output_cpu=_get(config, "cpu_offload", False),
        temporal_window_latents=int(_get(config, "temporal_window_latents", 10)),
        temporal_overlap_latents=int(_get(config, "temporal_overlap_latents", 5)),
        decoder_tile_pixels=int(_get(config, "tile_pixels", 0)),
        decoder_tile_overlap_pixels=int(_get(config, "tile_overlap_pixels", 64)),
    )
    _install_inference_optimizations(decoder.decoder, config, logger)

    mode = str(_get(config, "parallel_mode", "none")).strip().lower()
    if mode != "none":
        group, world_size, _ = get_model_parallel_info()
        if not (dist.is_available() and dist.is_initialized() and group is not None and world_size > 1):
            raise RuntimeError(f"PixelUMM decode mode {mode!r} requires an active multi-rank SP/TP model group")
        decoder.configure_distributed_decode(
            mode, process_group=group, output_rank=int(_get(config, "output_rank", 0))
        )

    logger.info(
        "PixelUMM decoder enabled: "
        f"checkpoint={config.checkpoint}, steps={decoder.denoising_step_list[: decoder.num_steps]}, "
        f"window={decoder.temporal_window_latents}/{decoder.temporal_overlap_latents}, "
        f"tile={decoder.decoder_tile_pixels}/{decoder.decoder_tile_overlap_pixels}, "
        f"local_attention_tile_tokens={decoder.decoder.local_attention_tile}, mode={mode}"
    )
    return decoder
