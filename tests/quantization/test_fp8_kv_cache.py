from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hymm.models.dit.modules import fp8_kv_cache as fk
from hymm.models.dit.modules.kv_cache_storage import cache_format, quantization_mode

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 casts are exercised on CUDA")


def _args(**inference):
    base = dict(kv_cache_quantization="none", rollout_type="seg_ar", cfg_scale_video=1.0, camera_cfg_scale=1.0,
                sliding_window=SimpleNamespace(enabled=False), sparse_attention=SimpleNamespace(enabled=False))
    base.update(inference)
    return SimpleNamespace(inference=SimpleNamespace(**base), dit=SimpleNamespace(seg_ar_attention_backend="magi"),
                           parallelism=SimpleNamespace(sp_size=1))


@pytest.mark.parametrize("override, message", [
    (dict(kv_cache_quantization="int8"), "kv_cache_quantization"),
    (dict(sliding_window=SimpleNamespace(enabled=True)), "sliding_window"),
    (dict(sparse_attention=SimpleNamespace(enabled=True)), "sparse_attention"),
    (dict(cfg_scale_video=3.0), "cfg_scale_video"),
    (dict(rollout_type="chunk_ar"), "rollout_type"),
])
def test_fp8_mode_rejects_uncovered_configurations(override, message):
    with pytest.raises(ValueError, match=message):
        fk.validate_fp8_mode(_args(**override))


def test_fp8_cache_is_not_a_kv_quantization_option():
    with pytest.raises(ValueError):
        quantization_mode(SimpleNamespace(kv_cache_quantization="fp8"))


def test_verify_fp8_model_rejects_unquantized_projections():
    from hymm.models.dit.modules.fp8_linear import verify_fp8_model

    block = nn.Module()
    block.img_attn_q = nn.Linear(128, 128)
    model = nn.Module()
    model.double_blocks = nn.ModuleList([block])
    with pytest.raises(RuntimeError, match="not quantized"):
        verify_fp8_model(model)
    with pytest.raises(RuntimeError, match="no quantized projection"):
        verify_fp8_model(nn.Module())


@cuda
def test_quantize_uses_power_of_two_head_descales_and_saturates():
    torch.manual_seed(0)
    x = (torch.randn(64, 4, 128, device="cuda") * torch.tensor([0.3, 1.0, 20.0, 300.0], device="cuda")[None, :, None]).bfloat16()
    descale = fk.head_descale(x)
    assert descale.shape == (4,) and torch.equal(descale, torch.exp2(torch.log2(descale).round()))
    restored = fk.quantize(x, descale).float() * descale.view(-1, 1)
    error = ((restored - x.float()).square().mean((0, 2)) / x.float().square().mean((0, 2))).sqrt()
    assert error.max() < 0.05
    clipped = fk.quantize(x * 8, descale).float()
    assert torch.isfinite(clipped).all() and clipped.abs().max() <= fk.FP8_MAX


@cuda
def test_cache_descales_freeze_with_the_first_segment():
    torch.manual_seed(0)
    cache = {name: torch.empty((1, 96, 2, 128), device="cuda", dtype=fk.FP8) for name in "kv"}
    assert cache_format(cache) == "fp8"
    first_k, first_v = (torch.randn(1, 32, 2, 128, device="cuda").bfloat16() for _ in range(2))
    full_k, full_v = fk.write_current(cache, first_k, first_v, 0)
    assert full_k.shape[1] == 32 and full_k.dtype == fk.FP8
    frozen = cache[fk.K_DESCALE].clone(), cache[fk.V_DESCALE].clone()
    # A later, larger segment reuses the frozen descales and saturates instead of rescaling history.
    full_k, full_v = fk.write_current(cache, first_k * 1000, first_v * 1000, 32)
    assert full_k.shape[1] == 64
    assert torch.equal(cache[fk.K_DESCALE], frozen[0]) and torch.equal(cache[fk.V_DESCALE], frozen[1])
    assert torch.isfinite(full_v.float()).all()
    history = full_k[:, :32].float() * frozen[0].view(-1, 1)
    assert ((history - first_k.float()).square().mean() / first_k.float().square().mean()).sqrt() < 0.05
    with pytest.raises(ValueError, match="no room"):
        fk.write_current(cache, first_k, first_v, 80)
    # An emptied cache derives fresh descales.
    fk.write_current(cache, first_k * 1000, first_v * 1000, 0)
    assert not torch.equal(cache[fk.K_DESCALE], frozen[0])
