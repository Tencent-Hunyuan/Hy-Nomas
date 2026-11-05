import pytest
import torch

from hymm.models.dit.modules.attention.range_attention import pad_tokens, unpad_positions
from hymm.models.dit.modules.attention.token_padding import inverse_positions, pad_tokens_coalesced


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton padding runs on CUDA")
@pytest.mark.parametrize("batch,length,heads", [(1, 23280, 4), (2, 257, 2), (1, 7, 16)])
@pytest.mark.parametrize("density", [0.0, 0.6, 1.0])
def test_coalesced_padding_is_bitwise_equal(batch, length, heads, density):
    generator = torch.Generator(device="cuda").manual_seed(batch * length + heads)
    mask = torch.rand(batch, length, generator=generator, device="cuda") < density
    positions, _ = unpad_positions(mask)
    x = torch.randn(positions.numel(), heads, 128, generator=generator, device="cuda").bfloat16()
    shared = {}
    inverse = inverse_positions(positions, batch * length, shared)
    assert inverse_positions(positions, batch * length, shared) is inverse
    assert torch.equal(pad_tokens_coalesced(x, inverse, batch, length), pad_tokens(x, positions, batch, length))
