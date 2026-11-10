import torch
import pytest
from hymm.models.dit.modules.fp8_linear import FP8Linear, quantize_weight, quantize_activation_packed


@pytest.mark.parametrize('shape', [(17,384),(4097,2048),(5500,8192)])
@pytest.mark.parametrize('with_bias', [False,True])
def test_fused_gelu_quantization(shape,with_bias):
    torch.manual_seed(42)
    x=torch.randn(*shape,device='cuda',dtype=torch.bfloat16)*3
    bias=torch.randn(shape[1],device='cuda',dtype=torch.bfloat16) if with_bias else None
    q,s=quantize_activation_packed(x,bias=bias,gelu=True)
    y=torch.nn.functional.gelu(x if bias is None else x+bias,approximate='tanh')
    ref_q,ref_s=quantize_activation_packed(y)
    torch.testing.assert_close(s,ref_s)
    error=((q.float()-ref_q.float()).square().mean()/ref_q.float().square().mean()).sqrt()
    assert error < .003


def test_fused_mlp_compile(monkeypatch):
    from hymm.models.dit.modules.nomas_compile_prepost import _mlp
    monkeypatch.setenv('HY_DIT_FP8_FUSED_GELU','1')
    layers=[]
    for n,k in [(1024,512),(512,1024)]:
        original=torch.nn.Linear(k,n,device='cuda',dtype=torch.bfloat16).requires_grad_(False)
        layer=FP8Linear(original)
        layer.weight_q,layer.weight_scale=quantize_weight(original.weight)
        layer.prepare_for_inference()
        layers.append(layer)
    a,b=layers
    x=torch.randn(1,257,512,device='cuda',dtype=torch.float32)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        reference=b(torch.nn.functional.gelu(a(x),approximate='tanh'))
        actual=torch.compile(_mlp,fullgraph=True)(x,a,a.bias,b,b.bias)
    error=((actual.float()-reference.float()).square().mean()/reference.float().square().mean()).sqrt()
    assert error < .005
