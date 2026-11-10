import torch
import pytest
from hymm.models.dit.modules.fp8_linear import FP8Linear, quantize_activation, quantize_weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('m', [1, 17, 256])
def test_fp8_linear(m):
    torch.manual_seed(42)
    original = torch.nn.Linear(512, 256, device='cuda', dtype=torch.bfloat16).eval()
    module = FP8Linear(original)
    module.weight_q, module.weight_scale = quantize_weight(original.weight.detach())
    x = torch.randn(m, 512, device='cuda', dtype=torch.bfloat16)
    with torch.inference_mode():
        # Separate bias avoids the environment's broken cuBLASLt bias epilogue.
        expected = torch.nn.functional.linear(x, original.weight) + original.bias
        actual = module(x)
    relative_rmse = (actual.float()-expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
    assert relative_rmse < 0.05
    assert torch.isfinite(actual).all()


def test_nonfinite_weight_rejected():
    with pytest.raises(ValueError, match='Nonfinite'):
        quantize_weight(torch.full((128, 128), float('nan')))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_activation_zero_and_outlier():
    x = torch.zeros(2, 256, device='cuda', dtype=torch.bfloat16)
    x[1, 0] = 1024
    x[1, 128:] = 0.125
    q, s = quantize_activation(x)
    restored = q.float().reshape(2, 2, 128) * s[:, :, None]
    torch.testing.assert_close(restored.reshape_as(x), x.float())
    assert torch.isfinite(s).all() and (s > 0).all()


@pytest.mark.parametrize('m,k', [(1,128), (17,384), (256,2048), (4097,2048)])
def test_packed_activation(m, k):
    import deep_gemm
    from hymm.models.dit.modules.fp8_linear import quantize_activation_packed
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    q, s = quantize_activation(x)
    packed_q, packed_s = quantize_activation_packed(x)
    torch.testing.assert_close(q.float(), packed_q.float(), rtol=0, atol=0)
    expected = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(s)
    torch.testing.assert_close(packed_s, expected, rtol=0, atol=0)


def test_compile_and_tp_shard():
    original = torch.nn.Linear(512, 512, device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    module = FP8Linear(original)
    module.weight_q, module.weight_scale = quantize_weight(original.weight)
    module.shard_rows(1, 2)
    module.prepare_for_inference()
    x = torch.randn(1, 17, 512, device='cuda', dtype=torch.bfloat16)
    compiled = torch.compile(module, fullgraph=True)
    with torch.inference_mode():
        expected = module(x)
        actual = compiled(x)
    torch.testing.assert_close(actual, expected)
    reference = torch.nn.functional.linear(x, original.weight[256:]) + original.bias[256:]
    assert ((actual.float()-reference.float()).square().mean()/reference.float().square().mean()).sqrt() < .05


@pytest.mark.parametrize('input_dtype', [torch.bfloat16, torch.float32])
def test_fused_qkv_matches_separate(input_dtype):
    from hymm.models.dit.modules.fp8_linear import prepare_fp8_qkv
    from hymm.models.dit.modules.nomas_compile_prepost import _qkv
    block=torch.nn.Module()
    for stream in ('img','txt'):
        for kind in 'qkv':
            original=torch.nn.Linear(512,256,device='cuda',dtype=torch.bfloat16).requires_grad_(False)
            layer=FP8Linear(original)
            layer.weight_q,layer.weight_scale=quantize_weight(original.weight)
            layer.prepare_for_inference()
            setattr(block,f'{stream}_attn_{kind}',layer)
    x=torch.randn(1,17,512,device='cuda',dtype=input_dtype)
    q,k,v=[getattr(block,'img_attn_'+kind) for kind in 'qkv']
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected=[layer(x) for layer in (q,k,v)]
        prepare_fp8_qkv(block)
        args=(x,q,q.bias,k,k.bias,v,v.bias)
        actual=torch.compile(_qkv,fullgraph=True)(*args)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=.02,atol=.02)


@pytest.mark.parametrize('tp_size', [2, 4])
def test_fused_qkv_shards_match_full_projection(tp_size):
    import copy
    from hymm.models.dit.modules.fp8_linear import prepare_fp8_qkv
    block=torch.nn.Module()
    for stream in ('img','txt'):
        for kind in 'qkv':
            original=torch.nn.Linear(2048,2048,device='cuda',dtype=torch.bfloat16).requires_grad_(False)
            layer=FP8Linear(original)
            layer.weight_q,layer.weight_scale=quantize_weight(original.weight)
            layer.prepare_for_inference()
            setattr(block,f'{stream}_attn_{kind}',layer)
    x=torch.randn(1,257,2048,device='cuda',dtype=torch.bfloat16)
    with torch.inference_mode():
        prepare_fp8_qkv(block)
        full=block.img_attn_q.forward_qkv(x,block.img_attn_k,block.img_attn_v)
        parts=[]
        for rank in range(tp_size):
            shard=copy.deepcopy(block)
            for stream in ('img','txt'):
                for kind in 'qkv':
                    layer=getattr(shard,f'{stream}_attn_{kind}')
                    layer.shard_rows(rank,tp_size)
                    layer.prepare_for_inference()
            prepare_fp8_qkv(shard)
            parts.append(shard.img_attn_q.forward_qkv(x,shard.img_attn_k,shard.img_attn_v))
        for index,expected in enumerate(full):
            actual=torch.cat([part[index] for part in parts],dim=-1)
            relative_rmse=((actual.float()-expected.float()).square().mean()/expected.float().square().mean()).sqrt()
            assert relative_rmse < .003


@pytest.mark.parametrize('fused_gelu', ['0', '1'])
def test_cudagraph_post_uses_current_weights_and_inputs(monkeypatch, fused_gelu):
    from hymm.models.dit.modules.tensor_parallel import _maybe_compiled, _post_attention
    monkeypatch.setenv('HY_DIT_FP8_CUDAGRAPHS','1')
    monkeypatch.setenv('HY_DIT_FP8_FUSED_GELU',fused_gelu)
    _maybe_compiled.cache_clear()
    def make(n,k):
        original=torch.nn.Linear(k,n,device='cuda',dtype=torch.bfloat16).requires_grad_(False)
        layer=FP8Linear(original)
        layer.weight_q,layer.weight_scale=quantize_weight(original.weight)
        layer.prepare_for_inference()
        return layer
    weights=[(make(256,256),make(512,256),make(256,512)) for _ in range(2)]
    fn=_maybe_compiled('post',True,True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for length in (127,256,127,255):
            torch.compiler.cudagraph_mark_step_begin()
            x=torch.randn(1,length,256,device='cuda',dtype=torch.bfloat16)
            attn=torch.randn_like(x)
            params=[torch.randn(1,256,device='cuda',dtype=torch.bfloat16) for _ in range(4)]
            for out,fc1,fc2 in weights:
                args=(x,attn,out,out.bias,*params[:3],fc1,fc1.bias,fc2,fc2.bias,params[3])
                reference=_post_attention(*args).float()
                actual=fn(*args).float().clone()
                error=((actual-reference).square().mean()/reference.square().mean()).sqrt()
                assert error < .015
    _maybe_compiled.cache_clear()
