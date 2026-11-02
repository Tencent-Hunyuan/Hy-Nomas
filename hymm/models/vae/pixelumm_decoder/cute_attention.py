"""Opt-in CuTe QK RMSNorm/RoPE preparation for dense SDPA and sparse STA.

Inference only. Uses validated kernels/qk_norm_rope.py kernels, preserves token-major
attention strides and FP32 QK normalization. STA keeps unexpanded K/V heads. Unsupported/low-parallelism inputs use the
original attention. Compilation and output allocation are included in model timing.
"""
from collections import Counter
from types import MethodType
import torch
from torch.nn import functional as F

_compiled={}
_calls=Counter()


def _launch(x,v,wt,wh,cos,sin,repeat,with_v,eps):
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda
    from .kernels.qk_norm_rope import QKNormRoPE
    b,n,h,d=x.shape
    if (d!=128 or n<=0 or b<=0 or not x.is_cuda or x.dtype!=torch.bfloat16
            or not x.is_contiguous() or v.shape!=x.shape or not v.is_contiguous()
            or v.dtype!=x.dtype or v.device!=x.device or repeat<1
            or any(t.dtype!=torch.bfloat16 or t.numel()!=64 or not t.is_contiguous()
                   or t.device!=x.device for t in (wt,wh))
            or any(t.dtype!=torch.float32 or t.numel()!=n*128 or not t.is_contiguous()
                   or t.device!=x.device for t in (cos,sin))):
        raise ValueError('Unsupported CuTe QK input: requires contiguous CUDA BF16 heads/weights and FP32 rotary tables')
    y=torch.empty((b,n,h*repeat,d),device=x.device,dtype=x.dtype)
    vo=torch.empty_like(y) if with_v else y
    stream=cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    args=[from_dlpack(t.reshape(-1),assumed_align=16) for t in (x,v,wt,wh,cos,sin,y,vo)]+[n,stream]
    key=(x.device.index,tuple(x.shape),repeat,with_v,eps)
    if key not in _compiled:
        _compiled[key]=cute.compile(QKNormRoPE(h,repeat,4,with_v,eps),*args)
    _compiled[key](*args)
    return y,vo


@torch.library.custom_op('hymm_pixelumm_cute::qkv_prepare',mutates_args=())
def qkv_prepare(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,qt:torch.Tensor,qh:torch.Tensor,
                kt:torch.Tensor,kh:torch.Tensor,cos:torch.Tensor,sin:torch.Tensor,eps:float,expand_kv:bool=True
                )->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    yq,_=_launch(q,q,qt,qh,cos,sin,1,False,eps)
    yk,yv=_launch(k,v,kt,kh,cos,sin,q.shape[2]//k.shape[2] if expand_kv else 1,True,eps)
    _calls['qkv_prepare']+=1
    return yq,yk,yv


@qkv_prepare.register_fake
def _fake(q,k,v,qt,qh,kt,kh,cos,sin,eps,expand_kv=True):
    return torch.empty_like(q),torch.empty_like(q if expand_kv else k),torch.empty_like(q if expand_kv else v)


def _supported(self,x,cos,sin):
    return (not self.training and not torch.is_grad_enabled() and x.ndim==3
            and x.dtype==torch.bfloat16 and x.device==self._cute_device
            and all(t.dtype==torch.float32 and t.device==x.device
                    and t.shape==(1,x.shape[1],1,128) and t.is_contiguous()
                    for t in (cos,sin)))


def _profitable(self,x):
    # Conservative occupancy floor, not a shape whitelist.
    return x.shape[0]*x.shape[1]*self.arch.num_kv_heads>=self._cute_min_rows


def _forward(self,x,cos,sin,block_mask=None):
    if not _supported(self,x,cos,sin) or not _profitable(self,x):
        return self._cute_original_forward(x,cos,sin,block_mask)
    arch=self.arch;b,n,_=x.shape
    q=self.q_proj(x).view(b,n,arch.num_heads,arch.head_dim)
    k=self.k_proj(x).view(b,n,arch.num_kv_heads,arch.head_dim)
    v=self.v_proj(x).view(b,n,arch.num_kv_heads,arch.head_dim)
    magi = self.attention_backend == "magi"
    q,k,v=qkv_prepare(q,k,v,self.q_norm_t.weight,self.q_norm_hw.weight,
                    self.k_norm_t.weight,self.k_norm_hw.weight,cos,sin,arch.rms_norm_eps,
                    block_mask is None and not magi)
    if magi:
        from .magi_attention import attention, dense
        out = attention(q,k,v,*block_mask) if block_mask is not None else dense(q,k,v)
        return self.o_proj(out.reshape(b,n,arch.num_heads*arch.head_dim))
    if block_mask is None:
        out=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
    else:
        from .model import _flex_attention
        out=_flex_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),block_mask)
    return self.o_proj(out.transpose(1,2).reshape(b,n,arch.num_heads*arch.head_dim))


def enable_cute_attention(decoder):
    """Install explicitly after eval/weight loading and before torch.compile."""
    from .model import DCAEVPixelUMMDecoder
    if type(decoder) is not DCAEVPixelUMMDecoder or decoder.training:
        raise ValueError('CuTe preparation requires an eval PixelUMM decoder')
    device=next(decoder.parameters()).device
    if device.type!='cuda' or torch.cuda.get_device_capability(device) not in {(9,0),(10,3)}:
        raise ValueError('CuTe preparation validated on SM90 and SM103 only')
    if decoder.arch.head_dim!=128 or decoder.arch.rope_sections!=(64,32,32):
        raise ValueError('CuTe preparation requires the PixelUMM 128-dim rotary layout')
    import cutlass.cute
    minimum=4*torch.cuda.get_device_properties(device).multi_processor_count
    for layer in decoder.layers:
        attn=layer.attn
        if hasattr(attn,'_cute_original_forward'):raise ValueError('CuTe already installed')
        for norm in (attn.q_norm_t,attn.q_norm_hw,attn.k_norm_t,attn.k_norm_hw):
            if norm.weight.dtype!=torch.bfloat16 or norm.eps!=decoder.arch.rms_norm_eps:
                raise ValueError('CuTe preparation requires BF16 weights and matching RMS epsilon')
    for layer in decoder.layers:
        attn=layer.attn
        attn._cute_original_forward=attn.forward
        attn._cute_device=device
        attn._cute_min_rows=minimum
        attn.forward=MethodType(_forward,attn)
    return len(decoder.layers)


def execution_stats():return dict(_calls)
