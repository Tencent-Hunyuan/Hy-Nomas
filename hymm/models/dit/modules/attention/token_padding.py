"""Restore padded attention rows in one coalesced pass, including zero padding."""
import torch
import triton
import triton.language as tl


def inverse_positions(positions, length, shared=None):
    if shared is not None and 'inverse_positions' in shared:
        return shared['inverse_positions']
    inverse=torch.full((length,),-1,device=positions.device,dtype=torch.int64)
    inverse[positions]=torch.arange(positions.numel(),device=positions.device)
    if shared is not None: shared['inverse_positions']=inverse
    return inverse


@triton.jit
def _pad(X, INVERSE, OUT, ELEMENTS:tl.constexpr, WIDTH:tl.constexpr, BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=index//WIDTH
    source=tl.load(INVERSE+row,index<ELEMENTS,other=-1)
    value=tl.load(X+source*WIDTH+index%WIDTH,(index<ELEMENTS)&(source>=0),other=0)
    tl.store(OUT+index,value,index<ELEMENTS)


def pad_tokens_coalesced(x,inverse,batch,length):
    width=x.shape[1]*x.shape[2]
    output=torch.empty((batch,length,*x.shape[1:]),device=x.device,dtype=x.dtype)
    _pad[(triton.cdiv(output.numel(),1024),)](x,inverse,output,output.numel(),width,1024)
    return output
