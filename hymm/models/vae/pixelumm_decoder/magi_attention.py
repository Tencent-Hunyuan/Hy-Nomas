"""Blackwell Magi inference for PixelUMM's tile-major local attention."""

import torch

_layouts = {}


def local_ranges(grid, tile, device):
    key = (grid, tile, str(device))
    if key in _layouts:
        return _layouts[key]
    frames, height, width = grid
    nh, nw = (height + tile - 1) // tile, (width + tile - 1) // tile
    offsets = [0]
    for ih in range(nh):
        for iw in range(nw):
            offsets.append(offsets[-1] + frames * min(tile, height - ih * tile)
                           * min(tile, width - iw * tile))
    queries, keys, merged, csr = [], [], [], [0]
    for ih in range(nh):
        for iw in range(nw):
            idx = ih * nw + iw
            qr = [offsets[idx], offsets[idx + 1]]
            ch, cw = min(max(ih, 1), max(nh - 2, 1)), min(max(iw, 1), max(nw - 2, 1))
            merged.append(qr)
            for kh in range(max(0, ch - 1), min(nh, ch + 2)):
                queries.append(qr)
                keys.append([offsets[kh * nw + max(0, cw - 1)],
                             offsets[kh * nw + min(nw, cw + 2)]])
            csr.append(len(keys))
    # Magi's merge IR retains the original relation count, with zero outer
    # ranges after the unique query groups and a padded CSR tail.
    merged.extend([[0, 0]] * (len(queries) - len(merged)))
    csr.extend([csr[-1]] * (len(queries) + 1 - len(csr)))
    tensors = tuple(torch.tensor(v, dtype=torch.int32, device=device)
                    for v in (queries, keys, merged, csr))
    result = tensors + (
        max(end - start for start, end in queries),
        max(end - start for start, end in keys),
    )
    _layouts[key] = result
    return result


@torch.library.custom_op("hymm_pixelumm_magi::attention", mutates_args=())
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              qr: torch.Tensor, kr: torch.Tensor, merged: torch.Tensor,
              csr: torch.Tensor, max_q: int, max_k: int) -> torch.Tensor:
    from magi_attention.kernel.cutedsl import flex_flash_attn_func
    from magi_attention.kernel.cutedsl.range_merge import RangeMergePlan

    if torch.is_grad_enabled():
        raise ValueError("PixelUMM Magi currently supports inference only")
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device)[0] != 10:
        raise ValueError("PixelUMM Magi ranges require a Blackwell SM100 GPU")
    if q.shape[0] != 1:
        raise ValueError("PixelUMM Magi local ranges currently require batch size 1")
    mask = torch.zeros(kr.shape[0], dtype=torch.int32, device=q.device)
    plan = RangeMergePlan(merged, kr, mask, csr)
    output, _ = flex_flash_attn_func(
        q[0], k[0], v[0], q_ranges=qr, k_ranges=kr,
        mask_types=mask, range_merge=plan, disable_fwd_atomic_reduction=True,
        max_seqlen_q=max_q, max_seqlen_k=max_k,
        pack_gqa=False, out_dtype=q.dtype,
    )
    return output.unsqueeze(0)


@attention.register_fake
def _attention_fake(q, k, v, qr, kr, merged, csr, max_q, max_k):
    return torch.empty_like(q)


@torch.library.custom_op("hymm_pixelumm_magi::dense", mutates_args=())
def dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    from magi_attention.kernel.cutedsl import flex_flash_attn_func
    if torch.is_grad_enabled():
        raise ValueError("PixelUMM Magi currently supports inference only")
    output, _ = flex_flash_attn_func(q, k, v)
    return output


@dense.register_fake
def _dense_fake(q, k, v):
    return torch.empty_like(q)
