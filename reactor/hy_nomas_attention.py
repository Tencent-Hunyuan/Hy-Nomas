"""Blackwell FA4 compatibility for native DiT ranges and PixelUMM local masks.

Only attention metadata is cached; Q/K/V and conditioning remain native-owned.
The native modules are configured inside the isolated model worker.
"""

from collections import OrderedDict

_metadata = OrderedDict()
_identities = OrderedDict()
_stats = {"calls": 0, "builds": 0}
_decoder_installed = False
_batched_ranges = OrderedDict()


def clear_fa4_cache() -> None:
    """Release range tensors and compiled-mask arguments at world reset."""
    _identities.clear()
    _metadata.clear()
    _batched_ranges.clear()


def fa4_attention(
    q,
    k,
    v,
    q_ranges,
    k_ranges,
    attn_type_map=None,
    *,
    softmax_scale=None,
    deterministic=False,
    **kwargs,
):
    import torch
    from magi_attention.common import AttnRanges
    from magi_attention.functional.fa4 import fa4_fwd
    from magi_attention.meta.collection.calc_meta import FA4AttnArg

    if deterministic or kwargs:
        raise ValueError(
            "FA4 requires nondeterministic inference without extra kernel flags"
        )
    if torch.is_grad_enabled():
        raise RuntimeError("Hy-Nomas FA4 adapter is inference-only")
    if not q.shape[0] or not k.shape[0]:
        return torch.zeros_like(q), None
    # Strong references prevent Python ID reuse; tensor versions invalidate
    # metadata if upstream mutates a range or attention type in place.
    tensors = (q_ranges, k_ranges, attn_type_map)
    identity = (
        q.shape[0],
        k.shape[0],
        str(q.device),
        tuple((id(t), t._version) if t is not None else None for t in tensors),
    )
    if identity in _identities:
        _identities.move_to_end(identity)
        arg = _identities[identity][1]
        _stats["calls"] += 1
        return fa4_fwd(
            q=q,
            k=k,
            v=v,
            sink=None,
            attn_arg=arg,
            softmax_scale=q.shape[-1] ** -0.5
            if softmax_scale is None
            else softmax_scale,
        )
    qr = tuple(map(tuple, q_ranges.detach().cpu().tolist()))
    kr = tuple(map(tuple, k_ranges.detach().cpu().tolist()))
    types = (
        (0,) * len(qr)
        if attn_type_map is None
        else tuple(attn_type_map.detach().cpu().tolist())
    )
    key = (q.shape[0], k.shape[0], str(q.device), qr, kr, types)
    if key not in _metadata:
        _metadata[key] = FA4AttnArg(
            q_ranges=AttnRanges.from_ranges(qr),
            k_ranges=AttnRanges.from_ranges(kr),
            attn_type_map=list(types),
            seqlen_q=q.shape[0],
            seqlen_k=k.shape[0],
        )
        _stats["builds"] += 1
        if len(_metadata) > 32:
            _metadata.popitem(last=False)
    _metadata.move_to_end(key)
    _identities[identity] = (tensors, _metadata[key])
    if len(_identities) > 64:
        _identities.popitem(last=False)
    _stats["calls"] += 1
    return fa4_fwd(
        q=q,
        k=k,
        v=v,
        sink=None,
        attn_arg=_metadata[key],
        softmax_scale=q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale,
    )


def install_fa4() -> None:
    """Route native range and cache-write attention to released FA4 kernels."""
    import sys
    import types
    from importlib.machinery import ModuleSpec

    from flash_attn_cute.interface import flash_attn_func, flash_attn_varlen_func
    from hymm.models.dit.modules.attention import backends
    from magi_attention import functional

    # Hy-Nomas calls the primitive directly. v1.1's environment flag only
    # changes distributed dispatch, so select the released FA4 primitive here.
    backends.magi_kernel = lambda: fa4_attention
    functional.flex_flash_attn_func = fa4_attention

    def varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs
    ):
        _stats["dense_calls"] = _stats.get("dense_calls", 0) + 1
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, **kwargs
        )[0]

    # The native cache-write path imports the FA3 interface by name.
    # Preserve its call signature while executing FA4 on Blackwell.
    interface = types.ModuleType("flash_attn_interface")
    interface.__spec__ = ModuleSpec("flash_attn_interface", loader=None)
    interface.flash_attn_func = lambda *args, **kwargs: flash_attn_func(
        *args, **kwargs
    )[0]
    interface.flash_attn_varlen_func = varlen
    sys.modules["flash_attn_interface"] = interface


def decoder_ranges(
    grid: tuple[int, int, int], tile: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Represent the native tile-major 3x3 spatial neighborhood as rectangles.

    Each key rectangle spans one tile row and all temporal frames. Boundary
    centers follow the native clamping rule, including partial edge tiles.
    """
    if tile <= 0 or len(grid) != 3 or min(grid) <= 0:
        raise ValueError("Decoder grid and tile dimensions must be positive.")
    frames, height, width = grid
    rows, columns = -(-height // tile), -(-width // tile)
    offsets = [0]
    for row in range(rows):
        for column in range(columns):
            offsets.append(
                offsets[-1]
                + frames
                * min(tile, height - row * tile)
                * min(tile, width - column * tile)
            )
    qr, kr = [], []
    for row in range(rows):
        for column in range(columns):
            center_row = max(1, min(row, max(rows - 2, 1)))
            center_column = max(1, min(column, max(columns - 2, 1)))
            q_index = row * columns + column
            for key_row in range(max(0, center_row - 1), min(rows, center_row + 2)):
                first = key_row * columns + max(0, center_column - 1)
                last = key_row * columns + min(columns, center_column + 2)
                qr.append((offsets[q_index], offsets[q_index + 1]))
                kr.append((offsets[first], offsets[last]))
    return qr, kr


def install_decoder_fa4() -> None:
    """Keep native permutations/local neighborhoods and replace their kernel."""
    global _decoder_installed
    if _decoder_installed:
        return
    import hymm.models.vae.pixelumm_decoder.model as native
    import torch

    original_layout = native.local_attention_layout
    layouts = OrderedDict()

    @torch.library.custom_op("hy_nomas_fa4::pixelumm_attention", mutates_args=())
    def run(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qr: torch.Tensor,
        kr: torch.Tensor,
        types: torch.Tensor,
    ) -> torch.Tensor:
        batch, heads, count, dim = q.shape
        if batch > 1:
            key = (tuple((id(t), t._version) for t in (qr, kr, types)), batch, count)
            if key not in _batched_ranges:
                offsets = (
                    torch.arange(batch, device=q.device, dtype=qr.dtype)[:, None, None]
                    * count
                )
                _batched_ranges[key] = (
                    qr,
                    kr,
                    types,
                    (qr[None] + offsets).flatten(0, 1),
                    (kr[None] + offsets).flatten(0, 1),
                    types.repeat(batch),
                )
                if len(_batched_ranges) > 8:
                    _batched_ranges.popitem(last=False)
            qr, kr, types = _batched_ranges[key][3:]
        out, _ = fa4_attention(
            q=q.transpose(1, 2).reshape(batch * count, heads, dim),
            k=k.transpose(1, 2).reshape(batch * count, k.shape[1], dim),
            v=v.transpose(1, 2).reshape(batch * count, v.shape[1], dim),
            q_ranges=qr,
            k_ranges=kr,
            attn_type_map=types,
        )
        return out.reshape(batch, count, heads, dim).transpose(1, 2).contiguous()

    @run.register_fake
    def fake(q, k, v, qr, kr, types):
        return q.new_empty(q.shape)

    def layout(grid, tile, device):
        key = (grid, tile, str(device))
        if key not in layouts:
            native_layout = original_layout(grid, tile, device)
            if native_layout is None:
                return None
            perm, inverse, _ = native_layout
            qr, kr = decoder_ranges(grid, tile)
            metadata = (
                torch.tensor(qr, dtype=torch.int32, device=device),
                torch.tensor(kr, dtype=torch.int32, device=device),
                torch.zeros(len(qr), dtype=torch.int32, device=device),
            )
            layouts[key] = perm, inverse, metadata
            if len(layouts) > 8:
                layouts.popitem(last=False)
        return layouts[key]

    native.local_attention_layout = layout
    native._flex_attention = lambda q, k, v, mask: run(q, k, v, *mask)
    from hymm.models.vae.pixelumm_decoder import autoencoder

    autoencoder.local_attention_layout = layout
    _decoder_installed = True
