"""Stage native BF16 cache attention without recopying the history prefix."""

from collections import OrderedDict

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["TOTAL", "VISIBLE", "IMAGE", "HISTORY"])
def _gather_pair(
    K,
    V,
    HISTORY_INDEX,
    TEXT_INDEX,
    OUT_K,
    OUT_V,
    TOTAL,
    WIDTH: tl.constexpr,
    VISIBLE,
    IMAGE,
    HISTORY,
    KEY_STRIDE: tl.constexpr,
    VALUE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    element = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = element // WIDTH, element % WIDTH
    history = tl.load(HISTORY_INDEX + row, (element < TOTAL) & (row < VISIBLE), other=0)
    text = tl.load(
        TEXT_INDEX + row - VISIBLE - IMAGE,
        (element < TOTAL) & (row >= VISIBLE + IMAGE),
        other=0,
    )
    source = tl.where(
        row < VISIBLE,
        history,
        tl.where(
            row < VISIBLE + IMAGE, HISTORY + row - VISIBLE, HISTORY + IMAGE + text
        ),
    )
    key = tl.load(K + source * KEY_STRIDE + column, element < TOTAL, other=0)
    value = tl.load(V + source * VALUE_STRIDE + column, element < TOTAL, other=0)
    tl.store(OUT_K + element, key, element < TOTAL)
    tl.store(OUT_V + element, value, element < TOTAL)


def compact_kv(key, value, history_index, text_index, history, image):
    """Gather the native [visible history | image | valid text] token order."""
    heads, dim = key.shape[-2:]
    count = history_index.numel() + image + text_index.numel()
    keys = torch.empty((count, heads, dim), dtype=key.dtype, device=key.device)
    values = torch.empty_like(keys)
    # INT32 is a transport type: each lane copies two BF16 bit patterns.
    total = keys.numel() // 2
    _gather_pair[(triton.cdiv(total, 1024),)](
        triton.reinterpret(key, tl.int32),
        triton.reinterpret(value, tl.int32),
        history_index,
        text_index,
        triton.reinterpret(keys, tl.int32),
        triton.reinterpret(values, tl.int32),
        total,
        heads * dim // 2,
        history_index.numel(),
        image,
        history,
        key.stride(0) // 2,
        value.stride(0) // 2,
        1024,
        num_warps=4,
    )
    return keys, values


class CacheStaging:
    """Reuse immutable visibility metadata; leave canonical cache ownership native."""

    def __init__(self):
        import hymm.models.dit.nomas_cpt as native
        from flash_attn_interface import flash_attn_varlen_func
        from hymm.models.dit.modules.attention.range_attention import (
            note_non_range_call,
        )
        from hymm.models.dit.modules.kv_cache_storage import cache_format

        self._native = native
        self._flash = flash_attn_varlen_func
        self._note_dense = note_non_range_call
        self._format = cache_format
        self._original = native.causal_attention_with_cache_mp
        self._visibility = OrderedDict()
        self._lengths = OrderedDict()
        self.stats = {
            "calls": 0,
            "fallbacks": 0,
            "visibility_builds": 0,
            "visibility_hits": 0,
        }
        native.causal_attention_with_cache_mp = self

    def reset(self):
        """Release metadata references without changing native K/V or its position."""
        self._visibility.clear()
        self._lengths.clear()

    def _visible(self, mask, count):
        identity = (id(mask), mask._version, count, str(mask.device))
        if identity not in self._visibility:
            boolean = mask.bool()
            indices = boolean[0, :count].nonzero(as_tuple=False).squeeze(-1)
            # Retaining the source prevents Python identity reuse; its version
            # invalidates both normalization and indices after any mutation.
            self._visibility[identity] = (mask, boolean, indices)
            self.stats["visibility_builds"] += 1
            if len(self._visibility) > 128:
                self._visibility.popitem(last=False)
        else:
            self.stats["visibility_hits"] += 1
        self._visibility.move_to_end(identity)
        return self._visibility[identity][1:]

    def _attend(
        self,
        query,
        key,
        value,
        history_index,
        text_index,
        history,
        image,
        offsets,
        windows,
    ):
        native = self._native
        keys, values = compact_kv(
            key[0], value[0], history_index, text_index, history, image
        )
        queries = torch.cat(
            (query[0, :image], query[0, image:][text_index]), dim=0
        ).contiguous()
        nq, nk = queries.shape[0], keys.shape[0]
        if offsets is None:
            identity = (nq, nk, str(query.device))
            if identity not in self._lengths:
                self._lengths[identity] = (
                    torch.tensor([0, nq], dtype=torch.int32, device=query.device),
                    torch.tensor([0, nk], dtype=torch.int32, device=query.device),
                )
                if len(self._lengths) > 32:
                    self._lengths.popitem(last=False)
            self._lengths.move_to_end(identity)
            cuq, cuk = self._lengths[identity]
            self._note_dense()
            output = self._flash(
                queries,
                keys,
                values,
                cuq,
                cuk,
                nq,
                nk,
                causal=False,
                deterministic=native._FLASH_CACHE_DETERMINISTIC,
            )
        else:
            valid = text_index.numel()
            qr, kr, types = native._build_intra_seg_cache_ranges(
                hvis=history_index.numel(),
                limg=image,
                valid=valid,
                offs=[min(int(x), valid) for x in offsets],
                windows=windows,
                device=query.device,
            )
            output, _ = native.range_attention(
                q=queries,
                k=keys,
                v=values,
                q_ranges=qr,
                k_ranges=kr,
                attn_type_map=types,
                backend="magi",
                softmax_scale=None,
                deterministic=native._INTRA_SEG_CACHE_DETERMINISTIC,
            )
        hidden = torch.zeros_like(query)
        hidden[0, :image] = output[:image]
        hidden[0, image:][text_index] = output[image:]
        return hidden

    def __call__(
        self,
        q,
        k,
        v,
        kv_cache,
        cache_valid_len=0,
        attn_mode=None,
        text_mask=None,
        update_cache=False,
        cache_txt=True,
        cache_mask=None,
        intra_seg_txt_offsets=None,
        intra_seg_vid_windows=None,
        tokens_per_frame=0,
        seg_ar_attention_backend="magi",
        sparse_context=None,
    ):
        native = self._native
        eligible = (
            not torch.is_grad_enabled()
            and q[0].shape[0] == 1
            and cache_txt
            and self._format(kv_cache) == "none"
            and "swa_capacity" not in kv_cache
            and kv_cache["k"].dtype == kv_cache["v"].dtype == torch.bfloat16
            and kv_cache["k"].shape[0] == 1
            and sparse_context is None
            and attn_mode == "flash3"
            and seg_ar_attention_backend == "magi"
            and cache_mask is not None
            and not cache_mask.is_inference()
            and (text_mask is None or not text_mask.is_inference())
            and (intra_seg_txt_offsets is None or len(intra_seg_txt_offsets) <= 1)
            and kv_cache["k"].device == kv_cache["v"].device
            and all(
                x.shape[-1] % 2 == 0
                and x.stride(-1) == 1
                and x.stride(-2) == x.shape[-1]
                and x.stride(1) % 2 == 0
                for x in (kv_cache["k"], kv_cache["v"])
            )
            and all(
                x.device == kv_cache["k"].device for pair in (q, k, v) for x in pair
            )
        )
        if not eligible:
            self.stats["fallbacks"] += 1
            return self._original(
                q,
                k,
                v,
                kv_cache,
                cache_valid_len,
                attn_mode,
                text_mask,
                update_cache,
                cache_txt,
                cache_mask,
                intra_seg_txt_offsets,
                intra_seg_vid_windows,
                tokens_per_frame,
                seg_ar_attention_backend,
                sparse_context,
            )
        img_q, txt_q = q
        img_k, txt_k = k
        img_v, txt_v = v
        img_dtype, txt_dtype = img_q.dtype, txt_q.dtype
        parallel = native.get_sequence_parallel_state()
        group = native.nccl_info.sp_group
        if parallel:
            img_q = native.all_to_all_4D(
                img_q, group=group, scatter_dim=2, gather_dim=1
            )
            img_k = native.all_to_all_4D(
                img_k, group=group, scatter_dim=2, gather_dim=1
            )
            img_v = native.all_to_all_4D(
                img_v, group=group, scatter_dim=2, gather_dim=1
            )
            if txt_q.shape[2] != img_q.shape[2]:
                heads = img_q.shape[2]
                start = native.nccl_info.rank_within_spgroup * heads
                txt_q, txt_k, txt_v = (
                    x[:, :, start : start + heads] for x in (txt_q, txt_k, txt_v)
                )
        img_q, txt_q, img_k, txt_k, img_v, txt_v = (
            x.to(dtype=torch.bfloat16)
            for x in (img_q, txt_q, img_k, txt_k, img_v, txt_v)
        )
        image, text = img_k.shape[1], txt_k.shape[1]
        total = cache_valid_len + image + text
        if total > kv_cache["k"].shape[1]:
            raise ValueError(
                "Native KV cache capacity is smaller than the current segment"
            )
        _, history_index = self._visible(cache_mask, cache_valid_len)
        if text_mask is None:
            text_mask = torch.ones(1, text, dtype=torch.bool, device=img_q.device)
        text_valid, text_index = self._visible(text_mask, text)
        # The native valid prefix remains intact. Unused tail rows stage each
        # denoising step; the native write pass commits identical values/masks.
        for name, image_kv, text_kv in (("k", img_k, txt_k), ("v", img_v, txt_v)):
            target = kv_cache[name]
            target[:, cache_valid_len : cache_valid_len + image].copy_(image_kv)
            target[:, cache_valid_len + image : total].copy_(text_kv)
        if update_cache:
            cache_mask[:, cache_valid_len : cache_valid_len + image] = True
            cache_mask[:, cache_valid_len + image : total].copy_(text_valid)
        intra = (
            intra_seg_vid_windows is not None
            and len(intra_seg_vid_windows) > 0
            and intra_seg_txt_offsets is not None
            and any(x is not None for x in intra_seg_txt_offsets)
        )
        offsets, windows = None, None
        if intra:
            offsets = intra_seg_txt_offsets[0]
            per_frame = int(tokens_per_frame) if tokens_per_frame > 0 else 1
            windows = [
                [int(w[0]) * per_frame, int(w[1]) * per_frame, int(w[2])]
                for w in intra_seg_vid_windows
            ]
        hidden = self._attend(
            torch.cat((img_q, txt_q), dim=1),
            kv_cache["k"][:, :total],
            kv_cache["v"][:, :total],
            history_index,
            text_index,
            cache_valid_len,
            image,
            offsets,
            windows,
        )
        img_attn, txt_attn = hidden[:, :image], hidden[:, image:]
        if parallel:
            img_attn = native.all_to_all_4D(
                img_attn.contiguous(), group=group, scatter_dim=1, gather_dim=2
            )
            txt_attn = native.all_gather(txt_attn.contiguous(), dim=2, group=group)
        img_attn, txt_attn = img_attn.to(img_dtype), txt_attn.to(txt_dtype)
        self.stats["calls"] += 1
        return (
            img_attn.reshape(img_attn.shape[0], img_attn.shape[1], -1),
            txt_attn.reshape(txt_attn.shape[0], txt_attn.shape[1], -1),
        )
