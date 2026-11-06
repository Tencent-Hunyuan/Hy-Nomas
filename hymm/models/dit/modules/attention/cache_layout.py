"""Unpadding metadata shared only within one transformer forward.

Current-token visibility is identical in every block. History visibility is
layer-specific and is always recomputed. No metadata survives the forward.
"""
from .range_attention import unpad_positions


def _layout(mask):
    positions, cu_seqlens = unpad_positions(mask)
    maximum = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
    return positions, cu_seqlens, maximum


def attention_layouts(query_mask, key_mask, shared, *, same_keys=False):
    if shared is None:
        return _layout(query_mask), _layout(key_mask)
    geometry = (tuple(query_mask.shape), query_mask.device)
    if 'query' not in shared:
        shared['geometry'], shared['query'] = geometry, _layout(query_mask)
    elif shared['geometry'] != geometry:
        raise ValueError('Shared attention layout changed within one forward')
    query = shared['query']
    return query, query if same_keys else _layout(key_mask)
