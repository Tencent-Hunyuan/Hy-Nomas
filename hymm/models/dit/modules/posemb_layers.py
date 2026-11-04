import torch
from typing import Union, Tuple, List


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2, device=None):
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (np.ndarray): [dim, ...]
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32, device=device)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [T, W, H]
    grid = torch.stack(grid, dim=0)  # [dim, T, W, H]

    return grid


#################################################################################
#                   Rotary Positional Embedding Functions                       #
#################################################################################
# https://github.com/meta-llama/llama/blob/be327c427cc5e89cc1d3ab3d3fec4484df771245/llama/model.py#L80


def reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    x: torch.Tensor,
    head_first=False,
):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Notes:
        When using FlashMHAModified, head_first should be False.
        When using Attention, head_first should be True.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis[0].shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = (
        x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    )  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (torch.Tensor or tuple): Precomputed frequency tensor for complex exponential.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], -1, 2)
        )  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(
            xq.device
        )  # [S, D//2] --> [1, S, 1, D//2]
        # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
        # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(
            xk.float().reshape(*xk.shape[:-1], -1, 2)
        )  # [B, S, H, D//2]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
    device=None,
):
    """
    This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
            part and an imaginary part separately.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.

    Returns:
        pos_embed (torch.Tensor): [HW, D/2]
    """

    grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list), device=device
    )  # [3, T, W, H] / [2, W, H]

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(rope_dim_list), (
        "len(theta_rescale_factor) should equal to len(rope_dim_list)"
    )

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(rope_dim_list), (
        "len(interpolation_factor) should equal to len(rope_dim_list)"
    )

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
            device=device,
        )  # 2 x [TWH, rope_dim_list[i]]
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)  # (TWH, D)
        sin = torch.cat([emb[1] for emb in embs], dim=1)  # (TWH, D)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)  # (TWH, D//2)
        return emb


def get_teacher_forcing_temporal_positions(
    num_ar_chunks: int,
    chunk_size: int,
    num_repeat: int = 1,
) -> torch.Tensor:
    """
    Compute temporal positions for teacher forcing training following LaCT approach.

    LaCT-style RoPE: Both noisy and clean chunks at the same logical chunk index
    share the SAME RoPE temporal positions.

    For structure [noisy_0_r0, noisy_0_r1, clean_0, noisy_1_r0, noisy_1_r1, clean_1, ...]:
    - noisy_0_r0 (frames 0,1,2 in sequence) -> RoPE positions [0, 1, 2]
    - noisy_0_r1 (frames 3,4,5 in sequence) -> RoPE positions [0, 1, 2] (SAME!)
    - clean_0    (frames 6,7,8 in sequence) -> RoPE positions [0, 1, 2] (SAME!)
    - noisy_1_r0 (frames 9,10,11 in sequence) -> RoPE positions [3, 4, 5]
    - noisy_1_r1 (frames 12,13,14 in sequence) -> RoPE positions [3, 4, 5] (SAME!)
    - clean_1    (frames 15,16,17 in sequence) -> RoPE positions [3, 4, 5] (SAME!)

    Args:
        num_ar_chunks: Number of AR chunks (e.g., 7 for 21 latent frames with window=3)
        chunk_size: Frames per AR chunk (e.g., 3)
        num_repeat: Number of noise repetitions per chunk (e.g., 1 or 2)

    Returns:
        temporal_positions: [total_interleaved_frames] tensor of RoPE temporal positions
            where total_interleaved_frames = num_ar_chunks * (num_repeat + 1) * chunk_size
    """
    chunks_per_window = num_repeat + 1  # noisy repeats + 1 clean
    total_frames = num_ar_chunks * chunks_per_window * chunk_size

    # Create temporal positions with repeated indices
    temporal_positions = torch.zeros(total_frames, dtype=torch.float32)

    for chunk_idx in range(num_ar_chunks):
        # Base RoPE position for this logical chunk
        base_pos = chunk_idx * chunk_size

        # Each window has (num_repeat + 1) sub-chunks, all sharing same RoPE positions
        window_start = chunk_idx * chunks_per_window * chunk_size

        for sub_chunk in range(chunks_per_window):
            chunk_start = window_start + sub_chunk * chunk_size
            for frame in range(chunk_size):
                temporal_positions[chunk_start + frame] = base_pos + frame

    return temporal_positions


def get_nd_rotary_pos_embed_with_temporal_positions(
    rope_dim_list: List[int],
    temporal_positions: torch.Tensor,
    spatial_sizes: Tuple[int, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
):
    """
    Compute 3D RoPE with custom temporal positions for teacher forcing.

    This is similar to get_nd_rotary_pos_embed but uses provided temporal positions
    instead of computing them from a meshgrid. This enables LaCT-style RoPE where
    noisy and clean chunks share the same temporal positions.

    Args:
        rope_dim_list: [temporal_dim, height_dim, width_dim] for RoPE
        temporal_positions: [num_temporal_frames] custom temporal positions
        spatial_sizes: (H, W) spatial dimensions after patch embedding
        theta: RoPE theta parameter
        use_real: Return real cos/sin (True) or complex (False)
        theta_rescale_factor: Per-dimension theta rescale factors
        interpolation_factor: Per-dimension interpolation factors

    Returns:
        If use_real: (freqs_cos, freqs_sin) each of shape [T*H*W, D]
        Else: freqs_cis of shape [T*H*W, D//2]
    """
    num_frames = len(temporal_positions)
    h, w = spatial_sizes

    # Handle theta_rescale_factor
    if isinstance(theta_rescale_factor, (int, float)):
        theta_rescale_factor = [theta_rescale_factor] * 3
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = theta_rescale_factor * 3

    # Handle interpolation_factor
    if isinstance(interpolation_factor, (int, float)):
        interpolation_factor = [interpolation_factor] * 3
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = interpolation_factor * 3

    # Create full position grid
    # temporal: use provided custom positions (repeated for each spatial position)
    # spatial: use standard meshgrid
    device = temporal_positions.device

    # Temporal positions: [T] -> [T*H*W] by repeating for each spatial position
    # For each temporal frame, we have H*W spatial positions
    temporal_grid = temporal_positions.unsqueeze(1).unsqueeze(2)  # [T, 1, 1]
    temporal_grid = temporal_grid.expand(num_frames, h, w).reshape(-1)  # [T*H*W]

    # Spatial positions: standard meshgrid (on same device as temporal_positions)
    h_positions = torch.arange(h, dtype=torch.float32, device=device)
    w_positions = torch.arange(w, dtype=torch.float32, device=device)
    h_grid, w_grid = torch.meshgrid(
        h_positions, w_positions, indexing="ij"
    )  # [H, W] each

    # Expand spatial to match temporal
    h_grid = h_grid.unsqueeze(0).expand(num_frames, -1, -1).reshape(-1)  # [T*H*W]
    w_grid = w_grid.unsqueeze(0).expand(num_frames, -1, -1).reshape(-1)  # [T*H*W]

    # Compute RoPE for each dimension
    embs = []
    grids = [temporal_grid, h_grid, w_grid]

    for i, (dim, grid) in enumerate(zip(rope_dim_list, grids)):
        emb = get_1d_rotary_pos_embed(
            dim,
            grid,
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)  # (T*H*W, D)
        sin = torch.cat([emb[1] for emb in embs], dim=1)  # (T*H*W, D)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)  # (T*H*W, D//2)
        return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
    device=None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Precompute the frequency tensor for complex exponential (cis) with given dimensions.
    (Note: `cis` means `cos + i * sin`, where i is the imaginary unit.)

    This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (int or torch.FloatTensor): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.
        theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.

    Returns:
        freqs_cis: Precomputed frequency tensor with complex exponential. [S, D/2]
        freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
    """
    if isinstance(pos, int):
        pos = torch.arange(pos, device=device).float()

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    # Compute frequency tensor on same device as pos
    device = pos.device if isinstance(pos, torch.Tensor) else None
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )  # [D/2]
    # assert interpolation_factor == 1.0, f"interpolation_factor: {interpolation_factor}"
    freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )  # complex64     # [S, D/2]
        return freqs_cis


def get_rotary_pos_embed_with_offset(
    rope_sizes,
    time_offset: int,
    hidden_size: int,
    heads_num: int,
    rope_dim_list=None,
    rope_theta: float = 10000.0,
    device=None,
):
    """
    Get rotary position embeddings with time offset for causal generation.

    When generating block N, the time positions should be
    [N*block_size, N*block_size+1, ...] instead of [0, 1, 2, ...].
    """
    tt, th, tw = rope_sizes
    target_ndim = 3
    head_dim = hidden_size // heads_num

    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]

    start = (time_offset, 0, 0)
    stop = (time_offset + tt, th, tw)

    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        start,
        stop,
        theta=rope_theta,
        use_real=True,
        theta_rescale_factor=1,
        device=device,
    )
    return freqs_cos, freqs_sin


def get_packed_rotary_pos_embed(
    rope_dim_list: List[int],
    sample_shapes: List[Tuple[int, int, int]],
    patch_size: Tuple[int, int, int] = (1, 1, 1),
    theta: float = 10000.0,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate concatenated RoPE for a token-packed sequence.

    Each sample in the packed batch may have a different (T, H, W).  This
    function computes independent 3-D RoPE for every sample and concatenates
    them along the token dimension so that each sample's tokens receive the
    correct positional encoding regardless of packing order.

    Args:
        rope_dim_list: Per-axis RoPE dimensions, e.g. [16, 56, 56].
        sample_shapes: List of (T, H, W) **latent** shapes for each packed
            sample.  These are in latent-frame units (before patch embedding).
        patch_size: (pt, ph, pw) — temporal and spatial patch sizes.  Used to
            convert latent shapes to patched-token counts.
        theta: RoPE frequency scaling factor.
        device: Target device for the output tensors.

    Returns:
        (freqs_cos, freqs_sin): Each of shape ``[L_total, D]`` where
        ``L_total = sum(tt_i * th_i * tw_i)`` over all samples.
    """
    pt, ph, pw = patch_size
    cos_parts: List[torch.Tensor] = []
    sin_parts: List[torch.Tensor] = []

    for T, H, W in sample_shapes:
        tt = T // pt
        th = H // ph
        tw = W // pw
        cos_i, sin_i = get_nd_rotary_pos_embed(
            rope_dim_list,
            (tt, th, tw),
            theta=theta,
            use_real=True,
            theta_rescale_factor=1,
            device=device,
        )
        cos_parts.append(cos_i)
        sin_parts.append(sin_i)

    return torch.cat(cos_parts, dim=0), torch.cat(sin_parts, dim=0)
