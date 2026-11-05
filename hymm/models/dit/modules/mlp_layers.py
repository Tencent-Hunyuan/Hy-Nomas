# Modified from timm library:
# https://github.com/huggingface/pytorch-image-models/blob/648aaa41233ba83eb38faf5ba9d415d574823241/timm/layers/mlp.py#L13

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modulate_layers import modulate
from hymm.utils.helpers import to_2tuple


def fixed_m_linear(x: torch.Tensor, linear: nn.Linear, tile_m: int) -> torch.Tensor:
    """Apply a Linear with a fixed token-row GEMM shape.

    Splitting every caller into the same ``M=tile_m`` shape makes cuBLAS kernel
    selection independent of the local SP text length. A short final tile is
    zero-padded and sliced back after the Linear.
    """
    tile_m = int(tile_m)
    if tile_m <= 0:
        return linear(x)
    if x.ndim < 2:
        raise ValueError(f"fixed_m_linear expects ndim>=2, got shape={tuple(x.shape)}")

    outputs = []
    for chunk in torch.split(x, tile_m, dim=-2):
        valid_m = chunk.shape[-2]
        if valid_m < tile_m:
            chunk = F.pad(chunk, (0, 0, 0, tile_m - valid_m))
        output = linear(chunk)
        outputs.append(output[..., :valid_m, :])
    return torch.cat(outputs, dim=-2)


class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(
            in_channels, hidden_channels, bias=bias[0], **factory_kwargs
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_channels, **factory_kwargs)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = linear_layer(
            hidden_channels, out_features, bias=bias[1], **factory_kwargs
        )
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LinearWarpforSingle(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias, **factory_kwargs)

    def forward(self, x, y):
        input = torch.cat([x.contiguous(), y.contiguous()], dim=2).contiguous()
        return self.fc(input)


#
class MLPEmbedder(nn.Module):
    """copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py"""

    def __init__(self, in_dim: int, hidden_dim: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True, **factory_kwargs)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(
        self, hidden_size, patch_size, out_channels, act_layer, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Just use LayerNorm for the final layer
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        if isinstance(patch_size, int):
            self.linear = nn.Linear(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
                **factory_kwargs,
            )
        else:
            self.linear = nn.Linear(
                hidden_size,
                patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
                bias=True,
            )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Here we don't distinguish between the modulate types. Just use the simple one.
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c, tokens_per_frame: int = None, token_start_idx: int = 0):
        """
        Forward pass with optional per-frame modulation.

        Args:
            x: Input tensor [B, L, D] where L may be partial (sequence parallel)
            c: Condition tensor [B, C] or [B, T, C] (per-frame)
            tokens_per_frame: Required if c is [B, T, C], specifies tokens per frame
            token_start_idx: Starting token index for this shard (for sequence parallel)

        Returns:
            Output tensor [B, L, out_dim]
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x


class FinalLayerPerBlock(nn.Module):
    """The final layer of DiT."""

    def __init__(
        self, hidden_size, patch_size, out_channels, act_layer, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Just use LayerNorm for the final layer
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        if isinstance(patch_size, int):
            self.linear = nn.Linear(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
                **factory_kwargs,
            )
        else:
            self.linear = nn.Linear(
                hidden_size,
                patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
                bias=True,
            )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Here we don't distinguish between the modulate types. Just use the simple one.
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c, tokens_per_frame: int = None, token_start_idx: int = 0):
        """
        Forward pass with optional per-frame modulation.

        Args:
            x: Input tensor [B, L, D] where L may be partial (sequence parallel)
            c: Condition tensor [B, C] or [B, L, C] (per-token) or [B, T, C] (per-frame)
            tokens_per_frame: Required if c is [B, T, C] (per-frame), specifies tokens per frame.
                              Not needed when c is already per-token (c.shape[1] == x.shape[1]).
            token_start_idx: Starting token index for this shard (for sequence parallel)

        Returns:
            Output tensor [B, L, out_dim]
        """
        c_is_3d = len(c.shape) == 3

        if c_is_3d:
            B, T, C = c.shape
            L_local = x.shape[1]

            c_is_per_token = T == L_local

            if c_is_per_token:
                shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
                x_norm = self.norm_final(x)
                x = self.linear(x_norm * (1 + scale) + shift)
            else:
                c_flat = c.flatten(0, 1)  # [B*T, C]
                modulation = self.adaLN_modulation(c_flat)  # [B*T, 2*C]
                shift, scale = modulation.chunk(2, dim=1)  # each [B*T, C]
                shift = shift.unflatten(0, (B, T))  # [B, T, C]
                scale = scale.unflatten(0, (B, T))  # [B, T, C]

                local_token_indices = (
                    torch.arange(L_local, device=x.device) + token_start_idx
                )
                frame_indices = local_token_indices // tokens_per_frame  # [L_local]

                shift_per_token = shift[:, frame_indices, :]  # [B, L_local, C]
                scale_per_token = scale[:, frame_indices, :]  # [B, L_local, C]

                x_norm = self.norm_final(x)
                x = self.linear(x_norm * (1 + scale_per_token) + shift_per_token)
        else:
            # Original path: c is [B, C]
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift=shift, scale=scale)
            x = self.linear(x)

        return x
