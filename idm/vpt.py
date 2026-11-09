"""VPT inference components for Hy-Nomas IDM."""
import sys
from pathlib import Path
import torch as th
import torch.nn as nn
FRAMES_PER_BUCKET = 6
VPT_ROOT = Path(__file__).resolve().parents[1] / 'vendor' / 'video-pre-training'

def _lazy_imports():
    """Import the bundled VPT runtime without colliding with other lib packages."""
    import sys, types, importlib
    if 'minerl' not in sys.modules:
        pkg = types.ModuleType('minerl')
        pkg.__path__ = []
        sys.modules['minerl'] = pkg
        for sub in ('minerl.herobraine', 'minerl.herobraine.hero', 'minerl.herobraine.hero.mc'):
            m = types.ModuleType(sub)
            m.__path__ = []
            sys.modules[sub] = m
        sys.modules['minerl.herobraine.hero.mc'].MINERL_ITEM_MAP = {}
    saved = {k: sys.modules[k] for k in list(sys.modules) if k == 'lib' or k.startswith('lib.')}
    for k in saved:
        del sys.modules[k]
    if sys.path[0] != str(VPT_ROOT):
        try:
            sys.path.remove(str(VPT_ROOT))
        except ValueError:
            pass
        sys.path.insert(0, str(VPT_ROOT))
    try:
        DictType = importlib.import_module('lib.gym3_types').DictType
        vpt_policy = importlib.import_module('lib.policy')
        vpt_action_mapping = importlib.import_module('lib.action_mapping')
        InverseActionPolicy = vpt_policy.InverseActionPolicy
        IDMActionMapping = vpt_action_mapping.IDMActionMapping
    finally:
        for k in list(sys.modules):
            if k == 'lib' or k.startswith('lib.'):
                del sys.modules[k]
        sys.modules.update(saved)
    return (DictType, InverseActionPolicy, IDMActionMapping)

def _make_vpt_policy(net_kwargs: dict, pi_head_kwargs: dict):
    """Construct a VPT inverse action policy."""
    DictType, InverseActionPolicy, IDMActionMapping = _lazy_imports()
    action_mapper = IDMActionMapping(n_camera_bins=11)
    action_space = action_mapper.get_action_space_update()
    action_space = DictType(**action_space)
    policy = InverseActionPolicy(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=action_space)
    return policy

class VPTIDMWrapper(nn.Module):
    """Encode RGB video frames and predict per-bin actions and auxiliary outputs."""

    def __init__(self, policy, n_label_dims: int=13, temporal_pool: int=FRAMES_PER_BUCKET, head_hidden: int=0, with_shake_head: bool=False, with_zoom_head: bool=False, with_velocity_head: bool=False, with_motion_score_head: bool=False):
        super().__init__()
        self.policy = policy
        for p in self.policy.pi_head.parameters():
            p.requires_grad = False
        self.hidsize = policy.net.output_latent_size()
        self.temporal_pool = temporal_pool
        self.shake_head = nn.Linear(self.hidsize, 1) if with_shake_head else None
        self.zoom_head = nn.Linear(self.hidsize, 2) if with_zoom_head else None
        self.velocity_head = nn.Linear(self.hidsize, 6) if with_velocity_head else None
        self.motion_score_head = nn.Linear(self.hidsize, 1) if with_motion_score_head else None
        if head_hidden > 0:
            self.head = nn.Sequential(nn.Linear(self.hidsize, head_hidden), nn.GELU(), nn.Linear(head_hidden, n_label_dims))
        else:
            self.head = nn.Linear(self.hidsize, n_label_dims)

    def encode(self, frames: th.Tensor) -> th.Tensor:
        """Convert [B,T,3,H,W] frames to [B,T,D] VPT features."""
        B, T, C, H, W = frames.shape
        img = frames.permute(0, 1, 3, 4, 2).contiguous()
        target_dtype = next(self.policy.net.parameters()).dtype
        if img.dtype != target_dtype:
            img = img.to(dtype=target_dtype)
        obs = {'img': img}
        first = th.zeros(B, T, dtype=th.bool, device=frames.device)
        state_in = self.policy.initial_state(B)
        (pi_latent, _), _ = self.policy.net(obs, state_in=state_in, context={'first': first})
        return pi_latent

    def pool(self, frames: th.Tensor) -> th.Tensor:
        """Average each group of temporal_pool frames."""
        h = self.encode(frames)
        B, T, D = h.shape
        assert T % self.temporal_pool == 0, f'T={T} not divisible by pool={self.temporal_pool}'
        L = T // self.temporal_pool
        return h.reshape(B, L, self.temporal_pool, D).mean(dim=2)

    def forward(self, frames: th.Tensor, return_aux: bool=False):
        h = self.pool(frames)
        motion = self.head(h)
        if not return_aux:
            return motion
        return {'motion': motion, 'velocity': self.velocity_head(h) if self.velocity_head is not None else None, 'motion_score': self.motion_score_head(h.mean(dim=1)).squeeze(-1) if self.motion_score_head is not None else None}

    def forward_shake(self, frames: th.Tensor) -> th.Tensor:
        """Predict clip-level shake evidence."""
        if self.shake_head is None:
            return self.head(self.pool(frames))[..., 15].amax(dim=1)
        h = self.pool(frames).mean(dim=1)
        return self.shake_head(h).squeeze(-1)

    def forward_zoom(self, frames: th.Tensor) -> th.Tensor:
        """Predict two zoom logits per temporal bin."""
        if self.zoom_head is None:
            raise RuntimeError('zoom_head 未启用；构造时传 with_zoom_head=True')
        return self.zoom_head(self.pool(frames))

    @property
    def main_motion_head(self):
        return self.head

def enable_bf16_compat(model: nn.Module) -> int:
    """Align attention query, key and value dtypes for BF16 autocast."""
    patched = 0
    seen = set()
    for m in model.modules():
        fwd = getattr(type(m), 'forward', None)
        g = getattr(fwd, '__globals__', None)
        if g is None or id(g) in seen:
            continue
        if g.get('__name__') != 'lib.xf' or 'attention' not in g:
            continue
        seen.add(id(g))
        if g.get('_bf16_dtype_following', False):
            patched += 1
            continue
        _orig_attention = g['attention']

        def _attention_dtype_following(Q_bte, K_bTe, V_bTe, dtype=None, _orig=_orig_attention, **kw):
            dt = Q_bte.dtype
            if K_bTe.dtype != dt:
                K_bTe = K_bTe.to(dt)
            if V_bTe.dtype != dt:
                V_bTe = V_bTe.to(dt)
            return _orig(Q_bte, K_bTe, V_bTe, dtype=dt, **kw)
        g['attention'] = _attention_dtype_following
        g['_bf16_dtype_following'] = True
        g['_orig_attention'] = _orig_attention
        patched += 1
    return patched
