"""Transolver and Transolver++, adapted from the authors' MIT-licensed code.

  Transolver   Wu, Luo, Wang, Wang & Long, ICML 2024. github.com/thuml/Transolver (MIT). Physics-Attention: points are
               softly assigned to learned "slices", attention runs among slice tokens, and results are broadcast back.
               Structured grids and meshes use the Structured_Mesh_2D variant (3x3 conv projections); point clouds
               use the Irregular_Mesh variant.
  Transolver++ Luo, Wu, Zhou, Xing, Di, Wang & Long, ICML 2025. github.com/thuml/Transolver_plus (MIT). Eidetic
               physics-attention: a shared projection, per-point adaptive temperature (Ada-Temp) and Gumbel-softmax
               slice assignment (Rep-Slice). Distributed all-reduce calls removed for single-GPU use.

Copyright (c) 2024 THUML @ Tsinghua University, MIT License (see third_party/transolver*/LICENSE). Changes: merged
into one module, a common (a, x) interface, timm/einops dependencies replaced with torch equivalents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def gumbel_softmax(logits, tau=1.0):
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    return F.softmax((logits + g) / tau, dim=-1)


class PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=8, slice_num=64, plus=False, grid=None):
        super().__init__()
        dh = dim // heads
        self.heads, self.dh, self.plus, self.grid = heads, dh, plus, grid
        inner = dh * heads
        if plus:
            self.in_x = nn.Linear(dim, inner)
            self.bias = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
            self.proj_temp = nn.Sequential(nn.Linear(dh, slice_num), nn.GELU(), nn.Linear(slice_num, 1), nn.GELU())
        else:
            self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
            if grid is not None:
                self.in_x = nn.Conv2d(dim, inner, 3, 1, 1)
                self.in_fx = nn.Conv2d(dim, inner, 3, 1, 1)
            else:
                self.in_x = nn.Linear(dim, inner)
                self.in_fx = nn.Linear(dim, inner)
        self.in_slice = nn.Linear(dh, slice_num)
        nn.init.orthogonal_(self.in_slice.weight)
        self.q, self.k, self.v = (nn.Linear(dh, dh, bias=False) for _ in range(3))
        self.out = nn.Linear(inner, dim)

    def _heads(self, t, B, N):
        return t.reshape(B, N, self.heads, self.dh).permute(0, 2, 1, 3)

    def forward(self, x):
        B, N, C = x.shape
        if self.plus:
            x_mid = fx_mid = self._heads(self.in_x(x), B, N)
            temp = torch.clamp(self.proj_temp(x_mid) + self.bias, min=0.01)
            w = gumbel_softmax(self.in_slice(x_mid), temp)
        else:
            if self.grid is not None:
                H, W = self.grid_shape
                xg = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x_mid = self._heads(self.in_x(xg).permute(0, 2, 3, 1), B, N)
                fx_mid = self._heads(self.in_fx(xg).permute(0, 2, 3, 1), B, N)
                temp = torch.clamp(self.temperature, 0.1, 5)
            else:
                x_mid, fx_mid = self._heads(self.in_x(x), B, N), self._heads(self.in_fx(x), B, N)
                temp = self.temperature
            w = F.softmax(self.in_slice(x_mid) / temp, dim=-1)  # [B, H, N, G]
        tok = torch.einsum("bhnc,bhng->bhgc", fx_mid, w) / (w.sum(2)[..., None] + 1e-5)
        o = F.scaled_dot_product_attention(self.q(tok), self.k(tok), self.v(tok))
        out = torch.einsum("bhgc,bhng->bhnc", o, w).permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, dim, heads, slice_num, plus, grid, mlp_ratio=1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = PhysicsAttention(dim, heads, slice_num, plus, grid)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        if self.training:  # checkpoint attention and MLP, as the Transolver++ code does, to fit a 12 GB GPU
            x = x + checkpoint(self.attn, self.ln1(x), use_reentrant=False)
            return x + checkpoint(self.mlp, self.ln2(x), use_reentrant=False)
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


def set_grid(model, shape):
    """Structured-mesh Physics-Attention reshapes tokens to the grid; tell it the current grid shape."""
    for m in model.modules():
        if isinstance(m, PhysicsAttention):
            m.grid_shape = shape


class Transolver(nn.Module):
    """unified_pos: replace raw coordinates with distances to a ref x ref grid of reference points, as the authors'
    Darcy and Navier-Stokes configurations do (--unified_pos 1 --ref 8)."""

    def __init__(self, task, plus=False, width=128, layers=8, heads=8, slice_num=64, unified_pos=False, ref=8):
        super().__init__()
        grid = task.shape if (task.kind in ("grid2d", "mesh2d") and not plus) else None
        self.unified_pos = unified_pos
        if unified_pos:
            g = torch.linspace(0, 1, ref)
            self.register_buffer("refs", torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2))
        pos_dim = ref * ref if unified_pos else task.d
        self.pre = nn.Sequential(nn.Linear(task.cin + pos_dim, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.placeholder = nn.Parameter(torch.rand(width) / width)
        self.blocks = nn.ModuleList([Block(width, heads, slice_num, plus, grid) for _ in range(layers)])
        self.ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, task.cout)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, a, x):
        B, shape = a.shape[0], a.shape[1:-1]
        set_grid(self, tuple(shape))
        if self.unified_pos:
            x = torch.cdist(x.reshape(x.shape[0], -1, x.shape[-1]), self.refs[None].expand(x.shape[0], -1, -1))
            x = x.view(x.shape[0], *shape, -1)
        z = torch.cat([a, x.expand(B, *x.shape[1:])], -1).reshape(B, -1, a.shape[-1] + x.shape[-1])
        h = self.pre(z) + self.placeholder
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h)).view(B, *shape, -1)
