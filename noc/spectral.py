"""Grid-based spectral operators: FNO, local-kernel FNO (LocalNO) and an adapted KANO.

  FNO      Li et al., ICLR 2021. neuraloperator library implementation (Kossaifi et al. 2024, MIT license).
  LocalNO  Liu-Schiaffini et al., ICML 2024, "Neural operators with localized integral and differential kernels".
           neuraloperator's LocalNO, which adds a local integral (DISCO) kernel and a finite-difference kernel in
           parallel with each Fourier layer.
  KANO     Lee, Liu, Yu, Wang, Jeong, Niu & Zhang, arXiv:2509.16825. Layer: v = Phi(Op(p) a, a), where Op(p) is the
           Kohn-Nirenberg quantization of a symbol p(x, xi) on truncated Fourier modes and Phi is a KAN.
           ADAPTED, not the reference code. The reference implementation is single-channel, 1D, and evaluates the
           quantization as a dense N x N x N sum. Here, to run on 2D benchmarks:
             * the symbol is a rank-R separable expansion p_oi(x, xi) = S_oi(xi) + sum_r c_r(x) s_r,oi(xi), so
               Op(p) a = F^-1[S F a] + sum_r c_r(x) F^-1[s_r F a], which is exact for this symbol class and costs
               R + 1 FFT convolutions; c_r(x) is a small MLP of position, so position-dependent terms such as
               x^2 f stay expressible, which is KANO's point against FNO;
             * lifting and projection layers give it the same channel width as FNO (the paper removes them for
               interpretability on single-channel toy problems);
             * Phi is a cubic B-spline KAN (grid 10, as in the paper) acting on (Op(p) a, a) channel-wise.
"""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from neuralop.models import FNO
from neuralop.models.local_no import LocalNO

from .kan import BSplineKAN


def _channels_first(a, x):
    z = torch.cat([a, x.expand(a.shape[0], *x.shape[1:])], -1)
    return z.movedim(-1, 1)


class FNOWrap(nn.Module):
    def __init__(self, task, local=False, modes=None, width=None, layers=4):
        super().__init__()
        dim = len(task.shape)
        # FNO paper settings (16 modes in 1D, 12 x 12 in 2D, per sign). neuraloperator counts positive and negative
        # frequencies together, so the equivalent n_modes is doubled.
        modes = modes or ((32,) if dim == 1 else (24, 24))
        width = width or (64 if dim == 1 else 32)
        pad = None if task.periodic else 0.125
        kw = dict(n_modes=modes, in_channels=task.cin + task.d, out_channels=task.cout, hidden_channels=width,
                  n_layers=layers, positional_embedding=None, domain_padding=pad)
        if local:
            # the local (DISCO) kernels are built for the shape they see, i.e. after symmetric domain padding
            shape = tuple(n + 2 * round(pad * n) for n in task.shape) if pad else task.shape
            self.net = LocalNO(default_in_shape=shape, **kw,
                               conv_padding_mode="periodic" if task.periodic else "zeros",
                               disco_layers=dim == 2, domain_length=[1.0] * dim)
        else:
            self.net = FNO(**kw)

    def forward(self, a, x):
        return self.net(_channels_first(a, x)).movedim(1, -1)


class KANOLayer(nn.Module):
    def __init__(self, width, modes, dim, rank=2):
        super().__init__()
        self.modes, self.dim, self.rank = modes, dim, rank
        nm = math.prod(modes) * (2 ** (dim - 1))  # kept modes (both signs on all but the last axis)
        scale = 1 / width
        self.S = nn.Parameter(scale * torch.randn(rank + 1, width, width, nm, dtype=torch.cfloat))
        self.c = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, rank * width))
        self.phi = BSplineKAN(2 * width, width, grid=10)

    def _modes(self, ah):
        if self.dim == 1:
            return ah[..., :self.modes[0]]
        m0, m1 = self.modes
        return torch.cat([ah[..., :m0, :m1], ah[..., -m0:, :m1]], -2)

    def _spec(self, a, W):  # a: [B, C, *S]; W: [C, C, nm] -> [B, C, *S]
        S = a.shape[2:]
        ah = torch.fft.rfftn(a, dim=tuple(range(2, 2 + self.dim)))
        km = self._modes(ah).flatten(2)
        out_m = torch.einsum("bik,oik->bok", km, W)
        oh = torch.zeros(a.shape[0], W.shape[0], *ah.shape[2:], dtype=ah.dtype, device=a.device)
        if self.dim == 1:
            oh[..., :self.modes[0]] = out_m
        else:
            m0, m1 = self.modes
            out_m = out_m.view(a.shape[0], -1, 2 * m0, m1)
            oh[..., :m0, :m1], oh[..., -m0:, :m1] = out_m[..., :m0, :], out_m[..., m0:, :]
        return torch.fft.irfftn(oh, s=S, dim=tuple(range(2, 2 + self.dim)))

    def forward(self, a, x):  # a: [B, C, *S], x: [1 or B, *S, d]
        g = self._spec(a, self.S[0])
        c = self.c(x).movedim(-1, 1)  # [., rank*C, *S]
        C = a.shape[1]
        for r in range(self.rank):
            g = g + c[:, r * C:(r + 1) * C] * self._spec(a, self.S[r + 1])
        z = torch.cat([g, a], 1).movedim(1, -1)
        if self.training:  # the spline basis expansion is memory-heavy; recompute it in the backward pass
            return checkpoint(self.phi, z, use_reentrant=False).movedim(-1, 1)
        return self.phi(z).movedim(-1, 1)


class KANOWrap(nn.Module):
    def __init__(self, task, modes=None, width=32, layers=4, rank=2):
        super().__init__()
        dim = len(task.shape)
        modes = modes or ((16,) if dim == 1 else (12, 12))
        self.lift = nn.Linear(task.cin + task.d, width)
        self.layers = nn.ModuleList([KANOLayer(width, modes, dim, rank) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, task.cout))
        self.pad = not task.periodic

    def forward(self, a, x):
        x = x.expand(a.shape[0], *x.shape[1:])
        h = self.lift(torch.cat([a, x], -1)).movedim(-1, 1)
        S = h.shape[2:]
        if self.pad:  # zero-pad non-periodic domains, as FNO does
            p = [int(0.125 * s) for s in S]
            h = nn.functional.pad(h, [v for s in reversed(p) for v in (0, s)])
            xp = nn.functional.pad(x.movedim(-1, 1), [v for s in reversed(p) for v in (0, s)], mode="replicate")
            xp = xp.movedim(1, -1)
        else:
            xp = x
        for layer in self.layers:
            h = layer(h, xp)
        if self.pad:
            h = h[(..., *[slice(0, s) for s in S])]
        return self.proj(h.movedim(1, -1))
