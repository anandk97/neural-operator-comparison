"""Kolmogorov-Arnold layers.

RBFKAN follows DeepOKAN (Abueidda, Pantidis & Mobasher, CMAME 2025): each edge function is a sum of Gaussian radial
basis functions on a fixed grid, plus a base SiLU branch. BSplineKAN follows the original KAN (Liu et al., ICLR 2025)
and KANO (Lee et al., 2025): cubic B-splines on a uniform grid plus a SiLU base term. Both were written from the
papers' equations; neither upstream repository has a license that would allow copying.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFKAN(nn.Module):
    def __init__(self, din, dout, n_centers=8, lo=-2.0, hi=2.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(lo, hi, n_centers))
        self.gamma = (n_centers - 1) / (hi - lo)  # width so that neighbouring bumps overlap
        self.norm = nn.LayerNorm(din)
        self.spline = nn.Linear(din * n_centers, dout, bias=False)
        self.base = nn.Linear(din, dout)
        nn.init.normal_(self.spline.weight, std=1.0 / (din * n_centers) ** 0.5)

    def forward(self, x):
        z = self.norm(x)
        phi = torch.exp(-((z[..., None] - self.centers) * self.gamma) ** 2)  # [..., din, K]
        return self.spline(phi.flatten(-2)) + self.base(F.silu(x))


class BSplineKAN(nn.Module):
    def __init__(self, din, dout, grid=10, order=3, lo=-2.0, hi=2.0):
        super().__init__()
        h = (hi - lo) / grid
        knots = torch.arange(-order, grid + order + 1) * h + lo
        self.register_buffer("knots", knots)
        self.order = order
        nb = grid + order
        self.coef = nn.Parameter(torch.randn(dout, din, nb) / (din * nb) ** 0.5)
        self.base = nn.Linear(din, dout)
        self.norm = nn.LayerNorm(din)

    def bases(self, x):
        if self.order == 3:
            return self.cubic_bases(x)
        return self.bases_recursive(x)

    def cubic_bases(self, x):
        """Closed-form uniform cubic B-splines: only 4 bases are non-zero at any x, so compute them directly and
        scatter them into place. Agrees with the Cox-de Boor recursion to ~1e-6, with far less memory traffic."""
        t = self.knots
        h = t[1] - t[0]
        nb = t.numel() - self.order - 1
        s = (x - t[0]) / h
        k = torch.floor(s)
        u = s - k
        u2, u3 = u * u, u * u * u
        vals = torch.stack([(1 - u) ** 3, 3 * u3 - 6 * u2 + 4, -3 * u3 + 3 * u2 + 3 * u + 1, u3], -1) / 6
        idx = k.long()[..., None] + torch.arange(-3, 1, device=x.device)  # bases k-3 .. k
        ok = (idx >= 0) & (idx < nb)
        out = torch.zeros(*x.shape, nb + 1, dtype=x.dtype, device=x.device)
        out.scatter_add_(-1, torch.where(ok, idx, nb), vals * ok)
        return out[..., :nb]

    def bases_recursive(self, x):
        t = self.knots
        x = x[..., None]
        b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
        for k in range(1, self.order + 1):  # Cox-de Boor recursion
            b = ((x - t[:-(k + 1)]) / (t[k:-1] - t[:-(k + 1)]) * b[..., :-1]
                 + (t[k + 1:] - x) / (t[k + 1:] - t[1:-k]) * b[..., 1:])
        return b  # [..., din, nb]

    def forward(self, x):
        z = self.norm(x)
        return torch.einsum("...ik,oik->...o", self.bases(z), self.coef) + self.base(F.silu(x))


def kan_mlp(dims, kind="rbf", **kw):
    layer = RBFKAN if kind == "rbf" else BSplineKAN
    return nn.Sequential(*[layer(a, b, **kw) for a, b in zip(dims[:-1], dims[1:])])
