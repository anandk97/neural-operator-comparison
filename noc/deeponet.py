"""DeepONet family: DeepONet, Shift-DeepONet and DeepOKAN, all written from their papers.

  DeepONet       Lu, Jin, Pang, Zhang & Karniadakis, Nature Machine Intelligence 2021.
                 G(a)(y) = sum_k beta_k(a) tau_k(y) + b0
  Shift-DeepONet Lanthaler, Molinaro, Hadorn & Mishra, ICLR 2023 ("Nonlinear reconstruction for operator learning of
                 PDEs with discontinuities"). G(a)(y) = sum_k beta_k(a) tau_k(A_k(a) y + gamma_k(a)) + b0
                 The input-dependent scale A_k and shift gamma_k let one basis function move with a discontinuity.
  DeepOKAN       Abueidda, Pantidis & Mobasher, CMAME 2025. DeepONet with the branch and trunk MLPs replaced by
                 Gaussian-RBF KANs.

For 2D inputs on a grid or structured mesh the branch starts with a small CNN encoder, as in Lu et al.'s CMAME 2022
comparison; for 1D and point-cloud inputs the branch reads the input at fixed sensor points. The encoder is the
same across the three variants, so differences come from the part each paper changes.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .kan import kan_mlp


def mlp(dims, act=nn.GELU):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class CNNEncoder(nn.Module):
    def __init__(self, cin, width=32, out=256):
        super().__init__()
        c = [cin, width, 2 * width, 4 * width, 4 * width]
        layers = []
        for a, b in zip(c[:-1], c[1:]):
            layers += [nn.Conv2d(a, b, 5, stride=2, padding=2), nn.GELU()]
        self.conv = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Linear(c[-1] * 16, out)

    def forward(self, a):  # a: [B, H, W, C]
        return self.fc(self.conv(a.permute(0, 3, 1, 2)).flatten(1))


class SensorEncoder(nn.Module):
    """Reads the input at a fixed set of sensor locations (every `stride`-th point) and flattens."""

    def __init__(self, n_points, cin, stride):
        super().__init__()
        self.stride = stride
        self.out = len(range(0, n_points, stride)) * cin

    def forward(self, a):
        return a.flatten(1, -2)[:, ::self.stride].flatten(1)


class DeepONetFamily(nn.Module):
    def __init__(self, task, variant="deeponet", p=128, width=128, depth=4, sensors=256):
        super().__init__()
        self.variant, self.p, self.cout, self.d = variant, p, task.cout, task.d
        if task.kind in ("grid2d", "mesh2d"):
            self.enc = CNNEncoder(task.cin, out=width)
            enc_out = width
        else:
            n_points = int(torch.tensor(task.shape).prod())
            self.enc = SensorEncoder(n_points, task.cin, max(1, n_points // sensors))
            enc_out = self.enc.out
        hidden = [width] * (depth - 1)
        if variant == "deepokan":
            self.branch = kan_mlp([enc_out, *hidden, p * self.cout], kind="rbf")
            self.trunk = kan_mlp([self.d, *hidden, p * self.cout], kind="rbf")
        else:
            self.branch = mlp([enc_out, *hidden, p * self.cout])
            self.trunk = nn.Sequential(mlp([self.d, *hidden, p * self.cout]), nn.GELU())
        if variant == "shift":
            # A_k(a) (d x d) and gamma_k(a) (d) for every basis function k, from the same encoded input.
            self.scale = mlp([enc_out, width, p * self.d * self.d])
            self.shift = mlp([enc_out, width, p * self.d])
            # With per-basis shifts each basis function needs its own trunk evaluation; a grouped trunk
            # (one small MLP per basis function, evaluated in parallel) keeps that affordable.
            tw = 32
            self.t_w1 = nn.Parameter(torch.randn(p, self.d, tw) / self.d ** 0.5)
            self.t_b1 = nn.Parameter(torch.zeros(p, tw))
            self.t_w2 = nn.Parameter(torch.randn(p, tw, tw) / tw ** 0.5)
            self.t_b2 = nn.Parameter(torch.zeros(p, tw))
            self.t_w3 = nn.Parameter(torch.randn(p, tw, self.cout) / tw ** 0.5)
            del self.trunk
        self.b0 = nn.Parameter(torch.zeros(self.cout))

    def grouped_trunk(self, y):  # y: [B, P, p, d] -> [B, P, p, cout]
        h = torch.nn.functional.gelu(torch.einsum("bnkd,kdh->bnkh", y, self.t_w1) + self.t_b1)
        h = torch.nn.functional.gelu(torch.einsum("bnkh,khg->bnkg", h, self.t_w2) + self.t_b2)
        return torch.einsum("bnkg,kgc->bnkc", h, self.t_w3)

    def forward(self, a, x):
        B = a.shape[0]
        shape = a.shape[1:-1]
        e = self.enc(a)
        beta = self.branch(e).view(B, 1, self.p, self.cout)
        y = x.expand(B, *x.shape[1:]).reshape(B, -1, self.d)  # [B, P, d]
        if self.variant == "shift":
            A = self.scale(e).view(B, 1, self.p, self.d, self.d) + torch.eye(self.d, device=a.device)
            g = self.shift(e).view(B, 1, self.p, self.d)
            z = torch.einsum("bkij,bnj->bnki", A[:, 0], y) + g  # [B, P, p, d]
            # evaluate in chunks of points so the [B, P, p, width] activations fit in GPU memory
            tau = torch.cat([checkpoint(self.grouped_trunk, zc, use_reentrant=False) if self.training
                             else self.grouped_trunk(zc) for zc in z.split(2048, 1)], 1)
        else:
            trunk = (lambda t: checkpoint(self.trunk, t, use_reentrant=False)) if self.training else self.trunk
            tau = trunk(y).view(B, y.shape[1], self.p, self.cout)
        out = (beta * tau).sum(2) / self.p ** 0.5 + self.b0
        return out.view(B, *shape, self.cout)
