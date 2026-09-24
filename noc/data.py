"""Benchmark tasks. Each task is a map from an input function a to an output function u.

Every task returns tensors shaped [N, *S, C], where S is the spatial layout (a grid, a structured mesh, or a flat point
list). Coordinates are shaped [N, *S, d], either shared across samples or, for meshes, one set per sample. Splits and
resolutions follow the papers that report published numbers for each task, so our numbers can sit next to theirs.

  burgers     FNO (Li et al., ICLR 2021)          viscous Burgers, nu = 0.1 on a 2 pi-periodic domain, u0 -> u(t=1), smooth
  advection   Lanthaler et al. (ICLR 2023) style  linear advection of random square waves, u0 -> u(t=0.5)
  darcy       FNO / Transolver                    piecewise-constant coefficient -> pressure, 85 x 85
  ns          FNO / Transolver                    2D vorticity, nu = 1e-5, 10 frames -> next 10, 64 x 64
  airfoil     Geo-FNO / Transolver                NACA mesh (221 x 51) -> Mach number
  elasticity  Geo-FNO / Transolver                unit-cell point cloud (972 points) -> von Mises stress
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Task:
    name: str
    kind: str  # "grid1d" | "grid2d" | "mesh2d" | "points"
    a_train: torch.Tensor
    u_train: torch.Tensor
    a_test: torch.Tensor
    u_test: torch.Tensor
    x_train: torch.Tensor  # coordinates [N or 1, *S, d]
    x_test: torch.Tensor
    periodic: bool = False
    rollout: int = 1  # >1: autoregressive, the model maps the last `in_steps` frames to the next frame
    in_steps: int = 1
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return tuple(self.a_train.shape[1:-1])

    @property
    def d(self):
        return self.x_train.shape[-1]

    @property
    def cin(self):
        return self.a_train.shape[-1]

    @property
    def cout(self):
        return 1 if self.rollout > 1 else self.u_train.shape[-1]


def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _grid(*sizes):
    axes = [torch.linspace(0, 1, s) for s in sizes]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)[None]


def _periodic_grid(n):
    return (torch.arange(n) / n)[None, :, None]


def _loadmat(path):
    try:
        from scipy.io import loadmat

        return loadmat(path)
    except NotImplementedError:  # MATLAB v7.3 files are HDF5
        import h5py

        f = h5py.File(path, "r")
        return {k: np.transpose(f[k][()]) for k in f.keys()}


def burgers(ntrain=1000, ntest=200, sub=8, **_):
    m = _loadmat(DATA / "burgers_data_R10.mat")
    a, u = _t(m["a"][:, ::sub])[..., None], _t(m["u"][:, ::sub])[..., None]
    x = _periodic_grid(a.shape[1])
    return Task("burgers", "grid1d", a[:ntrain], u[:ntrain], a[-ntest:], u[-ntest:], x, x, periodic=True,
                meta={"resolution": a.shape[1]})


def advection(ntrain=1000, ntest=200, n=1024, T=0.5, seed=0, **_):
    """u_t + u_x = 0 on the periodic unit interval, so u(x, T) = u0(x - T) exactly.

    Initial data are square waves h * 1{|x - c| < w/2} with h in [0.2, 0.8] and w in [0.05, 0.3] (Lanthaler et al.'s
    ranges) and centre c uniform on [0, 1). Lanthaler et al. (2023) use this setting to show that any operator with a
    linear reconstruction, such as DeepONet, needs many basis functions for transported discontinuities (slow
    Kolmogorov n-width decay).
    """
    rng = np.random.default_rng(seed)
    N = ntrain + ntest
    h, c, w = rng.uniform(0.2, 0.8, N), rng.uniform(0, 1, N), rng.uniform(0.05, 0.3, N)
    x = np.arange(n) / n

    def square(shift):
        dist = np.abs(((x[None] - shift - c[:, None]) + 0.5) % 1.0 - 0.5)
        return h[:, None] * (dist < w[:, None] / 2)

    a, u = _t(square(0.0))[..., None], _t(square(T))[..., None]
    xg = _periodic_grid(n)
    return Task("advection", "grid1d", a[:ntrain], u[:ntrain], a[-ntest:], u[-ntest:], xg, xg, periodic=True,
                meta={"resolution": n})


def darcy(ntrain=1000, ntest=200, sub=5, **_):
    tr = _loadmat(DATA / "piececonst_r421_N1024_smooth1.mat")
    te = _loadmat(DATA / "piececonst_r421_N1024_smooth2.mat")
    s = (421 - 1) // sub + 1

    def get(m, n):
        return (_t(m["coeff"][:n, ::sub, ::sub][:, :s, :s])[..., None],
                _t(m["sol"][:n, ::sub, ::sub][:, :s, :s])[..., None])

    a, u = get(tr, ntrain)
    at, ut = get(te, ntest)
    x = _grid(s, s)
    return Task("darcy", "grid2d", a, u, at, ut, x, x, meta={"resolution": s})


def ns(ntrain=1000, ntest=200, T_in=10, T=10, **_):
    m = _loadmat(DATA / "NavierStokes_V1e-5_N1200_T20.mat")
    w = _t(m["u"])  # [N, 64, 64, 20]
    a, u = w[..., :T_in], w[..., T_in:T_in + T]
    x = _grid(64, 64)  # the vorticity field is periodic on the torus
    return Task("ns", "grid2d", a[:ntrain], u[:ntrain], a[-ntest:], u[-ntest:], x, x, periodic=True, rollout=T,
                in_steps=T_in, meta={"resolution": 64})


def airfoil(ntrain=1000, ntest=200, **_):
    d = DATA / "airfoil"
    X, Y = np.load(d / "NACA_Cylinder_X.npy"), np.load(d / "NACA_Cylinder_Y.npy")
    Q = np.load(d / "NACA_Cylinder_Q.npy")[:, 4]  # Mach number
    xy = _t(np.stack([X, Y], -1))  # [N, 221, 51, 2]
    u = _t(Q)[..., None]
    return Task("airfoil", "mesh2d", xy[:ntrain], u[:ntrain], xy[ntrain:ntrain + ntest], u[ntrain:ntrain + ntest],
                xy[:ntrain], xy[ntrain:ntrain + ntest], meta={"resolution": "221x51"})


def elasticity(ntrain=1000, ntest=200, **_):
    d = DATA / "elasticity"
    sigma = _t(np.load(d / "Random_UnitCell_sigma_10.npy")).T[..., None]  # [2000, 972, 1]
    xy = _t(np.load(d / "Random_UnitCell_XY_10.npy")).permute(2, 0, 1)  # [2000, 972, 2]
    return Task("elasticity", "points", xy[:ntrain], sigma[:ntrain], xy[-ntest:], sigma[-ntest:], xy[:ntrain],
                xy[-ntest:], meta={"resolution": "972 points"})


TASKS = {f.__name__: f for f in (burgers, advection, darcy, ns, airfoil, elasticity)}


def load(name, **kw):
    return TASKS[name](**kw)


class UnitGaussian:
    """Pointwise mean/std normalizer fitted on training data (as in the FNO and Transolver code)."""

    def __init__(self, x, eps=1e-5):
        self.mean, self.std, self.eps = x.mean(0, keepdim=True), x.std(0, keepdim=True), eps

    def encode(self, x):
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)

    def decode(self, x):
        return x * (self.std.to(x.device) + self.eps) + self.mean.to(x.device)
