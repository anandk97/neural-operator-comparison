"""Physics-informed baselines: solve individual test instances from the PDE alone (no training data).

  PINN       Raissi, Perdikaris & Karniadakis, J. Comput. Phys. 2019. tanh MLP, fixed loss weights, Adam.
  PirateNet  Wang, Li, Chen & Perdikaris, JMLR 2024. Random Fourier features, residual blocks with gated
             "adaptive" skips initialised to identity (alpha = 0), physics-informed initialisation of the last layer
             by least squares on the initial/boundary data, and gradient-norm loss balancing (Wang et al. 2023).
             Written from the paper; the reference implementation (jaxpi) is in JAX.

These are not operators: every new input function needs a new optimisation. We compare them per instance against
the operators, reporting error and wall-clock time, on the three tasks whose PDE is simple to write down:

  burgers    u_t + u u_x = 0.1 u_xx, periodic on [0, 1), t in [0, 1]
  advection  u_t + u_x = 0, periodic, t in [0, 0.5], square-wave initial data
  darcy      -div(a grad u) = 1 on (0, 1)^2, u = 0 on the boundary. Written in mixed form (flux q = a grad u) so the
             piecewise-constant coefficient is never differentiated.

    uv run python pinn.py --task burgers --model piratenet --instances 5
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from noc.data import DATA, _loadmat, load

RUNS = Path(__file__).parent / "runs"
dev = "cuda"


class MLP(nn.Module):
    def __init__(self, din, dout, width=128, depth=5, periodic_x=False):
        super().__init__()
        self.periodic_x = periodic_x
        d = din + (1 if periodic_x else 0)
        layers = []
        for i in range(depth):
            layers += [nn.Linear(d if i == 0 else width, width), nn.Tanh()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, dout)

    def embed(self, z):
        if self.periodic_x:  # exact periodicity in x (first coordinate)
            x = z[:, :1]
            return torch.cat([torch.cos(2 * math.pi * x), torch.sin(2 * math.pi * x), z[:, 1:]], 1)
        return z

    def features(self, z):
        return self.body(self.embed(z))

    def forward(self, z):
        return self.head(self.features(z))


class PirateNet(nn.Module):
    def __init__(self, din, dout, width=256, blocks=3, sigma=2.0, periodic_x=False):
        super().__init__()
        self.periodic_x = periodic_x
        d = din + (1 if periodic_x else 0)
        n_ff = width // 2  # embedding width equals block width so the adaptive skip is an identity
        self.register_buffer("B", sigma * torch.randn(d, n_ff))
        self.U = nn.Linear(2 * n_ff, width)
        self.V = nn.Linear(2 * n_ff, width)
        self.blocks = nn.ModuleList([nn.ModuleList([nn.Linear(width, width),
                                                    nn.Linear(width, width), nn.Linear(width, width)])
                                     for i in range(blocks)])
        self.alpha = nn.Parameter(torch.zeros(blocks))  # identity at initialisation
        self.head = nn.Linear(width, dout)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def embed(self, z):
        if self.periodic_x:
            x = z[:, :1]
            z = torch.cat([torch.cos(2 * math.pi * x), torch.sin(2 * math.pi * x), z[:, 1:]], 1)
        p = z @ self.B
        return torch.cat([torch.cos(p), torch.sin(p)], 1)

    def features(self, z):
        phi = self.embed(z)
        U, V = torch.tanh(self.U(phi)), torch.tanh(self.V(phi))
        x = phi
        for (f, g, h), a in zip(self.blocks, self.alpha):
            fx = torch.tanh(f(x))
            z1 = fx * U + (1 - fx) * V
            gz = torch.tanh(g(z1))
            z2 = gz * U + (1 - gz) * V
            hz = torch.tanh(h(z2))
            x = a * hz + (1 - a) * x
        return x

    def forward(self, z):
        return self.head(self.features(z))

    @torch.no_grad()
    def physics_init(self, z, target):
        """Least-squares fit of the last layer to known data (initial or boundary values). A small ridge penalty keeps
        the fit well-conditioned: an exact interpolant of the initial data has huge weights that blow up for t > 0."""
        F = self.features(z)
        F1 = torch.cat([F, torch.ones_like(F[:, :1])], 1).double()
        G = F1.T @ F1
        lam = 1e-3 * G.diagonal().mean()
        sol = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=G.device, dtype=G.dtype),
                                 F1.T @ target.double()).float()
        self.head.weight.copy_(sol[:-1].T)
        self.head.bias.copy_(sol[-1])


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def problem(task, i):
    """Returns (residual_fn, data_points, data_values, eval_points, truth, dim, periodic)."""
    if task in ("burgers", "advection"):
        T = load(task, ntrain=1)
        a, u = T.a_test[i, :, 0], T.u_test[i, :, 0]
        n = a.shape[0]
        xs = torch.arange(n) / n
        tf = 1.0 if task == "burgers" else 0.5
        z0 = torch.stack([xs, torch.zeros(n)], 1)
        ze = torch.stack([xs, torch.full((n,), tf)], 1)

        def residual(net, z):
            z.requires_grad_(True)
            v = net(z)
            g = grad(v, z)
            vx, vt = g[:, :1], g[:, 1:]
            if task == "advection":
                return [vt + vx]
            vxx = grad(vx, z)[:, :1]
            return [vt + v * vx - 0.1 * vxx]

        def sample(m):
            return torch.stack([torch.rand(m), tf * torch.rand(m)], 1)

        return residual, sample, z0, a[:, None], ze, u, True
    # darcy: coefficient at full 421 resolution for collocation, truth on the 85 x 85 evaluation grid
    m = _loadmat(DATA / "piececonst_r421_N1024_smooth2.mat")
    coeff = torch.as_tensor(m["coeff"][i], dtype=torch.float32)
    truth = torch.as_tensor(m["sol"][i, ::5, ::5], dtype=torch.float32).flatten()
    g = torch.linspace(0, 1, 85)
    ze = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)

    def coef_at(z):
        idx = (z.detach() * 420).round().long().clamp(0, 420)
        return coeff.to(z.device)[idx[:, 0], idx[:, 1]][:, None]

    def residual(net, z):
        z.requires_grad_(True)
        out = net(z)
        # hard boundary condition u = 0 via the distance function x(1-x)y(1-y)
        dist = (z[:, :1] * (1 - z[:, :1]) * z[:, 1:] * (1 - z[:, 1:]))
        u, q = 16 * dist * out[:, :1], out[:, 1:]
        gu = grad(u, z)
        div = grad(q[:, :1], z)[:, :1] + grad(q[:, 1:], z)[:, 1:]
        return [q - coef_at(z) * gu, -div - 1.0]

    def sample(m):
        return torch.rand(m, 2)

    return residual, sample, None, None, ze, truth, False


def solve(task, model, i, iters, seed=0):
    torch.manual_seed(seed)
    residual, sample, zd, vd, ze, truth, periodic = problem(task, i)
    dout = 3 if task == "darcy" else 1
    net = (PirateNet(2, dout, periodic_x=periodic) if model == "piratenet" else MLP(2, dout, periodic_x=periodic)).to(dev)
    if zd is not None:
        zd, vd = zd.to(dev), vd.to(dev)
        if model == "piratenet":
            net.physics_init(zd, vd)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9 ** (1 / 2000))  # 0.9 every 2000 steps
    n_res = 4096
    w = None
    t0 = time.time()

    def u_of(z):
        out = net(z)
        if task == "darcy":
            dist = z[:, :1] * (1 - z[:, :1]) * z[:, 1:] * (1 - z[:, 1:])
            return 16 * dist * out[:, :1]
        return out[:, :1]

    for it in range(iters):
        z = sample(n_res).to(dev)
        losses = [(r ** 2).mean() for r in residual(net, z)]
        if zd is not None:
            losses.append(((net(zd) - vd) ** 2).mean())
        if model == "piratenet" and it % 1000 == 0:  # gradient-norm loss balancing
            norms = []
            for L in losses:
                g = torch.autograd.grad(L, list(net.parameters()), retain_graph=True, allow_unused=True)
                norms.append(torch.sqrt(sum((x ** 2).sum() for x in g if x is not None)))
            tot = sum(norms)
            new = torch.stack([tot / (n + 1e-8) for n in norms]).detach()
            w = new if w is None else 0.9 * w + 0.1 * new
        weights = w if w is not None else torch.ones(len(losses), device=dev)
        loss = sum(wi * L for wi, L in zip(weights, losses))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    secs = time.time() - t0
    with torch.no_grad():
        pred = u_of(ze.to(dev)).squeeze(1).cpu()
    err = ((pred - truth).norm() / truth.norm()).item()
    return err, secs, pred.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["burgers", "advection", "darcy"])
    ap.add_argument("--model", required=True, choices=["pinn", "piratenet"])
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20000)
    args = ap.parse_args()
    errs, times = [], []
    for i in range(args.instances):
        e, s, _ = solve(args.task, args.model, i, args.iters)
        errs.append(e)
        times.append(s)
        print(f"{args.task} {args.model} instance {i}: rel L2 {e:.4f}  {s:.0f}s", flush=True)
    out = RUNS / args.task
    out.mkdir(parents=True, exist_ok=True)
    res = {"task": args.task, "model": args.model, "instances": args.instances, "iters": args.iters,
           "test_rel_l2": float(np.mean(errs)), "per_instance": errs, "seconds_per_instance": float(np.mean(times))}
    (out / f"{args.model}_instances.json").write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    main()
