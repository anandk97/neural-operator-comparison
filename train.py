"""Train one model on one task and write metrics to runs/<task>/<model>_n<ntrain>.json.

    uv run python train.py --task darcy --model fno --epochs 300

Protocol (the same for every data-driven model): unit-Gaussian normalisation of inputs and outputs, relative L2
loss, AdamW with a one-cycle learning-rate schedule, a fixed epoch budget. For the autoregressive Navier-Stokes task,
models map the last 10 frames to the next one and are trained on the full 10-step rollout, as in FNO and Transolver.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from noc.data import UnitGaussian, load
from noc.models import build

RUNS = Path(__file__).parent / "runs"
BATCH = {"burgers": 20, "advection": 20, "darcy": 20, "ns": 20, "airfoil": 4, "elasticity": 8}


def rel_l2(pred, true):
    B = pred.shape[0]
    return ((pred - true).reshape(B, -1).norm(dim=1) / true.reshape(B, -1).norm(dim=1))


def rollout(model, a, x, steps, na, nu):
    """Autoregressive prediction in physical units; na/nu normalise per frame."""
    preds = []
    for _ in range(steps):
        y = nu.decode(model(na.encode(a), x))
        preds.append(y)
        a = torch.cat([a[..., 1:], y], -1)
    return torch.cat(preds, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ntrain", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="relative Gaussian noise added to test inputs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True  # same setting for every model
    torch.backends.cudnn.allow_tf32 = True
    task = load(args.task, ntrain=args.ntrain)
    model = build(args.model, task).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    bs = args.batch or BATCH[task.name]

    ar = task.rollout > 1
    na = UnitGaussian(task.a_train.reshape(-1, *task.shape, 1) if ar else task.a_train)
    nu = UnitGaussian(task.u_train.reshape(-1, *task.shape, 1) if ar else task.u_train)
    if ar:  # one normaliser shared by every frame
        na.mean, na.std = na.mean.mean(), na.std.mean()
        nu.mean, nu.std = na.mean, na.std

    shared_x = task.x_train.shape[0] == 1
    xtr = task.x_train.to(dev) if shared_x else task.x_train
    xte = task.x_test.to(dev) if shared_x else task.x_test
    a_tr, u_tr = task.a_train, task.u_train
    ntr = a_tr.shape[0]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = args.epochs * ((ntr + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 20), pct_start=0.1)

    def batch_x(xs, idx):
        return xs if shared_x else xs[idx].to(dev)

    history, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(ntr)
        tot = 0.0
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            a, u, x = a_tr[idx].to(dev), u_tr[idx].to(dev), batch_x(xtr, idx)
            if ar:
                pred = rollout(model, a, x, task.rollout, na, nu)
            else:
                pred = nu.decode(model(na.encode(a), x))
            loss = rel_l2(pred, u).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        history.append(tot / ntr)
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:4d}  train relL2 {history[-1]:.4f}  {time.time() - t0:.0f}s", flush=True)
    train_s = time.time() - t0

    train_peak = torch.cuda.max_memory_allocated() / 2**30
    model.eval()
    errs, t1 = [], time.time()
    ebs = max(1, bs // 4)
    with torch.no_grad():
        for i in range(0, task.a_test.shape[0], ebs):
            sl = slice(i, i + ebs)
            a, u = task.a_test[sl].to(dev), task.u_test[sl].to(dev)
            x = xte if shared_x else xte[sl].to(dev)
            if args.noise:
                a = a + args.noise * a.std() * torch.randn_like(a)
            pred = rollout(model, a, x, task.rollout, na, nu) if ar else nu.decode(model(na.encode(a), x))
            errs.append(rel_l2(pred, u).cpu())
    torch.cuda.synchronize()
    infer_s = (time.time() - t1) / task.a_test.shape[0]
    errs = torch.cat(errs)

    out = RUNS / task.name
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.model}_n{args.ntrain}" + (f"_noise{args.noise}" if args.noise else "") + (f"_{args.tag}" if args.tag else "")
    res = {
        "task": task.name, "model": args.model, "ntrain": args.ntrain, "epochs": args.epochs, "seed": args.seed,
        "noise": args.noise, "params": n_params, "test_rel_l2": errs.mean().item(),
        "test_rel_l2_median": errs.median().item(), "train_seconds": train_s, "infer_ms_per_sample": 1e3 * infer_s,
        "peak_gb": train_peak, "history": history,
    }
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    torch.save(model.state_dict(), out / f"{name}.pt")
    print(f"{task.name} {args.model}: test relL2 {res['test_rel_l2']:.4f}  params {n_params/1e6:.2f}M  "
          f"train {train_s/60:.1f} min  peak {res['peak_gb']:.1f} GB")


if __name__ == "__main__":
    main()
