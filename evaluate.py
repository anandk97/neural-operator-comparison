"""Post-hoc tests on trained checkpoints (no retraining). Writes runs/<task>/<model>_eval.json.

  noise        Gaussian noise on the test inputs, with standard deviation equal to 0.1%, 1% or 5% of each input's
               maximum absolute value (the definition Lu et al. 2022 use).
  resolution   models trained at one resolution, tested at others (burgers: 1024 -> 256..4096; darcy: 85 -> 43..211).
               Grid-based models see the input at the new resolution. DeepONet-family models read their fixed
               sensors from the training-resolution input and are queried at the new output points, which is how
               DeepONet is meant to generalise across resolutions.
  spectrum     error energy per Fourier mode (1D) or per radial wavenumber (2D), next to the energy of the truth,
               to show which scales each model gets wrong.
  samples      a few test predictions, downsampled, for the website.

    uv run python evaluate.py --task burgers --model fno
"""

import argparse
import json
from pathlib import Path

import torch

from noc.data import UnitGaussian, load
from noc.models import build

RUNS = Path(__file__).parent / "runs"
dev = "cuda"
DEEPONETS = {"deeponet", "shift_deeponet", "deepokan"}
RESOLUTIONS = {"burgers": [32, 16, 8, 4, 2], "darcy": [10, 5, 3, 2]}  # `sub` values; 8 and 5 are the training grids


def normalisers(task):
    ar = task.rollout > 1
    na = UnitGaussian(task.a_train.reshape(-1, *task.shape, 1) if ar else task.a_train)
    nu = UnitGaussian(task.u_train.reshape(-1, *task.shape, 1) if ar else task.u_train)
    if ar:
        na.mean, na.std = na.mean.mean(), na.std.mean()
        nu.mean, nu.std = na.mean, na.std
    return na, nu


@torch.no_grad()
def predict(model, task, a, x, na, nu, bs=4):
    """a: test inputs [N, *S, C]; x: output coordinates (shared [1, ...] or per sample)."""
    out = []
    shared = x.shape[0] == 1
    for i in range(0, a.shape[0], bs):
        ai = a[i:i + bs].to(dev)
        xi = x.to(dev) if shared else x[i:i + bs].to(dev)
        if task.rollout > 1:
            preds = []
            for _ in range(task.rollout):
                y = nu.decode(model(na.encode(ai), xi))
                preds.append(y)
                ai = torch.cat([ai[..., 1:], y], -1)
            out.append(torch.cat(preds, -1).cpu())
        else:
            out.append(nu.decode(model(na.encode(ai), xi)).cpu())
    return torch.cat(out)


def rel_l2(p, u):
    B = p.shape[0]
    return ((p - u).reshape(B, -1).norm(dim=1) / u.reshape(B, -1).norm(dim=1)).mean().item()


def spectrum(p, u, task):
    e, t = (p - u)[..., 0] if task.rollout == 1 else (p - u)[..., -1], u[..., 0] if task.rollout == 1 else u[..., -1]
    if e.dim() == 2:  # 1D
        E = (torch.fft.rfft(e, dim=1).abs() ** 2).mean(0)
        T = (torch.fft.rfft(t, dim=1).abs() ** 2).mean(0)
        return {"k": list(range(len(E))), "error": E.tolist(), "truth": T.tolist()}
    Ef, Tf = torch.fft.fft2(e).abs() ** 2, torch.fft.fft2(t).abs() ** 2
    H, W = e.shape[1:]
    ky, kx = torch.meshgrid(torch.fft.fftfreq(H) * H, torch.fft.fftfreq(W) * W, indexing="ij")
    kr = torch.sqrt(kx ** 2 + ky ** 2).round().long()
    n = int(kr.max()) + 1
    E = torch.zeros(n).index_add_(0, kr.flatten(), Ef.mean(0).flatten())
    T = torch.zeros(n).index_add_(0, kr.flatten(), Tf.mean(0).flatten())
    return {"k": list(range(n)), "error": E.tolist(), "truth": T.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ntrain", type=int, default=1000)
    args = ap.parse_args()
    torch.manual_seed(0)
    ckpt = RUNS / args.task / f"{args.model}_n{args.ntrain}.pt"
    task = load(args.task, ntrain=args.ntrain)
    model = build(args.model, task).to(dev).eval()
    state = torch.load(ckpt, map_location=dev, weights_only=False)  # our own checkpoints
    state.pop("_metadata", None)  # neuraloperator stores constructor metadata alongside the weights
    model.load_state_dict(state)
    na, nu = normalisers(task)
    xte = task.x_test
    res = {"task": args.task, "model": args.model}

    pred = predict(model, task, task.a_test, xte, na, nu)
    res["clean"] = rel_l2(pred, task.u_test)
    B = pred.shape[0]  # median relative L1, the metric Lanthaler et al. report for discontinuous problems
    l1 = (pred - task.u_test).reshape(B, -1).abs().sum(1) / task.u_test.reshape(B, -1).abs().sum(1)
    res["rel_l1_median"] = l1.median().item()
    res["spectrum"] = spectrum(pred, task.u_test, task)

    # a few samples for plots, downsampled
    idx = [0, 1, 2]
    if task.kind == "grid1d":
        st = max(1, task.shape[0] // 256)
        res["samples"] = {"x": xte[0, ::st, 0].tolist(), "input": task.a_test[idx, ::st, 0].tolist(),
                          "truth": task.u_test[idx, ::st, 0].tolist(), "pred": pred[idx, ::st, 0].tolist()}
    elif task.kind == "grid2d":
        st = max(1, task.shape[0] // 64)
        last = -1
        res["samples"] = {"truth": task.u_test[idx[:1], ::st, ::st, last].tolist(),
                          "pred": pred[idx[:1], ::st, ::st, last].tolist(),
                          "input": task.a_test[idx[:1], ::st, ::st, last].tolist()}

    if task.kind in ("grid1d", "grid2d"):
        noise = {}
        for lvl in [0.001, 0.01, 0.05]:
            a = task.a_test
            amax = a.reshape(a.shape[0], -1).abs().max(1).values.view(-1, *[1] * (a.dim() - 1))
            an = a + lvl * amax * torch.randn_like(a)
            noise[str(lvl)] = rel_l2(predict(model, task, an, xte, na, nu), task.u_test)
        res["noise"] = noise

    if args.task in RESOLUTIONS:
        out = {}
        for sub in RESOLUTIONS[args.task]:
            n = 8192 // sub if args.task == "burgers" else 420 // sub + 1
            try:
                t2 = load(args.task, ntrain=args.ntrain, sub=sub)
                nu2 = interp(nu, t2.shape)  # training statistics, interpolated to the new grid
                if args.model in DEEPONETS:  # branch reads training-resolution input; trunk queried on the new grid
                    p = predict(model, task, task.a_test, t2.x_test, na, nu2)
                else:
                    p = predict(model, t2, t2.a_test, t2.x_test, interp(na, t2.shape), nu2)
                out[str(n)] = rel_l2(p, t2.u_test)
            except Exception as e:  # some models cannot run off their training grid; record why
                out[str(n)] = f"error: {type(e).__name__}: {str(e)[:120]}"
        res["resolution"] = out
    (RUNS / args.task / f"{args.model}_eval.json").write_text(json.dumps(res))
    print(args.task, args.model, {k: v for k, v in res.items() if k in ("clean", "noise", "resolution")})


def interp(norm, shape):
    """A normaliser on a new grid: the pointwise training mean/std, interpolated."""
    import copy

    import torch.nn.functional as F

    new = copy.copy(norm)
    for name in ("mean", "std"):
        v = getattr(norm, name).movedim(-1, 1)
        v = F.interpolate(v, size=shape, mode="linear" if len(shape) == 1 else "bilinear", align_corners=True)
        setattr(new, name, v.movedim(1, -1))
    return new


if __name__ == "__main__":
    main()
