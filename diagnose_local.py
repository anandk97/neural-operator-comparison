"""Why does the local-kernel FNO break on finer grids? Tests on the trained checkpoints (no retraining).

Burgers (trained at 1024 points; 1D, so only Fourier + finite-difference branches):
  native        the dataset's own input at the test resolution
  bandlimited   the 1024-point input, Fourier-interpolated to the test resolution: no content the model has not seen
  no_fd         finite-difference branch output replaced by zeros
Darcy (trained at 85 x 85; Fourier + finite-difference + local integral (DISCO) branches):
  native        as trained
  no_fd, no_disco, no_local   branch outputs replaced by zeros
  disco_rebuilt the DISCO kernels recomputed for the test grid (neuraloperator precomputes them for the training grid)
Both tasks:
  fd_lowpass    the input to each finite-difference kernel low-pass filtered to the training grid's Nyquist band,
                i.e. the derivative only sees scales it could see during training
  fd_rms_layers RMS of each finite-difference branch output, layer by layer

    uv run python diagnose_local.py
"""

import json
import types
from pathlib import Path

import torch

import noc  # noqa: F401  (puts the vendored torch-harmonics subset on the path before neuraloperator imports)
from neuralop.layers.differential_conv import FiniteDifferenceConvolution
from neuralop.layers.discrete_continuous_convolution import EquidistantDiscreteContinuousConv2d

from evaluate import interp, normalisers, predict, rel_l2
from noc.data import load
from noc.models import build

RUNS = Path(__file__).parent / "runs"
dev = "cuda"


def load_model(name, task, ckpt_task):
    m = build(name, task).to(dev).eval()
    state = torch.load(RUNS / ckpt_task / f"{name}_n1000.pt", map_location=dev, weights_only=False)
    state.pop("_metadata", None)
    m.load_state_dict(state)
    return m


def zero(model, cls):
    return [mod.register_forward_hook(lambda _m, _i, out: torch.zeros_like(out))
            for mod in model.modules() if isinstance(mod, cls)]


def fd_rms(model):
    stats = []

    def hook(_m, _i, out):
        stats.append(out.pow(2).mean().sqrt().item())
    hs = [mod.register_forward_hook(hook) for mod in model.modules() if isinstance(mod, FiniteDifferenceConvolution)]
    return stats, hs


def lowpass_fd(model, keep_frac):
    """Pre-hook: keep only the lowest `keep_frac` of Fourier modes (per axis) in each finite-difference input."""
    def pre(_m, args):
        x, gw = args
        dims = tuple(range(2, x.dim()))
        xh = torch.fft.rfftn(x, dim=dims)
        mask = torch.ones(xh.shape[2:], device=x.device, dtype=torch.bool)
        for ax, n in enumerate(x.shape[2:]):
            k = torch.fft.rfftfreq(n, device=x.device) if ax == len(dims) - 1 else torch.fft.fftfreq(n, device=x.device).abs()
            kmask = k <= 0.5 * keep_frac + 1e-9  # cycles per sample; 0.5 is the Nyquist frequency of this grid
            shape = [1] * len(dims)
            shape[ax] = -1
            mask = mask & kmask.view(shape)
        return (torch.fft.irfftn(xh * mask, s=x.shape[2:], dim=dims), gw)
    return [mod.register_forward_pre_hook(pre) for mod in model.modules() if isinstance(mod, FiniteDifferenceConvolution)]


def bandlimit(a, n):
    """Fourier-interpolate [N, s, 1] periodic samples to n points."""
    s = a.shape[1]
    ah = torch.fft.rfft(a[..., 0].double(), dim=1)
    out = torch.zeros(a.shape[0], n // 2 + 1, dtype=ah.dtype)
    out[:, :ah.shape[1]] = ah
    return (torch.fft.irfft(out, n=n, dim=1) * (n / s)).float()[..., None]


def burgers():
    base = load("burgers")
    na, nu = normalisers(base)
    res = {}
    for name in ["fno", "local_fno"]:
        model = load_model(name, base, "burgers")
        for sub in [8, 4, 2]:
            t = load("burgers", sub=sub)
            n = t.shape[0]
            nai, nui = interp(na, t.shape), interp(nu, t.shape)
            r = {}
            stats, hs = fd_rms(model)
            r["native"] = rel_l2(predict(model, t, t.a_test, t.x_test, nai, nui), t.u_test)
            r["fd_rms_native"] = sum(stats) / max(len(stats), 1)
            stats.clear()
            abl = bandlimit(base.a_test, n)
            r["bandlimited"] = rel_l2(predict(model, t, abl, t.x_test, nai, nui), t.u_test)
            r["fd_rms_bandlimited"] = sum(stats) / max(len(stats), 1)
            for h in hs:
                h.remove()
            if name == "local_fno":
                hs = zero(model, FiniteDifferenceConvolution)
                r["no_fd"] = rel_l2(predict(model, t, t.a_test, t.x_test, nai, nui), t.u_test)
                for h in hs:
                    h.remove()
                hs = lowpass_fd(model, base.shape[0] / n)
                stats, hs2 = fd_rms(model)
                r["fd_lowpass"] = rel_l2(predict(model, t, t.a_test, t.x_test, nai, nui), t.u_test)
                for h in hs + hs2:
                    h.remove()
                stats2, hs2 = fd_rms(model)
                predict(model, t, t.a_test[:4], t.x_test, nai, nui, bs=4)
                r["fd_rms_layers"] = stats2[:]
                for h in hs2:
                    h.remove()
            res.setdefault(name, {})[n] = r
            print("burgers", name, n, {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in r.items()}, flush=True)
    return res


def darcy():
    base = load("darcy")
    na, nu = normalisers(base)
    res = {}
    model = load_model("local_fno", base, "darcy")
    for sub in [5, 3, 2]:
        t = load("darcy", sub=sub)
        n = t.shape[0]
        nai, nui = interp(na, t.shape), interp(nu, t.shape)
        r = {}

        def run(m):
            return rel_l2(predict(m, t, t.a_test, t.x_test, nai, nui), t.u_test)
        r["native"] = run(model)
        for label, classes in [("no_fd", [FiniteDifferenceConvolution]),
                               ("no_disco", [EquidistantDiscreteContinuousConv2d]),
                               ("no_local", [FiniteDifferenceConvolution, EquidistantDiscreteContinuousConv2d])]:
            hs = [h for c in classes for h in zero(model, c)]
            r[label] = run(model)
            for h in hs:
                h.remove()
        # rebuild with DISCO kernels computed for this grid; keep the finite-difference scaling of the original
        rebuilt = load_model("local_fno", t, "darcy")
        ratio = base.shape[0] / n
        for mod in rebuilt.modules():
            if isinstance(mod, FiniteDifferenceConvolution):
                orig = mod.forward
                mod.forward = types.MethodType(lambda self, x, gw, _o=orig, _r=ratio: _o(x, gw * _r), mod)
        r["disco_rebuilt"] = run(rebuilt)
        hs = lowpass_fd(model, base.shape[0] / n)
        r["fd_lowpass"] = run(model)
        for h in hs:
            h.remove()
        stats, hs = fd_rms(model)
        predict(model, t, t.a_test[:4], t.x_test, nai, nui, bs=4)
        r["fd_rms_layers"] = stats[:]
        for h in hs:
            h.remove()
        res[n] = r
        print("darcy local_fno", n, {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in r.items()}, flush=True)
    return res


if __name__ == "__main__":
    out = {"burgers": burgers(), "darcy": darcy()}
    (RUNS / "diagnose_local.json").write_text(json.dumps(out, indent=1))
