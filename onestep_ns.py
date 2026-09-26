"""One-step (teacher-forced) test error on Navier-Stokes, to separate one-step accuracy from error compounding over
the 10-step rollout. For each test trajectory and each of the 10 target frames, the model sees the true previous 10
frames.

    uv run python onestep_ns.py
"""

import json
from pathlib import Path

import torch

from evaluate import normalisers, predict, rel_l2
from noc.data import load
from noc.models import build

RUNS = Path(__file__).parent / "runs"
ARCH = {"transolver_n1000_authors": {"layers": 8, "width": 256, "slice_num": 32},
        "transolver_pp_n1000_authors": {"layers": 8, "width": 256, "slice_num": 32}}

task = load("ns")
na, nu = normalisers(task)
full = torch.cat([task.a_test, task.u_test], -1)
out = {}
for run in ["fno_n1000", "kano_n1000", "local_fno_n1000", "deeponet_n1000", "shift_deeponet_n1000", "transolver_n1000",
            "transolver_pp_n1000", "transolver_n1000_authors", "transolver_pp_n1000_authors"]:
    name = run.split("_n1000")[0]
    model = build(name, task, **ARCH.get(run, {})).cuda().eval()
    state = torch.load(RUNS / "ns" / f"{run}.pt", map_location="cuda", weights_only=False)
    state.pop("_metadata", None)
    model.load_state_dict(state)
    one = task.__class__(**{**task.__dict__, "rollout": 1})  # predict() then does a single step
    errs = []
    with torch.no_grad():
        for k in range(10):
            a, u = full[..., k:k + 10], full[..., k + 10:k + 11]
            errs.append(rel_l2(predict(model, one, a, task.x_test, na, nu), u))
    roll = json.loads((RUNS / "ns" / f"{run}.json").read_text())["test_rel_l2"]
    out[run] = {"one_step": sum(errs) / len(errs), "rollout": roll}
    print(f"{run:30s} one-step test {out[run]['one_step']:.4f}  10-step rollout {roll:.4f}  ratio {roll / out[run]['one_step']:.1f}", flush=True)
(RUNS / "ns" / "onestep.json").write_text(json.dumps(out, indent=1))
