"""Collect results into web/operators-data.json for the website's Projects page.

    uv run python export_web.py
"""

import json
from pathlib import Path

RUNS = Path(__file__).parent / "runs"
OUT = Path(__file__).parent / "web" / "operators-data.json"

TASKS = ["burgers", "advection", "darcy", "ns", "airfoil", "elasticity"]
MODELS = ["deeponet", "shift_deeponet", "fno", "local_fno", "deepokan", "kano", "transolver", "transolver_pp"]

# Published relative L2 on the same datasets (see LITERATURE.md). Keys are our model names.
PUBLISHED = {
    "darcy": {"fno": 0.0108, "transolver": 0.0057, "transolver_pp": 0.0049},
    "ns": {"fno": 0.1556, "transolver": 0.0900, "transolver_pp": 0.0719},
    "airfoil": {"transolver": 0.0053, "transolver_pp": 0.0048},
    "elasticity": {"transolver": 0.0064, "transolver_pp": 0.0052},
}


def rnd(v, n=5):
    if isinstance(v, float):
        return float(f"{v:.{n}g}")
    if isinstance(v, list):
        return [rnd(x, n) for x in v]
    if isinstance(v, dict):
        return {k: rnd(x, n) for k, x in v.items()}
    return v


def main():
    data = {"tasks": TASKS, "models": MODELS, "published": PUBLISHED, "main": {}, "scaling": {}, "eval": {},
            "pinn": {}, "curves": {}}
    for t in TASKS:
        for m in MODELS:
            f = RUNS / t / f"{m}_n1000.json"
            if f.exists():
                r = json.loads(f.read_text())
                data["main"].setdefault(t, {})[m] = {
                    "err": r["test_rel_l2"], "params": r["params"], "train_min": r["train_seconds"] / 60,
                    "infer_ms": r["infer_ms_per_sample"], "epochs": r["epochs"]}
                h = r["history"]
                st = max(1, len(h) // 60)
                data["curves"].setdefault(t, {})[m] = h[::st]
            for n in (100, 300):
                f = RUNS / t / f"{m}_n{n}.json"
                if f.exists():
                    data["scaling"].setdefault(t, {}).setdefault(m, {})[n] = json.loads(f.read_text())["test_rel_l2"]
            f = RUNS / t / f"{m}_eval.json"
            if f.exists():
                e = json.loads(f.read_text())
                sp = e["spectrum"]
                kmax = min(len(sp["k"]), 128)
                e["spectrum"] = {k: v[:kmax] for k, v in sp.items()}
                data["eval"].setdefault(t, {})[m] = e
        for m in ("pinn", "piratenet"):
            f = RUNS / t / f"{m}_instances.json"
            if f.exists():
                data["pinn"].setdefault(t, {})[m] = json.loads(f.read_text())
        for m in MODELS:  # scaling curves include the 1000-sample point
            if t in data["scaling"] and m in data["scaling"][t] and m in data["main"].get(t, {}):
                data["scaling"][t][m][1000] = data["main"][t][m]["err"]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rnd(data), separators=(",", ":")))
    print(f"{OUT} {OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
