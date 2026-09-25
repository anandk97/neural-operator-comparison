"""Run the experiment queue sequentially, skipping runs whose results already exist.

    uv run python run_all.py main      # every model on every task, 1000 training samples
    uv run python run_all.py scaling   # 1D tasks at 100 / 300 / 1000 samples
    uv run python run_all.py pinn      # PINN and PirateNet on 3 test instances per task, 10k steps each
    uv run python run_all.py ablation  # Shift-DeepONet with Lanthaler et al.'s shared trunk, on the 1D tasks
    uv run python run_all.py authors   # Transolver / Transolver++ at the authors' settings, 500 epochs (Darcy, NS)
    uv run python run_all.py eval      # noise / resolution / spectrum tests on every trained checkpoint
"""

import subprocess
import sys
from pathlib import Path

RUNS = Path(__file__).parent / "runs"
OPERATORS = ["fno", "local_fno", "deeponet", "shift_deeponet", "deepokan", "kano", "transolver", "transolver_pp"]
GRID_ONLY = {"fno", "local_fno", "kano"}
EPOCHS = {"burgers": 300, "advection": 300, "darcy": 100, "ns": 200, "airfoil": 100, "elasticity": 300}


def queue(which):
    if which == "main":
        for task in ["burgers", "advection", "elasticity", "darcy", "airfoil", "ns"]:
            for m in OPERATORS:
                if task == "elasticity" and m in GRID_ONLY:
                    continue
                yield f"{task}/{m}_n1000.json", ["train.py", "--task", task, "--model", m, "--epochs", str(EPOCHS[task])]
    elif which == "scaling":
        for task in ["burgers", "advection"]:
            for n in [100, 300]:
                for m in OPERATORS:
                    yield f"{task}/{m}_n{n}.json", ["train.py", "--task", task, "--model", m, "--ntrain", str(n),
                                                   "--epochs", str(EPOCHS[task])]
    elif which == "ablation":
        for task in ["advection", "burgers"]:
            yield f"{task}/shift_deeponet_shared_n1000.json", ["train.py", "--task", task, "--model",
                                                               "shift_deeponet_shared", "--epochs", "300"]
    elif which == "authors":
        # From thuml's scripts (Transolver) and the Transolver++ paper's per-benchmark table. NS keeps our teacher-forced
        # training; the authors train on the full rollout, which is about 10x the cost.
        cfgs = [("darcy", "transolver", 4, 0.1, "layers=8,width=128,slice_num=64"),
                ("darcy", "transolver_pp", 4, 0.1, "layers=4,width=128,slice_num=64"),
                ("ns", "transolver", 2, 1e9, "layers=8,width=256,slice_num=32"),
                ("ns", "transolver_pp", 4, 1e9, "layers=8,width=256,slice_num=32")]
        for task, m, bs, clip, arch in cfgs:
            yield f"{task}/{m}_n1000_authors.json", ["train.py", "--task", task, "--model", m, "--epochs", "500",
                                                    "--batch", str(bs), "--clip", str(clip), "--arch", arch,
                                                    "--tag", "authors"]
    elif which == "eval":
        for ckpt in sorted(RUNS.glob("*/*_n1000.pt")):
            task, model = ckpt.parent.name, ckpt.stem.rsplit("_n", 1)[0]
            yield f"{task}/{model}_eval.json", ["evaluate.py", "--task", task, "--model", model]
    elif which == "pinn":
        for task in ["burgers", "advection", "darcy"]:
            for m in ["pinn", "piratenet"]:
                yield f"{task}/{m}_instances.json", ["pinn.py", "--task", task, "--model", m, "--instances", "3",
                                                       "--iters", "10000"]


if __name__ == "__main__":
    for out, cmd in queue(sys.argv[1]):
        if (RUNS / out).exists():
            continue
        print(">>", " ".join(cmd), flush=True)
        r = subprocess.run(["uv", "run", "python", "-W", "ignore", *cmd])
        if r.returncode:
            print("!! failed:", " ".join(cmd), flush=True)
