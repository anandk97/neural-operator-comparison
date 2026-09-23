"""Watch the run directory and update the website whenever a task has all its models trained.

For each newly completed task: run evaluate.py on its checkpoints, export web data, copy it to the site repo, commit
and push. Checks every 10 minutes; exits once every task has been published.

    uv run python publish.py
"""

import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from export_web import MODELS, TASKS, pre_fix

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
SITE = ROOT.parent / "anandk97.github.io"
STATE = ROOT / "logs" / "published.txt"
GRID_ONLY = {"fno", "local_fno", "kano"}


def expected(task):
    return [m for m in MODELS if not (task == "elasticity" and m in GRID_ONLY)]


def complete(task):
    return all((RUNS / task / f"{m}_n1000.json").exists() for m in expected(task))


def run(cmd, cwd=ROOT):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def publish(task):
    for m in expected(task):
        if not (RUNS / task / f"{m}_eval.json").exists() and not pre_fix(task, m):
            run(["uv", "run", "python", "-W", "ignore", "evaluate.py", "--task", task, "--model", m])
    run(["uv", "run", "python", "export_web.py"])
    shutil.copy(ROOT / "web" / "operators-data.json", SITE / "projects" / "operators" / "operators-data.json")
    run(["git", "add", "projects/operators/operators-data.json"], cwd=SITE)
    msg = f"Neural operators: {task} results\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    c = run(["git", "commit", "-m", msg], cwd=SITE)
    p = run(["git", "push"], cwd=SITE)
    print(f"{datetime.now():%H:%M} published {task}: commit rc={c.returncode} push rc={p.returncode}", flush=True)


if __name__ == "__main__":
    done = set(STATE.read_text().split()) if STATE.exists() else set()
    while len(done) < len(TASKS):
        for t in TASKS:
            if t not in done and complete(t):
                publish(t)
                done.add(t)
                STATE.write_text("\n".join(sorted(done)))
        time.sleep(600)
    print("all tasks published", flush=True)
