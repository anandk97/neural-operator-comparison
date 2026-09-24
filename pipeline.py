"""Run the follow-up stages after the main queue: wait for any running `run_all.py main`, then re-run superseded
runs, the Shift-DeepONet ablation, evaluations and data scaling, and push web data at each stage.

    uv run python pipeline.py
"""

import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent
SITE = ROOT.parent / "anandk97.github.io"


def main_running():
    q = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*run_all.py main*' } | Measure-Object | Select -Expand Count")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", q], capture_output=True, text=True).stdout.strip()
    return out not in ("", "0")


def stage(name):
    print(f"== {name}", flush=True)
    with open(ROOT / "logs" / f"{name}.log", "a") as log:
        subprocess.run(["uv", "run", "python", "run_all.py", name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def push(msg):
    subprocess.run(["uv", "run", "python", "export_web.py"], cwd=ROOT, capture_output=True)
    shutil.copy(ROOT / "web" / "operators-data.json", SITE / "projects" / "operators" / "operators-data.json")
    subprocess.run(["git", "add", "projects/operators/operators-data.json"], cwd=SITE)
    subprocess.run(["git", "commit", "-q", "-m", f"Neural operators: {msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"], cwd=SITE)
    r = subprocess.run(["git", "push", "-q"], cwd=SITE)
    print(f"pushed {msg}: rc={r.returncode}", flush=True)


if __name__ == "__main__":
    while main_running():
        time.sleep(300)
    stage("main")  # picks up the superseded runs, which no longer have results
    stage("ablation")
    stage("eval")
    push("re-runs")
    stage("scaling")
    push("data scaling")
    print("pipeline done", flush=True)
