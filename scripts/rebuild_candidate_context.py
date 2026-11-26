"""Restore missing atlas panels using a field's archived visual-review plan."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def rebuild(plan_path, traces):
    plan = json.loads(plan_path.read_text())
    field = plan["field"]
    if plan_path.name != f"{field}.json":
        raise ValueError("Field and plan filename differ")
    base = OUTPUTS / "planet_nine_candidate_context" / field
    images = base / "measurements"
    images.mkdir(exist_ok=True, parents=True)
    wanted = set(traces)
    rows = []
    for row in plan["rows"]:
        if wanted and row["trace"] not in wanted:
            continue
        for sector in row["sectors"]:
            filename = f"{field}-{row['trace']}-{sector}.npz"
            path = images / filename
            if not path.exists():
                path = plan_path.parents[1] / "measurements" / filename
            if path.exists():
                with np.load(path) as pixels:
                    if {"field_image", "path", "field_residual"} <= set(pixels.files):
                        continue
            rows.append(row)
            break
    if not rows:
        return
    seed = Path(__file__).with_name("candidate_review")
    for name in ("fetch.py", "sector.py"):
        shutil.copyfile(seed / name, base / name)
    plans = base / "plans"
    plans.mkdir(exist_ok=True)
    plan["rows"] = rows
    dest = plans / plan_path.name
    dest.write_text(json.dumps(plan))
    python = plan["python"]
    digest = subprocess.check_output(
        [python, "-c", "from p9search.provenance import source_hash; print(source_hash())"],
        text=True,
    ).strip()
    # The visual-review runtime checks the raw image hashes and reproduces
    # archived sector fluxes before saving any reconstructed panels.
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    subprocess.run([python, str(ROOT / "scripts/review_visual_trace.py"), "--prepare", str(dest), digest], env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("traces", type=int, nargs="*")
    args = parser.parse_args()
    rebuild(args.plan.resolve(), args.traces)
