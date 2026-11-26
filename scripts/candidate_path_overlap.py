"""Compare paths on the detector in the sectors contributing positive flux."""
import json
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
from p9search.science.geometry import SectorGeometry, SkyFrame
from p9search.storage import DiskBudget, GIB
from p9search.tasks import fetch_bundle


def compare(plan, ids):
    rows = [row for row in plan["rows"] if row["trace"] in ids]
    if len(rows) != len(set(ids)):
        raise ValueError("Missing photometry for an active path")
    pairs = list(combinations(rows, 2))
    if not pairs:
        return []
    answer = []
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        if plan.get("task"):
            task = json.loads(Path(plan["task"]).read_text())
            data = fetch_bundle(task, folder, DiskBudget(folder, GIB, 25 * GIB))
        else:
            data = Path(plan["data"])
        frame = SkyFrame(data / f"s{plan['reference_sector']:04d}")
        for sid in sorted({int(s) for row in rows for s in row["sectors"]}):
            g = SectorGeometry(data / f"s{sid:04d}", frame)
            tracks = {row["trace"]: g.exact_positions(row["parameters"], kepler=True)[g.geom["keep"]]
                      for row in rows if str(sid) in row["sectors"]}
            for first, second in pairs:
                a, b = first["trace"], second["trace"]
                if a not in tracks or b not in tracks or not len(tracks[a]):
                    continue
                sep = np.linalg.norm(tracks[a] - tracks[b], axis=1)
                snr = [r["sectors"][str(sid)]["flux"] / r["sectors"][str(sid)]["sigma"]
                       for r in (first, second)]
                answer.append(dict(paths=[a, b], sector=sid, median_pixels=float(np.median(sep)),
                                   fraction_within_two_pixels=float(np.mean(sep <= 2)), snr=snr))
    return answer


if __name__ == "__main__":
    print(json.dumps(compare(json.loads(Path(sys.argv[1]).read_text()), list(map(int, sys.argv[2:]))),
                     allow_nan=False))
