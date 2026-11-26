"""Summarize the trial state used to align the TESS images."""
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
from p9search.science.geometry import SkyFrame
from p9search.storage import DiskBudget, GIB
from p9search.tasks import fetch_bundle


def properties(frame, row):
    x, y, east, north, inverse_distance = row["parameters"]
    distance = 1 / inverse_distance
    pos = frame.observer + distance * frame.ray([x, y])[0]
    vel = distance * (np.array([east, north]) @ frame.tangent)
    mu = .01720209895**2
    h = np.cross(pos, vel)
    r = np.linalg.norm(pos)
    a = 1 / (2/r - np.dot(vel, vel)/mu)
    e = np.linalg.norm(np.cross(vel, h)/mu - pos/r)
    tilt = np.deg2rad(23.4392911)
    pole = [0, -np.sin(tilt), np.cos(tilt)]
    inc = np.rad2deg(np.arccos(np.clip(h@pole/np.linalg.norm(h), -1, 1))) if np.linalg.norm(h) else None
    sectors = list(row["sectors"].values())
    flux = np.average([s["flux"] for s in sectors], weights=[s["sigma"]**-2 for s in sectors])
    mag = 20.44 - 2.5*math.log10(flux) if flux > 0 else None
    # Reflected-light diameter relation at opposition, using V - T = 0.5.
    radius = 664.5*r*distance*10**(-.2*(mag+.5))/(6378.1*math.sqrt(.4)) if mag is not None else None
    return dict(T=mag, radius=radius, distance=distance, a=a, e=e, i=inc)


if __name__ == "__main__":
    plan = json.loads(Path(sys.argv[1]).read_text())
    with tempfile.TemporaryDirectory() as tmp:
        if plan.get("task"):
            data = fetch_bundle(json.loads(Path(plan["task"]).read_text()),Path(tmp),DiskBudget(Path(tmp), GIB, 25*GIB))
        else:
            data = Path(plan["data"])
        frame = SkyFrame(data/f"s{plan['reference_sector']:04d}")
        print(json.dumps({row["trace"]: properties(frame,row) for row in plan["rows"]}, allow_nan=False))
