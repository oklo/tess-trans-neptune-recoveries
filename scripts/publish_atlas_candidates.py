"""Refresh the public candidate figures from the published atlas and reviewed photometry."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.path import Path as MarkerPath
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "outputs"
DEST = ROOT / "planet_nine_atlas_candidates"
SNAPSHOT = "https://oklo.org/planet-nine/coordination/snapshot.json"


def panel(ax, image, signed):
    finite = image[np.isfinite(image)]
    if not finite.size:
        return
    if signed:
        limit = max(np.percentile(abs(finite), 98), 1e-12)
        ax.imshow(image, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
    else:
        lo, scale, hi = np.percentile(finite, [10, 70, 99.7])
        ax.imshow(np.arcsinh((image-lo)/max(scale-lo, 1)), origin="lower", cmap="gray",
                  vmin=0, vmax=max(.01, np.arcsinh((hi-lo)/max(scale-lo, 1))))
    ax.set_xticks([])
    ax.set_yticks([])


def crosshair(ax, x, y):
    arms = MarkerPath([(-1, 0), (-.38, 0), (.38, 0), (1, 0),
                       (0, -1), (0, -.38), (0, .38), (0, 1)],
                      [MarkerPath.MOVETO, MarkerPath.LINETO] * 4)
    ax.plot(x, y, marker=arms, markersize=22, markeredgewidth=1.35,
            color="#ffdf38", linestyle="none", scalex=False, scaley=False,
            path_effects=[effects.withStroke(linewidth=2.4, foreground="#302a18")])


def render(path, code, row, files):
    sectors = sorted(row["sectors"], key=int)
    n = len(sectors)
    with_pixels = any(p.exists() for p in files)
    context = {"field_image", "field_residual"}
    columns = [("field_image", "Star field"), ("field_residual", "Stars subtracted")]
    columns.extend([("signal", "Along the path"), ("response", "Point-source model")])
    fig = plt.figure(figsize=(2.2*len(columns)+.4, 2*n+2.7) if with_pixels else (8, 3.1), layout="constrained")
    gs = fig.add_gridspec(n+1, len(columns), height_ratios=[1]*n+[.88]) if with_pixels else fig.add_gridspec(1, 1)
    if with_pixels:
        for i, (sid, file) in enumerate(zip(sectors, files)):
            if not file.exists():
                continue
            with np.load(file) as z:
                for j, (key, title) in enumerate(columns):
                    ax = fig.add_subplot(gs[i, j])
                    if i == 0:
                        ax.set_title(title, fontsize=11)
                    if j == 0:
                        ax.set_ylabel(f"S{sid}", rotation=0, labelpad=15)
                    if key in context:
                        if key not in z:
                            ax.set_axis_off()
                            ax.text(.5, .5, "Rebuilding image", ha="center", va="center", fontsize=9)
                            continue
                        panel(ax, z[key], key == "field_residual")
                        track = z["path"]
                        if key == "field_image":
                            ax.plot(*track.T, color="#ffdf38", linewidth=1.8,
                                    path_effects=[effects.withStroke(linewidth=3, foreground="#302a18")])
                        else:
                            crosshair(ax, *track[len(track) // 2])
                    else:
                        weight = z["weight"][0]
                        values = z["numerator"][0] if key == "signal" else z["response"]
                        panel(ax, np.divide(values, weight, where=weight>0, out=np.full_like(weight,np.nan)), True)
                        crosshair(ax, (weight.shape[1]-1)/2, (weight.shape[0]-1)/2)
    ax = fig.add_subplot(gs[-1, :])
    series = [row["sectors"][k] for k in sectors]
    ax.axhline(0, color="0.65", lw=.8)
    ax.errorbar(range(n), [s["flux"] for s in series], yerr=[s["sigma"] for s in series],
                fmt="o", color="#225f83", capsize=4)
    ax.set_xlim(-.5, n-.5)
    ax.set_xticks(range(n), [f"Sector {k}" for k in sectors])
    ax.set_ylabel("Flux (electrons/s)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{code}  ·  candidate {row['trace']}", fontsize=16)
    if with_pixels:
        ax.set_xlabel(("Stars subtracted: position halfway through the sampled path. " if "field_residual" in context else "") + "Each image is scaled separately.", fontsize=9)
    tmp = path.with_suffix(".partial")
    fig.savefig(tmp, format="png", dpi=max(135, int(np.ceil(928 / fig.get_figwidth()))))
    plt.close(fig)
    tmp.replace(path)


def main():
    snapshot = json.loads(subprocess.check_output(["/usr/bin/curl", "-fsS", "--max-time", "30", SNAPSHOT]))
    if snapshot["schema_version"] != 1 or not snapshot["fields"]:
        raise ValueError("Invalid atlas")
    DEST.mkdir(exist_ok=True)
    orbit_file = DEST/"orbits.json"
    previous = json.loads(orbit_file.read_text()) if orbit_file.exists() else {}
    orbits = {}
    helper = Path(__file__).with_name("atlas_trial_orbits.py")
    keep = set()
    for f in snapshot["fields"]:
        code = f["code"]
        if f["state"] != "needs_followup" or not re.fullmatch(r"N[0-9]+", code):
            continue
        base = ROOT/"planet_nine_backlog_service"/"visual"/str(f.get("task_id"))
        if not (base/"plans"/f"{code}.json").exists():
            base = ROOT/"planet_nine_visual_followup_20261123"
        plan = base/"plans"/f"{code}.json"
        ids = f.get("retained_candidate_flags", list(range(1, f["candidate_count"]+1)))
        if plan.exists():
            model = json.loads(plan.read_text())
            digest = hashlib.sha256(plan.read_bytes() + helper.read_bytes()).hexdigest()
            record = previous.get(code, {})
            if record.get("digest") != digest:
                values = json.loads(subprocess.check_output([model["python"], str(helper), str(plan)], timeout=120))
                record = {"digest": digest, "rows": values}
            orbits[code] = record
        for trace in ids:
            path = DEST/f"{code}-{trace}.png"
            keep.add(path)
            if not plan.exists():
                continue
            row = next((r for r in json.loads(plan.read_text())["rows"] if r["trace"] == trace), None)
            if not row:
                continue
            files = [base/"measurements"/f"{code}-{trace}-{s}.npz" for s in sorted(row["sectors"], key=int)]
            rebuilt = ROOT/"planet_nine_candidate_context"/code/"measurements"
            files = [rebuilt/f.name if (rebuilt/f.name).is_file() else f for f in files]
            newest = max(p.stat().st_mtime for p in [Path(__file__), plan, *files] if p.exists())
            if not path.exists() or path.stat().st_mtime < newest:
                render(path, code, row, files)
    for p in DEST.glob("N*-*.png"):
        if p not in keep:
            p.unlink()
    encoded = json.dumps(orbits, sort_keys=True, allow_nan=False)
    if not orbit_file.exists() or orbit_file.read_text() != encoded:
        tmp = orbit_file.with_suffix(".partial")
        tmp.write_text(encoded)
        tmp.replace(orbit_file)
    if "--publish" in sys.argv:
        subprocess.run(["rsync","-rt","--delete","--include=orbits.json","--include=*.png","--exclude=*","-e",
                        "ssh -o BatchMode=yes -o ConnectTimeout=12",str(DEST)+"/",
                        "oklo:.planet-nine/candidate-figures/"], check=True)


if __name__ == "__main__":
    main()
