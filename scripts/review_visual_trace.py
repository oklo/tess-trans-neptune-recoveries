"""Prepare sector images for new open traces and record a read-only image review."""

import csv
import fcntl
import time
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from p9search.storage import read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kbo_tess.repeat_support import assess as assess_repeat_support

BASE = ROOT / "outputs/planet_nine_backlog_service"
SEED = ROOT / "outputs/planet_nine_visual_followup_20261123"
STORE = BASE / "visual"
PROMPT = """Inspect each sector and compare the measured patch to the point-source model.
Panels have separate display stretches. Check whether the positive core is centered,
whether a bright star, subtraction ridge or off-center knot lies along the trial path,
and whether the morphology repeats in the other sectors. The first image is the
target trace; any later images show nearby alternative orbits in the same field. A real filtered point source
can have negative lobes. For a weak sector, check coverage and predicted S/N at the
reference sector's brightness before interpreting the missing signal. These are formal
uncertainties and the source may vary. Describe the image evidence separately.
Assess whether alternative paths pick up the same light in the sectors that support
a detection. Compare their detector separations, the measured cores, and the point-source
widths. Distinguish shared, distinct, and uncertain image evidence. Paths that separate
only in a sector with no signal do not supply independent positive evidence.
Name the relevant path numbers and sectors in your observation.
For public display, ask whether a reader could reasonably interpret this as a distinct
moving point source after seeing the full evidence. Set public_candidate true only
for a physically plausible, centered source repeating along one path across sectors,
with consistent flux given the local noise, complete comparison images, and no
unexplained sensitive dropout or convincing stellar residual. Mark ambiguous cases
false. List alternative path numbers drawing on the same positive images in
shared_paths. When independence is unclear, keep public_candidate false.
Use the nearby point-source comparisons and sector S/N as context, not probability.
Report likely artifact only for a concrete image-based explanation. Otherwise distinguish
inconclusive from worth follow-up. Give the sector evidence briefly. No tools are needed.
This is triage; the reviewer cannot award credit or change the public ledger."""
SCHEMA = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "string",
            "enum": ["likely artifact", "inconclusive", "worth follow-up"],
        },
        "observation": {"type": "string"},
        "public_candidate": {"type": "boolean"},
        "shared_paths": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["assessment", "observation", "public_candidate", "shared_paths"],
    "additionalProperties": False,
}


def previous_reviews():
    manifest = read_json(SEED / "review-manifest.json")
    catalog = SEED / "visual-review.csv"
    if sha256(catalog) != manifest["files"]["visual-review.csv"]:
        raise ValueError("Visual review catalog changed")
    with catalog.open() as stream:
        return {(r["source_search"], int(r["trace"])) for r in csv.DictReader(stream)}


def inspect_image(image, context, neighbors=()):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_json(tmp / "schema.json", SCHEMA)
        args = [
            str(Path.home() / ".local/bin/codex"),
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--cd",
            str(tmp),
            "-m",
            "gpt-6-luna",
            "--output-schema",
            str(tmp / "schema.json"),
            "--output-last-message",
            str(tmp / "result.json"),
            *[
                arg
                for path in [image, *neighbors]
                for arg in ("-i", str(path.resolve()))
            ],
            "--",
            PROMPT + "\n" + json.dumps(context),
        ]
        proc = subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=240,
        )
        if proc.returncode:
            raise RuntimeError("Visual assessment failed: " + proc.stderr[-500:])
        result = read_json(tmp / "result.json")
        if (
            result["assessment"] not in SCHEMA["properties"]["assessment"]["enum"]
            or not result["observation"].strip()
        ):
            raise ValueError("Incomplete image assessment")
        return dict(
            **result, image_sha256=sha256(image), reviewer="gpt-6-luna", public_credit=0
        )


def independent_checks(task_id, code, trace, parameters):
    checks = []
    manifest = BASE / "additional-holdouts" / f"{code}-{trace}.json"
    expected = {
        (row["sector"], row["camera"], row["ccd"])
        for row in (read_json(manifest) if manifest.is_file() else [])
        if row.get("task_id") == task_id and row.get("parameters") == parameters
        and row.get("field") == code and row.get("flag") == trace
    }
    for path in sorted((BASE / "measurements").glob(f"holdout-{code}-{trace}-*.json")):
        item = read_json(path)
        if (item.get("state") != "measured" or item.get("parameters") != parameters
                or item.get("field") != code or item.get("trace") != trace
                or not item.get("independent_sector")):
            continue
        detector = (item.get("sector"), item.get("camera"), item.get("ccd"))
        if item.get("task_id") != task_id and ("task_id" in item or detector not in expected):
            continue
        check = {k: item[k] for k in ("sector", "flux", "translated_track_fluxes", "injected_gain")}
        check["measurement_sha256"] = sha256(path)
        scan = item.get("translation_scan")
        if scan:
            peak, = (p["flux"] for p in scan["peaks"] if p["center"] == [0, 0])
            controls = [p["flux"] for p in scan["peaks"] if p["center"] != [0, 0]]
            check.update(radius_px=scan["radius_px"], brightest_target_peak=peak,
                         controls_equal_or_brighter=sum(c >= peak for c in controls),
                         control_regions=len(controls))
        checks.append(check)
    return checks


def nearby_alternatives(candidates, flags, trace):
    def path(i):
        row = candidates[i - 1]
        return (tuple(row["parameters"]), row.get("orientation", 1), row.get("sign", 1))

    target = path(trace)
    unique = {}
    for i in flags:
        key = path(i)
        if key != target:
            unique.setdefault(key, i)
    nearest = sorted(unique, key=lambda key: sum(
        (a - b) ** 2 for a, b in zip(target[0][:2], key[0][:2])
    ))
    return [unique[key] for key in nearest[:3]]


def review(report_dir, code, flags, trace, receipt):
    task = read_json(report_dir / "task.json")
    sub = read_json(report_dir / "submission.json")
    payload = sub["result_json"]
    assert hashlib.sha256(payload.encode()).hexdigest() == sub["submission_digest"]
    report = json.loads(payload)
    tid = task["task_id"]
    assert tid == report["task_id"] and receipt.parent.name == tid
    folder = receipt.parent
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / "images" / f"{code}-{trace}.png"
    python = read_json(
        ROOT / "outputs/planet_nine_travel_reviewer_20261120/runtime.json"
    )["interpreters"][report["source_sha256"]]
    if not image.exists() or not (folder / "comparison.csv").exists():
        for name in ("fetch.py", "sector.py", "gallery.py", "compare.py"):
            shutil.copyfile(Path(__file__).with_name("candidate_review") / name, folder / name)
        plans = folder / "plans"
        plans.mkdir(exist_ok=True)
        (folder / "measurements").mkdir(exist_ok=True)
        rows = []
        for i in flags:
            c = report["candidates"][i - 1]
            assert c.get("orientation", 1) == 1
            rows.append(
                dict(
                    trace=i,
                    parameters=c["parameters"],
                    sectors={
                        str(r["sector"]): dict(flux=r["flux"], sigma=r["sigma"])
                        for r in c["sectors"]
                    },
                )
            )
        plan = dict(
            field=code,
            reference_sector=task["reference_sector"],
            emp=task["policy"]["empirical_star_sectors"],
            rows=rows,
            task=str(report_dir / "task.json"),
            source="coordinated task " + tid,
            python=python,
            hashes={
                str(s): report["inputs"][f"data/s{s:04d}/download.json"]["sha256"]
                for s in task["sectors"]
            },
        )
        write_json(plans / f"{code}.json", plan)
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
        )
        subprocess.run(
            [
                python,
                str(Path(__file__).resolve()),
                "--prepare",
                str(plans / f"{code}.json"),
                report["source_sha256"],
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        for name in ("gallery.py", "compare.py"):
            subprocess.run(
                [python, str(folder / name)],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
    with (folder / "comparison.csv").open() as stream:
        context = [
            r
            for r in csv.DictReader(stream)
            if r["field"] == code and int(r["trace"]) == trace
        ]
    if len(context) != len(report["candidates"][trace - 1]["sectors"]):
        raise ValueError("Missing sector comparisons")
    pars = report["candidates"][trace - 1]["parameters"]
    others = nearby_alternatives(report["candidates"], flags, trace)
    neighbors = [folder / "images" / f"{code}-{i}.png" for i in others]
    strip = []
    for path in (BASE / "measurements").glob(f"{code}-{trace}-*.json"):
        item = read_json(path)
        if item.get("state") == "measured" and item.get("parameters") == pars:
            strip.append(
                {
                    k: item.get(k)
                    for k in (
                        "sector",
                        "independent_sector",
                        "flux",
                        "translated_track_fluxes",
                    )
                }
            )
    repeat = assess_repeat_support(report["candidates"][trace - 1]["sectors"])
    checks = independent_checks(tid, code, trace, pars)
    overlap = []
    if len(flags) > 1:
        comparison = subprocess.check_output(
            [python, str(Path(__file__).with_name("candidate_path_overlap.py")),
             str(folder / "plans" / f"{code}.json"), *map(str, flags)],
            text=True, timeout=120,
        )
        overlap = [pair for pair in json.loads(comparison) if trace in pair["paths"]]
    result = inspect_image(
        image,
        dict(
            field=code,
            trace=trace,
            sectors=context,
            repeat_support=repeat,
            alternate_traces=others,
            detector_overlap=overlap,
            strip_photometry=strip,
            independent_sectors=checks,
            columns="r = centered shape correlation; lower_or_equal = nearby point sources scoring no higher",
        ),
        neighbors,
    )
    result["reviewed_paths"] = sorted(flags)
    result["detector_overlap"] = overlap
    result["neighbor_images"] = {img.name: sha256(img) for img in neighbors}
    result["repeat_support"] = repeat
    result["independent_sectors"] = checks
    write_json(
        receipt,
        dict(
            **result,
            task_id=tid,
            field=code,
            trace=trace,
            submission_digest=sub["submission_digest"],
        ),
    )


def prepare(path, expected_source):
    from p9search.provenance import source_hash

    if source_hash() != expected_source:
        raise ValueError("Installed source differs from the submitted search")
    plan = read_json(path)
    folder = path.parents[1]
    sys.path[:0] = [str(folder), str(BASE), str(Path(__file__).with_name("candidate_review"))]
    from fetch import fetch
    from admission import local_slot, resume_public
    from yield_for_pending_review import yield_after_receipt

    registry = read_json(
        ROOT / "outputs/planet_nine_travel_publication_20261120/config.json"
    )
    work = Path(registry["revisits"]).parent / "worker/work"
    data = fetch(plan)
    try:
        with (BASE / "figure-request.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            while True:
                slot, reason = local_slot(work)
                if slot:
                    break
                write_json(folder / "status.json", dict(state="waiting", reason=reason))
                if reason == "public worker has the search slot":
                    yield_after_receipt(work, folder / "status.json", timeout=60)
                else:
                    time.sleep(5)
            with slot:
                for sid in sorted({int(s) for r in plan["rows"] for s in r["sectors"]}):
                    write_json(
                        folder / "status.json", dict(state="measuring", sector=sid)
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            str(folder / "sector.py"),
                            str(path),
                            str(sid),
                            str(data),
                        ],
                        check=True,
                        pass_fds=(slot.fileno(),),
                    )
        shutil.rmtree(data.parent)
    finally:
        resume_public(work)


if __name__ == "__main__":
    if sys.argv[1] == "--prepare":
        prepare(Path(sys.argv[2]), sys.argv[3])
    else:
        parent, field, ids, trace, receipt = sys.argv[1:]
        review(Path(parent), field, json.loads(ids), int(trace), Path(receipt))
