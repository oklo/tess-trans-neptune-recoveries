"""Rank pending revisits by measured recovery versus expected elapsed time."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from p9search.storage import read_json, write_json
from scipy.optimize import least_squares, minimize
from scipy.special import expit


def fit_recovery(snapshot):
    trials = {}
    for field in snapshot["fields"]:
        if field["state"] != "complete":
            continue
        for row in field.get("injections", []):
            if (
                type(row.get("recovered")) is not bool
                or type(row.get("reference_index")) is not int
            ):
                continue
            index, magnitude = row["reference_index"], float(row["V"])
            if not np.isfinite(magnitude):
                raise ValueError("Nonfinite calibration magnitude")
            item = trials.setdefault(index, [magnitude, []])
            if abs(item[0] - magnitude) > 0.02:
                raise ValueError("Conflicting prior magnitudes")
            item[1].append(int(row["recovered"]))
    if len(trials) < 10:
        raise ValueError("Need at least ten distinct recovery trials")
    x = np.array([r[0] - 20 for r in trials.values()])
    y = np.array([np.mean(r[1]) for r in trials.values()])

    # Weak ridge regularization and monotonicity keep sparse tails finite.
    def loss(b):
        z = b[0] + b[1] * x
        return np.logaddexp(0, z).sum() - y @ z + 0.5 * (b @ b)

    result = minimize(loss, [0.0, -1.0], bounds=[(-20, 20), (-10, 0)])
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Recovery model did not converge")
    return {
        "coefficients": result.x.tolist(),
        "trials": len(trials),
        "successes": float(y.sum()),
    }


def fit_runtime(rows):
    valid = [r for r in rows if 0 < r["seconds"] < 7200 and r["injections"] > 0]
    if len(valid) < 10:
        raise ValueError("Need at least ten completed runtime measurements")
    x = np.array([[1, r["sectors"], r["injections"]] for r in valid], float)
    y = np.array([r["seconds"] for r in valid], float)
    result = least_squares(
        lambda b: x @ b - y,
        [60.0, 60.0, 80.0],
        bounds=(0.0, np.inf),
        loss="soft_l1",
        f_scale=60,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Runtime model did not converge")
    return {"coefficients": result.x.tolist(), "measurements": len(valid)}


def score(task, recovery, runtime):
    magnitudes = np.array(
        [20.94 - 2.5 * np.log10(c["flux"]) for c in task["injections"]]
    )
    a, b = recovery["coefficients"]
    probabilities = expit(a + b * (magnitudes - 20))
    coefficients = runtime["coefficients"]
    seconds = max(
        60.0, float(np.dot(coefficients, [1, len(task["sectors"]), len(magnitudes)]))
    )
    weight_per_hour = 0.2 / 21740 / (seconds / 3600)
    return {
        "expected_weight_per_hour": float(probabilities.sum() * weight_per_hour),
        # A small exploration component prevents sparse calibration tails
        # from assigning effectively zero value to unfamiliar conditions.
        "priority_score": float((0.95 * probabilities + 0.025).sum() * weight_per_hour),
        "estimated_seconds": seconds,
    }


def remote(ssh, code):
    return json.loads(
        subprocess.check_output(
            ssh + ["/usr/local/bin/php"],
            input="<?php chdir('/home3/okloorg/public_html/planet-nine'); require 'common.php';\n"
            + code,
            text=True,
            timeout=90,
        )
    )


def run(batch, snapshot):
    socket_dir = Path("/private/tmp") / f"p9pub-{os.getuid()}"
    socket_dir.mkdir(mode=0o700, exist_ok=True)
    if socket_dir.is_symlink() or socket_dir.stat().st_mode & 0o077:
        raise ValueError("SSH socket directory is not private")
    ssh = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=120",
        "-o",
        "ControlPath=" + str(socket_dir / "ssh.sock"),
        "oklo",
    ]
    samples = remote(
        ssh,
        """$db=database();$out=[];
foreach(query($db,"SELECT id,result FROM results WHERE status='accepted' AND verified=1") as $row){
$r=json_decode($row['result'],true);if(($r['state']??'')!=='searched'||($r['device']??'')!=='mps')continue;
$out[]=['result_id'=>(int)$row['id'],'seconds'=>$r['elapsed_seconds'],'sectors'=>count($r['quality']),'injections'=>count($r['injections'])];}
echo json_encode($out);""",
    )
    raw = snapshot.read_bytes()
    recovery, runtime = fit_recovery(json.loads(raw)), fit_runtime(samples)
    ranked = []
    for path in (batch / "revisits").glob("*.json"):
        if path.name == "status.json":
            continue
        record = read_json(path)
        task = json.loads(record["task"]["spec"])
        ranked.append(dict(task_id=task["task_id"], **score(task, recovery, runtime)))
    ranked.sort(key=lambda r: (-r["priority_score"], r["task_id"]))
    code = f"""$rows=json_decode({json.dumps(json.dumps(ranked))},true);$db=database();$changed=0;
transaction($db,function()use($db,$rows,&$changed){{$ids=array_column($rows,'task_id');$base=0;
foreach(query($db,"SELECT id,priority FROM tasks WHERE state='pending'") as $r){{if(!in_array($r['id'],$ids,true))$base=max($base,(float)$r['priority']);}}
foreach($rows as $i=>$r){{$q=query($db,"UPDATE tasks SET priority=? WHERE id=? AND state='pending'",[$base+2-$i/count($rows),$r['task_id']]);$changed+=$q->rowCount();}}
event($db,'recovery-ranking',null,null,'Magnitude recovery model and measured runtime; '.$changed.' pending revisits');}});echo json_encode(['updated'=>$changed]);"""
    updated = remote(ssh, code)
    write_json(
        batch / "ranking.json",
        dict(
            updated_epoch=time.time(),
            method="recovery-runtime-v1",
            snapshot_sha256=hashlib.sha256(raw).hexdigest(),
            latest_runtime_result=max(r["result_id"] for r in samples),
            recovery=recovery,
            runtime=runtime,
            **updated,
            ranked=ranked,
        ),
        private=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    try:
        run(args.batch, args.snapshot)
    except Exception as error:
        write_json(
            args.batch / "ranking-error.json",
            {"epoch": time.time(), "reason": str(error)[:500]},
            private=True,
        )
        raise
    (args.batch / "ranking-error.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
