"""Review, account, and publish one coordinated field under the audit policy."""

import argparse
import copy
import csv
import hashlib
import json
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from p9search.provenance import source_hash
from p9search.review import review, verify_signed
from p9search.storage import canonical
from p9search.storage import read_json as read
from p9search.storage import sha256 as sha
from p9search.storage import write_json as save
from p9search.validation import check_report, require

ROOT = Path(__file__).resolve().parents[1]
SSH = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=/private/tmp/p9-service-%C",
    "oklo",
]
REMOTE = "/home3/okloorg/.planet-nine"
ADMIN = [
    "env",
    f"P9_REVIEW_KEY={REMOTE}/review.key",
    "/usr/local/bin/php",
    "/home3/okloorg/public_html/planet-nine/admin.php",
]
KEY = ROOT / "outputs/planet_nine_coordinator_deploy_2026_11_05/private/review.key"
LABELS = {
    "consistency": "Consistency checked",
    "full_replay": "Independent full replay",
    "trusted_local": "Trusted local run; consistency checked",
    "statistical": "Statistical acceptance; 10% replay audit policy",
}


def admin(*args):
    return subprocess.check_output(
        SSH + [shlex.join([*ADMIN, *map(str, args)])], text=True, timeout=90
    )


def config_for(cycle):
    config = read(cycle / "operator-config.json")
    require(
        cycle.parent == Path(config["batch"])
        and cycle.parent.parent == ROOT / "outputs",
        "Cycle outside configured batch",
    )
    require(source_hash() == config["source_sha256"], "Installed release differs")
    return config


def storage(config):
    from kbo_tess.scandir_storage import inspect_storage

    require(
        sha(config["storage_config"]) == config["storage_config_sha256"],
        "Storage configuration changed",
    )
    policy = read(config["storage_config"])
    inspect_storage(policy)
    return Path(policy["root_path"])


def review_reserve_gb(config, admission):
    require(
        admission["storage_config"] == config["storage_config"]
        and admission["storage_config_sha256"] == config["storage_config_sha256"]
        and sha(config["storage_config"]) == config["storage_config_sha256"],
        "Review admission storage policy differs",
    )
    reserve = read(config["storage_config"])["minimum_external_free_bytes"]
    require(type(reserve) is int and reserve >= 20 * 2**30,
            "Review reserve is below the operator policy")
    return reserve / 2**30


def worker_run(cycle, config):
    return (
        storage(config)
        / "workers"
        / config.get("namespace", "luna-repeat-20261105")
        / cycle.name
        / read(cycle / "task.json")["task_id"]
    )


def inspect_server(task_id):
    require(
        len(task_id) == 64 and all(c in "0123456789abcdef" for c in task_id),
        "Invalid task ID",
    )
    php = f"""<?php
    $db=new PDO('sqlite:/home3/okloorg/.planet-nine/queue.sqlite');
    $id={json.dumps(task_id)};
    function one($db,$sql,$id) {{$s=$db->prepare($sql);$s->execute([$id]);return $s->fetch(PDO::FETCH_ASSOC);}}
    $task=one($db,'SELECT id,state,started,attempts,failures FROM tasks WHERE id=?',$id);
    $result=one($db,"SELECT worker_id,status,verified,digest,submitted FROM results WHERE task_id=? AND status='accepted' ORDER BY id DESC LIMIT 1",$id);
    $time=one($db,"SELECT time FROM events WHERE task_id=? AND action='review' ORDER BY id DESC LIMIT 1",$id);
    $holds=$db->query("SELECT COUNT(*) FROM tasks WHERE state='audit_hold'")->fetchColumn();
    echo json_encode(['task'=>$task,'result'=>$result,'review_time'=>$time['time']??null,'audit_holds'=>(int)$holds]);
    """
    return json.loads(
        subprocess.check_output(
            SSH + ["/usr/local/bin/php"], input=php, text=True, timeout=60
        )
    )


def reviewed(cycle):
    config = config_for(cycle)
    plan = read(cycle / "review-plan.json")
    master = Path(plan["directory"])
    receipt = master / "receipt.json"
    require(not KEY.stat().st_mode & 0o077, "Review key is not private")
    payload = verify_signed(read(receipt), bytes.fromhex(KEY.read_text().strip()))
    task, submission = read(cycle / "task.json"), read(cycle / "submission.json")
    result_path = master / "reviewed-result.json"
    result = read(result_path)
    require(
        payload["contract"] == "statistical-review-v1"
        and payload["disposition"] == "accepted"
        and payload["review_method"] in LABELS
        and payload["task_id"] == task["task_id"] == submission["task_id"]
        and payload["worker_id"] == submission["worker_id"] == config["worker_id"]
        and payload["submission_digest"]
        == submission["submission_digest"]
        == hashlib.sha256(submission["result_json"].encode()).hexdigest()
        and payload["spec_sha256"]
        == hashlib.sha256(canonical(task).encode()).hexdigest()
        and payload["result_sha256"] == sha(result_path)
        and payload["reported_inputs"] == result["inputs"]
        and payload["sensitivity_credit"] == 0,
        "Review evidence differs",
    )
    check_report(task, result, config["approved_source_sha256"])
    return config, task, result, payload, receipt, result_path


def review_cycle(cycle):
    config = config_for(cycle)
    task = read(cycle / "task.json")
    distributed = config.get("worker_mode") == "distributed"
    if not distributed:
        guard = read(cycle / "service-worker-status.json")
        require(
            guard["returncode"] == 0 and not guard["active_pids"],
            "Local worker did not finish",
        )
    submissions = [
        json.loads(line) for line in admin("results").splitlines() if line.strip()
    ]
    selected = [r for r in submissions if r["task_id"] == task["task_id"]]
    require(
        len(selected) == 1 and selected[0]["worker_id"] == config["worker_id"],
        "Expected current coordinator submission",
    )
    save(cycle / "submission.json", selected[0], private=True)
    assignment = json.loads(admin("review-plan", task["task_id"]))
    save(cycle / "audit-plan.json", assignment, private=True)
    decision = verify_signed(assignment, bytes.fromhex(KEY.read_text().strip()))
    require(
        not distributed or decision["mode"] != "trusted_local",
        "Distributed workers must use the volunteer audit policy",
    )
    master = (
        storage(config)
        / "reviews"
        / config.get("namespace", "luna-repeat-20261105")
        / (cycle.name + "-audited")
    )
    require(not master.exists(), "Preserve previous review attempt")
    save(
        cycle / "review-plan.json",
        {
            "directory": str(master),
            "task_id": task["task_id"],
            "source_sha256": config["source_sha256"],
            "assigned_method": decision["mode"],
        },
    )
    pair = read(cycle / "pair-admission.json")
    args = argparse.Namespace(
        task=cycle / "task.json",
        submission=cycle / "submission.json",
        key=KEY,
        plan=cycle / "audit-plan.json",
        local_run=None if distributed else worker_run(cycle, config),
        directory=master,
        approved_source=config["approved_source_sha256"],
        max_disk_gb=pair["replay_budget_bytes"] / 2**30,
        reserve_gb=review_reserve_gb(config, pair),
        max_memory_gb=config["maximum_memory_gib"],
        max_hours=1,
        device="mps",
        threads=4,
        bundle_origin=["https://oklo.org"],
    )
    started = time.monotonic()
    try:
        review(args)
    except BaseException as error:
        save(
            cycle / "review-status.json",
            {
                "state": "failed",
                "returncode": 1,
                "elapsed_seconds": time.monotonic() - started,
                "error": str(error),
            },
        )
        raise
    payload = verify_signed(
        read(master / "receipt.json"), bytes.fromhex(KEY.read_text().strip())
    )
    save(
        cycle / "review-status.json",
        {
            "state": "complete",
            "returncode": 0,
            "elapsed_seconds": time.monotonic() - started,
            "review_method": payload["review_method"],
        },
    )


def accept_cycle(cycle):
    config = config_for(cycle)
    task, submission = read(cycle / "task.json"), read(cycle / "submission.json")
    master = Path(read(cycle / "review-plan.json")["directory"])
    path = master / "receipt.json"
    payload = verify_signed(read(path), bytes.fromhex(KEY.read_text().strip()))
    require(
        payload["task_id"] == task["task_id"]
        and payload["worker_id"] == config["worker_id"]
        and payload["submission_digest"] == submission["submission_digest"],
        "Receipt identity differs",
    )
    if payload["disposition"] == "accepted":
        reviewed(cycle)
    # On a lost response, inspect accepted state before considering any repeat mutation.
    before = inspect_server(task["task_id"])
    if before["task"]["state"] != "reviewed":
        # Disagreements must still reach the admin to suspend related reports.
        remote = f"{REMOTE}/receipts/{sha(path)}.json"
        subprocess.run(
            SSH
            + [
                "mkdir -p "
                + shlex.quote(REMOTE + "/receipts")
                + " && chmod 700 "
                + shlex.quote(REMOTE + "/receipts")
            ],
            check=True,
        )
        subprocess.run(
            ["scp", *SSH[1:-1], str(path), "oklo:" + remote], check=True, timeout=60
        )
        subprocess.run(SSH + ["chmod 600 " + shlex.quote(remote)], check=True)
        admin("accept", remote)
    require(
        payload["disposition"] == "accepted",
        "Audit disagreed; worker and related reports held",
    )
    server = inspect_server(task["task_id"])
    require(
        server["task"]["state"] == "reviewed"
        and server["result"]["verified"] == 1
        and server["result"]["digest"] == submission["submission_digest"],
        "Acceptance differs",
    )
    save(
        cycle / "accepted-review.json",
        {
            "server": server,
            "receipt_sha256": sha(path),
            "result_sha256": payload["result_sha256"],
            "review_method": payload["review_method"],
            "sensitivity_credit": 0,
        },
    )


def followup_ledger(previous, task, result, review, server, models):
    """Retain an audited candidate field without crediting its untested injections."""
    from kbo_tess.coordinated_yield_ledger import field_timing
    from kbo_tess.direct_prior_calibration import direct_yield

    count = result.get("candidate_count", 0)
    require(
        result["state"] == "needs_followup"
        and type(count) is int
        and count > 0
        and count == len(result["candidates"])
        and result["injections"] == []
        and result["exclusion_credit"] == 0
        and review["review_method"] in LABELS,
        "Candidate retention requires an accepted search without sensitivity credit",
    )
    require(
        previous["state"] == "complete"
        and previous["reference_models"] == 21740
        and previous["assumed_existence_probability"] == 0.2
        and previous["assumed_V_minus_T"] == 0.5,
        "Previous accounting assumptions differ",
    )
    ids = [m["reference_index"] for m in models]
    blocked = (
        previous["accounted_reference_ids"] + previous["retained_unassessed_prior_ids"]
    )
    require(
        len(ids) == len(set(ids)) and not set(ids) & set(blocked),
        "Overlapping prior trials",
    )
    outcomes = [
        dict(
            m,
            assessed=False,
            recovered=None,
            control_maximum=result["control_maximum"],
            reason="Calibration withheld pending candidate follow-up",
        )
        for m in models
    ]
    calibration = direct_yield(models, outcomes, result["control_maximum"])
    timestamps = [
        datetime.fromtimestamp(t, UTC).isoformat()
        for t in (
            server["task"]["started"],
            server["result"]["submitted"],
            server["review_time"],
        )
    ]
    timing = field_timing(previous, *timestamps)
    elapsed = timing["batch_elapsed_hours"]
    ledger = copy.deepcopy(previous)
    ledger["retained_followup_searches"].append(
        dict(
            source_search="coordinated task " + task["task_id"],
            candidate_flags=list(range(1, count + 1)),
            unassessed_prior_ids=ids,
        )
    )
    ledger.update(
        timing,
        previous_cumulative_proxy=previous["cumulative_expected_planets_proxy"],
        added_expected_planets_proxy=0.0,
        newly_accounted_reference_ids=[],
        retained_unassessed_prior_ids=sorted(
            set(previous["retained_unassessed_prior_ids"]) | set(ids)
        ),
        unresolved_flags_in_other_fields=previous["unresolved_flags_in_other_fields"]
        + count,
        physical_followup_flags=count,
        actual_discoveries=0,
        direct_calibration=calibration,
        new_batch_proxy_per_hour=previous["new_batch_expected_planets_proxy"] / elapsed,
        field_and_calibration_proxy_per_hour=0.0,
        field_verified_proxy_per_hour=0.0,
        interpretation="Candidate evidence and untested priors retained for follow-up. No sensitivity credit.",
    )
    return ledger, outcomes


def accounting(cycle):
    from account_planet_nine_quality_rejection import rejection_ledger, verify_tree

    from kbo_tess.coordinated_yield_ledger import append_reviewed_field

    _config, task, result, review_record, receipt, result_path = reviewed(cycle)
    accepted = read(cycle / "accepted-review.json")
    require(accepted["receipt_sha256"] == sha(receipt), "Acceptance receipt changed")
    server = inspect_server(task["task_id"])
    require(
        server == accepted["server"] and server["audit_holds"] == 0,
        "Accepted state changed or related audit needs reassessment",
    )
    require(
        result["state"] == "needs_followup" or not result.get("candidate_count", 0),
        "Unexpected candidates outside a follow-up result",
    )
    destination = cycle / "ledger"
    require(not destination.exists(), "Preserve existing sensitivity ledger")
    plan = read(cycle / "credit-plan.json")
    previous_path = ROOT / plan["previous_ledger"]
    verify_tree(previous_path)
    require(
        sha(previous_path / "summary.json") == plan["previous_ledger_sha256"],
        "Predecessor changed",
    )
    previous = read(previous_path / "summary.json")
    geometry = ROOT / plan["geometry_directory"]
    verify_tree(geometry)
    truth_path = geometry / "direct/prior_truth.csv"
    require(sha(truth_path) == plan["truth_sha256"], "Reference injections changed")
    population_path = (
        ROOT / "outputs/planet_nine_reference_2026_11_02/reference_population.csv"
    )
    require(
        sha(population_path) == plan["population_sha256"],
        "Reference population changed",
    )
    with population_path.open() as stream:
        population = {
            int(r["index"]): r
            for r in csv.DictReader(
                (line for line in stream if not line.startswith("#")),
                skipinitialspace=True,
            )
        }
    with truth_path.open() as stream:
        truths = list(csv.DictReader(stream))
    require(len(truths) == len(task["injections"]), "Declared injection count differs")
    blocked = set(
        previous["accounted_reference_ids"] + previous["retained_unassessed_prior_ids"]
    )
    models, outcomes = [], []
    for truth, case in zip(truths, task["injections"], strict=True):
        model = {
            "reference_index": int(truth["reference_index"]),
            "reference_V": float(truth["reference_V"]),
            "assumed_V_minus_T": 0.5,
            "flux": float(truth["flux"]),
        }
        catalog = population[model["reference_index"]]
        require(
            model["reference_index"] not in blocked
            and float(catalog["V"]) == model["reference_V"]
            and all(int(catalog[k]) == 0 for k in ("ZTF", "DES", "PS1"))
            and case["label"] == f"prior {model['reference_index']}"
            and case["flux"] == model["flux"]
            and case["parameters"]
            == [float(truth[k]) for k in ("x", "y", "qx", "qy", "inverse_distance")]
            and case["radial_velocity"] == float(truth["radial_velocity"]),
            "Prior trial differs",
        )
        models.append(model)
    if result["state"] == "needs_followup":
        ledger, outcomes = followup_ledger(
            previous, task, result, review_record, server, models
        )
    elif result["state"] == "quality_rejected":
        ledger = rejection_ledger(previous, task, result, server)
        ledger["interpretation"] = (
            "Baseline quality rejection. Declared trials remain unassessed; no sensitivity credit."
        )
    else:
        require(
            result["state"] == "searched"
            and all(r["assessed"] for r in result["injections"]),
            "Unassessed injections need scientific review",
        )
        for model, row in zip(models, result["injections"], strict=True):
            outcomes.append(
                dict(
                    model,
                    assessed=True,
                    recovered=row["recovered"],
                    control_maximum=result["control_maximum"],
                    strongest_nearby_score=row["best_score"],
                    all_nearby_alternatives_scored=True,
                    native_responses_passed=True,
                    baseline_above_control=False,
                )
            )
        utc = lambda t: datetime.fromtimestamp(t, UTC).isoformat()
        ledger = append_reviewed_field(
            previous,
            previous["accounted_reference_ids"],
            models,
            outcomes,
            result["control_maximum"],
            field_start_utc=utc(server["task"]["started"]),
            field_end_utc=utc(server["result"]["submitted"]),
            review_end_utc=utc(server["review_time"]),
            physical_flags=0,
        )
        for key in (
            "retained_followup_searches",
            "retained_unassessed_prior_ids",
            "unresolved_flags_in_other_fields",
        ):
            ledger[key] = previous[key]
        ledger["quality_rejected_fields"] = previous.get("quality_rejected_fields", 0)
    ledger.update(
        source_task=task["task_id"],
        center=plan["center"],
        sectors=task["sectors"],
        science_state=result["state"],
        quality=result["quality"],
        source_sha256=result["source_sha256"],
        full_replay_verified=review_record["full_replay_verified"],
        review_method=review_record["review_method"],
        audit_probability=review_record["audit_probability"],
        replay_seconds=review_record["replay_seconds"],
        public_coordinator_submission_remains_unverified=False,
    )
    breakdown = dict(
        previous.get(
            "expectation_by_validation",
            {"historical": previous["cumulative_expected_planets_proxy"]},
        )
    )
    method = review_record["review_method"]
    breakdown[method] = (
        breakdown.get(method, 0.0) + ledger["added_expected_planets_proxy"]
    )
    ledger["expectation_by_validation"] = breakdown
    destination.mkdir()
    # Freeze the small accounting and validation implementations with their outputs.
    import account_planet_nine_quality_rejection
    import p9search.validation

    import kbo_tess.coordinated_yield_ledger

    evidence = []
    for name, source in [
        ("accounting.py", __file__),
        ("validation.py", p9search.validation.__file__),
        ("yield_ledger.py", kbo_tess.coordinated_yield_ledger.__file__),
        ("quality_rejection.py", account_planet_nine_quality_rejection.__file__),
    ]:
        path = destination / name
        path.write_bytes(Path(source).read_bytes())
        evidence.append(path)
    evidence += [
        cycle / name
        for name in (
            "credit-plan.json",
            "task.json",
            "accepted-review.json",
            "review-plan.json",
            "submission.json",
            "audit-plan.json",
        )
    ]
    evidence += [
        receipt,
        result_path,
        previous_path / "summary.json",
        previous_path / "result_freeze.json",
        truth_path,
        population_path,
        geometry / "result_freeze.json",
    ]
    ledger["sha256"] = {str(p): sha(p) for p in evidence}
    save(destination / "summary.json", ledger)
    save(destination / "outcomes.json", outcomes)
    save(
        destination / "result_freeze.json",
        {"sha256": {p.name: sha(p) for p in destination.iterdir()}},
    )
    print(
        json.dumps(
            {
                k: ledger[k]
                for k in (
                    "added_expected_planets_proxy",
                    "cumulative_expected_planets_proxy",
                    "review_method",
                )
            }
        )
    )


def export(cycle, *, site=None):
    from collections import Counter

    from account_planet_nine_quality_rejection import verify_tree

    verify_tree(cycle / "ledger")
    ledger = read(cycle / "ledger/summary.json")
    for path, expected in ledger["sha256"].items():
        require(sha(path) == expected, "Accounting evidence changed")
    site = Path(site) if site is not None else (
        ROOT / "distribution/planet-nine-search/coordinator/coordination/snapshot.json"
    )
    original = site.read_bytes()
    snapshot = json.loads(original)
    before = snapshot["sensitivity_proxy"]["value"]
    require(
        before == ledger["previous_cumulative_proxy"], "Published predecessor differs"
    )
    field = next(
        f for f in snapshot["fields"] if f.get("task_id") == ledger["source_task"]
    )
    plan = read(cycle / "credit-plan.json")
    require(
        field["state"] == plan.get("previous_state", "queued")
        and field["state"] in ("queued", "preflight_failed")
        and field["code"] == plan["field"]
        and [field["ra_deg"], field["dec_deg"]] == ledger["center"]
        and field["planned_sectors"] == ledger["sectors"],
        "Published field differs",
    )
    if ledger.get("revisit"):
        require(field["state"] == "preflight_failed", "Only unsearched fields use this revisit path")
        field.setdefault("search_history", []).append({
            "state": field["state"], "result": field["result"],
            "reviewed_utc": field.get("reviewed_utc"),
            "result_id": ledger["revisit"]["previous_result_id"],
        })
        field["footprints"] = ledger["revisit"]["footprints"]
    outcomes = read(cycle / "ledger/outcomes.json")
    method = LABELS[ledger["review_method"]]
    field.update(
        review_method=ledger["review_method"],
        audit_probability=ledger["audit_probability"],
        reviewed_utc=ledger["field_reviewed_utc"],
        physical_followup_flags=0,
        quality=ledger["quality"],
    )
    diagnostics = [r["difficulty"] for r in ledger["quality"] if "difficulty" in r]
    if diagnostics:
        field["difficulty"] = {
            "method": "mean-sector-usable-area-exposure-noise-v1",
            "score": sum(r["score"] for r in diagnostics) / len(diagnostics),
            "sectors_measured": len(diagnostics),
        }
    if ledger["science_state"] == "needs_followup":
        count = ledger["physical_followup_flags"]
        require(
            count > 0
            and ledger["added_expected_planets_proxy"] == 0
            and ledger["newly_accounted_reference_ids"] == [],
            "Follow-up field must remain uncredited",
        )
        field.update(
            state="needs_followup",
            sectors=ledger["sectors"],
            candidate_count=count,
            physical_followup_flags=count,
            result=f"{count} candidate traces awaiting follow-up. {method}.",
            depth_label="Depth uncalibrated",
            depth_note="Recovery calibration withheld pending candidate follow-up.",
            depth_samples=[],
            injections=[],
        )
        snapshot["searched_fields"] += 1
        snapshot["physical_followup_flags"] += count
        require(
            snapshot["physical_followup_flags"]
            == ledger["unresolved_flags_in_other_fields"],
            "Candidate totals differ from retained evidence",
        )
    elif ledger["science_state"] == "quality_rejected":
        screened = {r["sector"] for r in ledger["quality"]}
        field.update(
            state="preflight_failed",
            sectors=[r["sector"] for r in ledger["quality"]],
            footprints=[p for p in field["footprints"] if p["sector"] in screened],
            candidate_count=None,
            result=f"Stopped at the baseline image-quality check. {method}.",
            depth_label="Not searched",
            depth_note="No depth or sensitivity credit.",
            depth_samples=[],
            injections=[],
        )
    else:
        require(
            ledger["science_state"] == "searched"
            and ledger["physical_followup_flags"] == 0,
            "Only cleared searches enter sensitivity accounting",
        )
        good = [r["reference_V"] for r in outcomes if r["recovered"]]
        groups = {}
        for row in outcomes:
            v = round(row["reference_V"], 2)
            group = groups.setdefault(
                v, {"band": "V", "magnitude": v, "recovered": 0, "tested": 0}
            )
            group["tested"] += int(row["assessed"])
            group["recovered"] += int(row["recovered"] is True)
        field.update(
            state="complete",
            sectors=ledger["sectors"],
            candidate_count=0,
            result=f"No surviving candidates. Recovered {len(good)} of {len(outcomes)} native injections. {method}.",
            depth_label=f"V {max(good):.2f} recovered"
            if good
            else "No injections recovered",
            depth_note="Faintest recovered injection; recovery depends on orbit and position."
            if good
            else "No limiting magnitude established.",
            depth_samples=[groups[v] for v in sorted(groups)],
            injections=[
                {
                    "reference_index": r["reference_index"],
                    "V": r["reference_V"],
                    "recovered": r["recovered"],
                    "assessed": r["assessed"],
                }
                for r in outcomes
            ],
        )
        snapshot["searched_fields"] += 1
    snapshot["sensitivity_proxy"].update(
        value=ledger["cumulative_expected_planets_proxy"],
        validation="Consistency-checked results; candidate traces retained for scientific follow-up.",
        by_validation=ledger["expectation_by_validation"],
    )
    snapshot["field_counts"] = dict(Counter(f["state"] for f in snapshot["fields"]))
    snapshot["updated_utc"] = ledger["field_reviewed_utc"]
    destination = cycle / "publication"
    destination.mkdir(exist_ok=True)
    (destination / "previous_snapshot.json").write_bytes(original)
    save(destination / "snapshot.json", snapshot)
    save(
        destination / "summary.json",
        {
            "field": plan["field"],
            "field_state": field["state"],
            "candidate_count": field["candidate_count"],
            "previous_sha256": sha(destination / "previous_snapshot.json"),
            "snapshot_sha256": sha(destination / "snapshot.json"),
            "previous_expectation": before,
            "expectation": ledger["cumulative_expected_planets_proxy"],
            "searched_fields": snapshot["searched_fields"],
            "flags": snapshot["physical_followup_flags"],
        },
    )
    site.write_bytes((destination / "snapshot.json").read_bytes())


def release_cycle(config, number):
    from account_planet_nine_quality_rejection import verify_tree
    from p9search.retention import release_pixels
    from retain_planet_nine_completed_arrays import idle

    cycle = Path(config["batch"]) / f"cycle-{number:02d}"
    if config.get("compact_helper"):
        from compact_planet_nine_cycles import compact_cycle

        require(sha(config["compact_helper"]) == config["tool_sha256"][config["compact_helper"]],
                "Compact retention helper changed")
        retained = compact_cycle(cycle, config)
        summary = {
            "state": retained["state"], "cycle": number, "sensitivity_credit": 0,
            "bytes_released": retained["pixel_bytes_released"] + retained["metadata_bytes_released"],
            "archive": retained["archive"], "archive_added_bytes": retained["archive_added_bytes"],
        }
        save(cycle / "pixel-release/complete.json", summary)
        return summary
    _, task, result, receipt, _, _ = reviewed(cycle)
    require(
        inspect_server(task["task_id"])["task"]["state"] == "reviewed",
        "Acceptance changed",
    )
    verify_tree(cycle / "ledger")
    ledger = read(cycle / "ledger/summary.json")
    for path, expected in ledger["sha256"].items():
        require(sha(path) == expected, "Accounting evidence changed")
    deployed = read(cycle / "publication/deployed.json")
    require(
        deployed.get("https_verified") is True
        and deployed["snapshot_sha256"] == sha(cycle / "publication/snapshot.json")
        and deployed["expectation"] == ledger["cumulative_expected_planets_proxy"],
        "Publication not confirmed",
    )
    if result.get("candidate_count", 0):
        require(
            ledger["science_state"] == result["state"] == "needs_followup"
            and ledger["physical_followup_flags"] == result["candidate_count"]
            and ledger["added_expected_planets_proxy"] == 0
            and receipt["full_replay_verified"] is True,
            "Candidate retention evidence differs",
        )
        summary = dict(
            state="complete",
            cycle=number,
            task_id=task["task_id"],
            disposition="retained_for_followup",
            bytes_released=0,
            sensitivity_credit=0,
            candidate_count=result["candidate_count"],
        )
        save(cycle / "pixel-release/complete.json", summary)
        return summary
    runs = [worker_run(cycle, config)]
    if receipt["review_method"] == "full_replay":
        runs.append(
            Path(read(cycle / "review-plan.json")["directory"])
            / "replay"
            / task["task_id"]
        )
    released = []
    for run in runs:
        storage(config)
        idle(list(run.glob("data/s*/*.npy")) + list(run.glob("data/s*/strips/*.npy")))
        _, _, record = release_pixels(run, task)
        released.append(
            {
                "directory": str(run),
                "record_sha256": sha(run / "retention.json"),
                "bytes_released": record["bytes_released"],
            }
        )
    summary = {
        "state": "complete",
        "cycle": number,
        "task_id": task["task_id"],
        "runs": released,
        "bytes_released": sum(r["bytes_released"] for r in released),
        "sensitivity_credit": 0,
    }
    save(cycle / "pixel-release/complete.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("review", "accept", "credit", "export"))
    parser.add_argument("--cycle", required=True, type=Path)
    args = parser.parse_args()
    {
        "review": review_cycle,
        "accept": accept_cycle,
        "credit": accounting,
        "export": export,
    }[args.action](args.cycle.resolve())


if __name__ == "__main__":
    main()
