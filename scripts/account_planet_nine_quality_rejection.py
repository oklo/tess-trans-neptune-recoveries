"""Retain a verified baseline rejection without claiming searched area or recovery."""
import argparse
import base64
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path

from kbo_tess.coordinated_yield_ledger import field_timing

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "distribution/planet-nine-search/coordinator/coordination"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_tree(directory):
    directory = Path(directory)
    for name, expected in read(directory / "result_freeze.json")["sha256"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError(f"Frozen evidence changed: {path}")


def rejection_ledger(previous, task, result, server):
    """A rejected baseline leaves every declared trial unassessed, never a miss."""
    if (previous["state"] != "complete" or previous["reference_models"] != 21740
            or previous["assumed_existence_probability"] != .2
            or previous["assumed_V_minus_T"] != .5):
        raise ValueError("Previous accounting assumptions differ")
    if (result["state"] != "quality_rejected" or result["injections"]
            or result.get("candidates") or result.get("candidate_count", 0)
            or result["exclusion_credit"] != 0):
        raise ValueError("Expected only a baseline quality rejection")
    quality = result["quality"]
    modern = result.get("quality_method") == "usable-pixels-v1"
    stopped = (quality and quality[-1]["masked_fraction"] == 1) if modern else (
        quality and .02 < quality[-1]["masked_fraction"] <= 1)
    passed = lambda r: (0 <= r["masked_fraction"] < 1) if modern else (
        0 <= r["masked_fraction"] <= .02)
    if (not quality or [r["sector"] for r in quality] != task["sectors"][:len(quality)]
            or not stopped or not all(passed(r) for r in quality[:-1])):
        raise ValueError("Quality evidence does not establish the released stop condition")
    ids = [int(case["label"].removeprefix("prior ")) for case in task["injections"]]
    blocked = set(previous["accounted_reference_ids"] + previous["retained_unassessed_prior_ids"])
    if len(ids) != len(set(ids)) or set(ids) & blocked:
        raise ValueError("Declared reference samples overlap previous accounting")
    utc = lambda seconds: datetime.fromtimestamp(seconds, timezone.utc).isoformat()
    start, finish, review = map(utc, (server["task"]["started"], server["result"]["submitted"], server["review_time"]))
    timing = field_timing(previous, start, finish, review)
    elapsed = timing["batch_elapsed_hours"]
    ledger = copy.deepcopy(previous)
    ledger.update(
        timing,
        science_state="quality_rejected", full_replay_verified=True,
        public_coordinator_submission_remains_unverified=False,
        source_task=task["task_id"], sectors=task["sectors"], quality=quality,
        previous_cumulative_proxy=previous["cumulative_expected_planets_proxy"],
        added_expected_planets_proxy=0., newly_accounted_reference_ids=[],
        retained_unassessed_prior_ids=sorted(set(previous["retained_unassessed_prior_ids"]) | set(ids)),
        quality_rejected_fields=previous.get("quality_rejected_fields", 0) + 1,
        new_batch_proxy_per_hour=previous["new_batch_expected_planets_proxy"] / elapsed,
        field_and_calibration_proxy_per_hour=0., field_verified_proxy_per_hour=0.,
        physical_followup_flags=0, actual_discoveries=0, replay_seconds=result["elapsed_seconds"],
        direct_calibration=dict(declared_models=len(ids), assessed_models=0, recovered_models=0,
            expected_planets_proxy=0., unassessed_ids=ids,
            unassessed_expectation_upper_bound=.2 * len(ids) / 21740),
        source_sha256=result["source_sha256"], utc=review,
        interpretation="Baseline quality rejection confirmed by independent replay. No moving-object search or injection recovery was performed. Declared trials remain unassessed; expected recovery is unchanged.",
    )
    return ledger


def account(cycle):
    from p9search.provenance import source_hash
    from p9search.storage import canonical

    destination = cycle / "ledger"
    if destination.exists():
        raise ValueError("Preserve the existing ledger; reconcile before retrying")
    plan, task = read(cycle / "credit-plan.json"), read(cycle / "task.json")
    accepted, submission = read(cycle / "accepted-review.json"), read(cycle / "submission.json")
    review_plan = read(cycle / "review-plan.json")
    master = Path(review_plan["directory"])
    receipt_path = master / "receipt.json"
    receipt = read(receipt_path)
    signed = base64.b64decode(receipt["payload"], validate=True)
    key_path = ROOT / "outputs/planet_nine_coordinator_deploy_2026_11_05/private/review.key"
    if key_path.stat().st_mode & 0o077:
        raise ValueError("Review key is not private")
    key = bytes.fromhex(key_path.read_text().strip())
    if not hmac.compare_digest(hmac.new(key, signed, hashlib.sha256).hexdigest(), receipt["signature"]):
        raise ValueError("Invalid signed review")
    review = json.loads(signed)
    run = master / "replay" / task["task_id"]
    result_path = run / "result.json"
    result, server = read(result_path), accepted["server"]
    assert sha(receipt_path) == accepted["receipt_sha256"]
    assert sha(result_path) == accepted["replay_result_sha256"] == review["replay_result_sha256"]
    assert review["contract"] == "trusted-full-replay-v1" and review["disposition"] == "accepted"
    assert review["replay_state"] == result["state"] == "quality_rejected"
    assert review["task_id"] == task["task_id"] == server["task"]["id"] == submission["task_id"] == result["task_id"]
    assert server["task"]["state"] == "reviewed" and server["result"]["verified"] == 1
    assert server["result"]["status"] == "accepted"
    assert review["worker_id"] == server["result"]["worker_id"] == submission["worker_id"]
    assert review["submission_digest"] == server["result"]["digest"] == submission["submission_digest"]
    assert review["spec_sha256"] == result["spec_sha256"] == hashlib.sha256(canonical(task).encode()).hexdigest()
    assert result["source_sha256"] == result["launch"]["source_sha256"] == review["execution"]["source_sha256"] == source_hash()
    assert review["policy"] == result["policy"] == task["policy"]
    assert review["trusted_inputs"] == result["inputs"] and review["sensitivity_credit"] == 0
    for name, digest in result["artifacts"].items():
        assert sha(run / name) == digest
    for name, record in result["inputs"].items():
        assert read(run / name) == record
        for member, digest in record["sha256"].items():
            assert sha((run / name).parent / member) == digest
    previous_dir = ROOT / plan["previous_ledger"]
    verify_tree(previous_dir)
    assert sha(previous_dir / "summary.json") == plan["previous_ledger_sha256"]
    geometry = ROOT / plan["geometry_directory"]
    verify_tree(geometry)
    assert sha(geometry / "direct/prior_truth.csv") == plan["truth_sha256"]
    ledger = rejection_ledger(read(previous_dir / "summary.json"), task, result, server)
    ledger["center"] = plan["center"]
    destination.mkdir()
    source_snapshot = destination / "accounting.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    evidence = [source_snapshot, cycle / "credit-plan.json", cycle / "task.json", cycle / "accepted-review.json",
                cycle / "review-plan.json", cycle / "submission.json", receipt_path, result_path,
                previous_dir / "summary.json", previous_dir / "result_freeze.json", geometry / "result_freeze.json"]
    ledger["sha256"] = {str(p): sha(p) for p in evidence}
    save(destination / "summary.json", ledger)
    save(destination / "outcomes.json", [])
    save(destination / "result_freeze.json", {"sha256": {p.name: sha(p) for p in destination.iterdir()}})


def export(cycle):
    verify_tree(cycle / "ledger")
    ledger = read(cycle / "ledger/summary.json")
    for name, digest in ledger["sha256"].items():
        assert sha(name) == digest
    assert ledger["science_state"] == "quality_rejected" and ledger["full_replay_verified"]
    assert ledger["added_expected_planets_proxy"] == 0 and ledger["newly_accounted_reference_ids"] == []
    plan = read(cycle / "credit-plan.json")
    before_bytes = (SITE / "snapshot.json").read_bytes()
    before = json.loads(before_bytes)
    assert before["sensitivity_proxy"]["value"] == ledger["cumulative_expected_planets_proxy"] == ledger["previous_cumulative_proxy"]
    after = copy.deepcopy(before)
    field = next(f for f in after["fields"] if f.get("task_id") == ledger["source_task"])
    assert field["code"] == plan["field"] and field["state"] == "queued"
    assert [field["ra_deg"], field["dec_deg"]] == ledger["center"]
    screened = {r["sector"] for r in ledger["quality"]}
    field.update(state="preflight_failed", sectors=[r["sector"] for r in ledger["quality"]],
                 footprints=[p for p in field["footprints"] if p["sector"] in screened],
                 candidate_count=None, physical_followup_flags=0, reviewed_utc=ledger["field_reviewed_utc"],
                 result="Stopped at the baseline image-quality check; independent replay confirmed the rejection.",
                 depth_label="Not searched", depth_note="No depth or sensitivity credit.", depth_samples=[], injections=[])
    after["field_counts"] = dict(Counter(f["state"] for f in after["fields"]))
    after["updated_utc"] = ledger["field_reviewed_utc"]
    assert after["searched_fields"] == before["searched_fields"]
    destination = cycle / "publication"
    destination.mkdir()
    (destination / "previous_snapshot.json").write_bytes(before_bytes)
    save(destination / "snapshot.json", after)
    save(destination / "summary.json", dict(field=plan["field"], field_state="preflight_failed",
        previous_sha256=sha(destination / "previous_snapshot.json"), snapshot_sha256=sha(destination / "snapshot.json"),
        previous_expectation=before["sensitivity_proxy"]["value"], expectation=after["sensitivity_proxy"]["value"],
        searched_fields=after["searched_fields"], flags=after["physical_followup_flags"]))
    (SITE / "snapshot.json").write_bytes((destination / "snapshot.json").read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("account", "export"))
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--cycle", required=True, type=int)
    args = parser.parse_args()
    batch = args.batch.resolve()
    if batch.parent != ROOT / "outputs" or args.cycle < 1:
        parser.error("Expected an internal output batch and a positive cycle")
    cycle = batch / f"cycle-{args.cycle:02d}"
    (account if args.stage == "account" else export)(cycle)
    print(json.dumps({"cycle": args.cycle, "stage": args.stage, "added_credit": 0}))


if __name__ == "__main__":
    main()
