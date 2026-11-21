"""Publish accepted travel reviews through the existing scientific ledger.

Run once per minute with the review environment. Configuration and checkpoints
live in an operator-owned directory; no participant credentials are used.
"""

import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import planet_nine_adaptive_cycle as science
import planet_nine_publication as publication
from account_planet_nine_quality_rejection import rejection_ledger, verify_tree
from kbo_tess.coordinated_yield_ledger import append_reviewed_field
from p9search.provenance import source_hash
from p9search.storage import read_json as read, sha256 as sha, write_json as save
from p9search.validation import require


def reference_models(task, population, previous):
    blocked = set(previous["accounted_reference_ids"] + previous["retained_unassessed_prior_ids"])
    models = []
    for case in task["injections"]:
        require(re.fullmatch(r"prior [0-9]+", case["label"]), "Unknown reference label")
        index = int(case["label"].split()[1])
        row = population[index]
        magnitude = float(row["V"])
        require(index not in blocked and all(int(row[k]) == 0 for k in ("ZTF", "DES", "PS1")),
                "Overlapping or screened reference sample")
        require(case["flux"] == 10 ** ((20.44 - (magnitude - .5)) / 2.5),
                "Declared injection differs from frozen catalog brightness")
        blocked.add(index)
        models.append(dict(reference_index=index, reference_V=magnitude,
                           assumed_V_minus_T=.5, flux=case["flux"]))
    return models


def make_ledger(previous, task, result, review, server, models):
    outcomes = []
    if result["state"] == "needs_followup":
        ledger, outcomes = science.followup_ledger(previous, task, result, review, server, models)
    elif result["state"] == "quality_rejected":
        ledger = rejection_ledger(previous, task, result, server)
    else:
        require(result["state"] == "searched" and not result.get("candidate_count", 0)
                and all(r["assessed"] for r in result["injections"]),
                "Incomplete calibration requires scientific review")
        for model, row in zip(models, result["injections"], strict=True):
            require(row["label"] == f"prior {model['reference_index']}", "Injection order changed")
            outcomes.append(dict(model, assessed=True, recovered=row["recovered"],
                                 control_maximum=result["control_maximum"],
                                 strongest_nearby_score=row["best_score"],
                                 all_nearby_alternatives_scored=True,
                                 native_responses_passed=True, baseline_above_control=False))
        utc = lambda t: datetime.fromtimestamp(t, UTC).isoformat()
        ledger = append_reviewed_field(previous, previous["accounted_reference_ids"],
            models, outcomes, result["control_maximum"],
            field_start_utc=utc(server["task"]["started"]),
            field_end_utc=utc(server["result"]["submitted"]),
            review_end_utc=utc(server["review_time"]), physical_flags=0)
        for key in ("retained_followup_searches", "retained_unassessed_prior_ids",
                    "unresolved_flags_in_other_fields"):
            ledger[key] = copy.deepcopy(previous[key])
        ledger["quality_rejected_fields"] = previous.get("quality_rejected_fields", 0)
    breakdown = dict(previous["expectation_by_validation"])
    method = review["review_method"]
    breakdown[method] = breakdown.get(method, 0.) + ledger["added_expected_planets_proxy"]
    ledger.update(source_task=task["task_id"], sectors=task["sectors"], science_state=result["state"],
        quality=result["quality"], source_sha256=result["source_sha256"],
        full_replay_verified=review["full_replay_verified"], review_method=method,
        audit_probability=review["audit_probability"], replay_seconds=review["replay_seconds"],
        public_coordinator_submission_remains_unverified=False, expectation_by_validation=breakdown)
    return ledger, outcomes


def stored_acceptance(number, receipt, submission):
    code = """<?php
    $db=new PDO('sqlite:/home3/okloorg/.planet-nine/queue.sqlite');
    $s=$db->prepare('SELECT task_id,worker_id,status,verified,digest,receipt FROM results WHERE id=?');
    $s->execute([%d]); echo json_encode($s->fetch(PDO::FETCH_ASSOC));
    """ % number
    row = json.loads(subprocess.check_output(science.SSH + ["/usr/local/bin/php"],
                                             input=code, text=True, timeout=60))
    require(row and row["status"] == "accepted" and row["verified"] == 1
            and row["task_id"] == submission["task_id"]
            and row["worker_id"] == submission["worker_id"]
            and row["digest"] == submission["submission_digest"]
            and json.loads(row["receipt"]) == receipt, "Coordinator acceptance differs")


def prepare(cycle, config, head, evidence):
    task, submission = read(evidence / "task.json"), read(evidence / "submission.json")
    require(submission["result_id"] == int(cycle.name.split("-")[1]), "Result identity differs")
    cycle.mkdir(exist_ok=True)
    plan_path = cycle / "credit-plan.json"
    if not plan_path.exists():
        snapshot = read(Path(config["checkout"]) / publication.SNAPSHOT)
        matches = [f for f in snapshot["fields"] if f.get("task_id") == task["task_id"]]
        require(len(matches) == 1 and matches[0]["state"] == "queued", "Field is not pending in atlas")
        field = matches[0]
        require(field["planned_sectors"] == task["sectors"], "Planned sectors differ")
        save(plan_path, dict(previous_ledger=head["ledger"], previous_ledger_sha256=head["ledger_sha256"],
                            field=field["code"], center=[field["ra_deg"], field["dec_deg"]]))
    plan = read(plan_path)
    require(plan["previous_ledger"] == head["ledger"]
            and plan["previous_ledger_sha256"] == head["ledger_sha256"], "Checkpoint predecessor differs")
    save(cycle / "task.json", task)
    save(cycle / "submission.json", submission, private=True)
    save(cycle / "review-plan.json", {"directory": str(evidence)})
    save(cycle / "operator-config.json", dict(batch=str(cycle.parent), worker_id=submission["worker_id"],
        source_sha256=config["runtime_source"], approved_source_sha256=config["approved_sources"]))
    _, task, result, review, receipt, result_path = science.reviewed(cycle)
    require(review["review_method"] in ("full_replay", "statistical"), "Unexpected participant review method")
    stored_acceptance(submission["result_id"], read(receipt), submission)
    server = science.inspect_server(task["task_id"])
    require(server["audit_holds"] == 0 and server["task"]["state"] == "reviewed"
            and server["result"]["digest"] == submission["submission_digest"], "Accepted state is held or changed")
    destination = cycle / "ledger"
    if not destination.exists():
        previous_path = Path(head["ledger"])
        verify_tree(previous_path)
        require(sha(previous_path / "summary.json") == head["ledger_sha256"], "Predecessor changed")
        catalog = Path(config["population"])
        require(sha(catalog) == config["population_sha256"], "Reference catalog changed")
        with catalog.open() as stream:
            rows = list(csv.DictReader((line for line in stream if not line.startswith("#")), skipinitialspace=True))
        population = {int(r["index"]): r for r in rows}
        require(len(population) == len(rows), "Duplicate catalog IDs")
        previous = read(previous_path / "summary.json")
        models = reference_models(task, population, previous)
        ledger, outcomes = make_ledger(previous, task, result, review, server, models)
        ledger["center"] = plan["center"]
        save(cycle / "accepted-review.json", {"server": server, "receipt_sha256": sha(receipt)})
        inputs = [receipt, result_path, catalog, plan_path, previous_path / "summary.json",
                  previous_path / "result_freeze.json", cycle / "task.json", cycle / "submission.json",
                  cycle / "accepted-review.json", evidence / "plan.json"]
        # Freeze code alongside each small ledger, so future code releases do not invalidate history.
        with tempfile.TemporaryDirectory(prefix=".ledger-", dir=cycle) as temporary:
            stage = Path(temporary) / "ledger"
            stage.mkdir()
            hashes = {str(p): sha(p) for p in inputs}
            for index, name in enumerate(config["source_hashes"]):
                src = ROOT / name
                target = stage / f"source-{index}.py"
                target.write_bytes(src.read_bytes())
                hashes[str(destination / target.name)] = sha(target)
            ledger["sha256"] = hashes
            save(stage / "summary.json", ledger)
            save(stage / "outcomes.json", outcomes)
            save(stage / "result_freeze.json", {"sha256": {p.name: sha(p) for p in stage.iterdir()}})
            os.replace(stage, destination)
    verify_tree(destination)
    for path, digest in read(destination / "summary.json")["sha256"].items():
        require(sha(path) == digest, "Accounting evidence changed")


def publish(cycle, checkout):
    folder = cycle / "publication"
    site = checkout / publication.SNAPSHOT
    if not (folder / "summary.json").exists():
        science.export(cycle, site=site)
    record = read(folder / "summary.json")
    expected = (folder / "snapshot.json").read_bytes()
    previous = (folder / "previous_snapshot.json").read_bytes()
    require(publication.digest(expected) == record["snapshot_sha256"]
            and publication.digest(previous) == record["previous_sha256"], "Publication evidence changed")
    require(site.read_bytes() in (previous, expected), "Publication checkout changed")
    if site.read_bytes() == previous:
        site.write_bytes(expected)
    commit = publication.confirm_git_publication(checkout, expected, previous)
    publication.publish_site(expected, record["previous_sha256"], "travel-" + cycle.name.split("-")[1], commit)
    save(folder / "deployed.json", dict(git_commit=commit, snapshot_sha256=record["snapshot_sha256"],
                                       https_verified=True, epoch=time.time()))


def sync_checkout(checkout, head):
    """Adopt unrelated GitHub changes only while no publication is in flight."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=checkout, timeout=60)
    if git("status", "--porcelain") or sha(checkout / publication.SNAPSHOT) != head["snapshot_sha256"]:
        return
    git("fetch", "--quiet", "origin", "main")
    require(publication.digest(git("show", "origin/main:" + publication.SNAPSHOT)) == head["snapshot_sha256"],
            "Another publisher changed the scientific snapshot")
    git("merge", "--ff-only", "origin/main")


def run(batch):
    config = read(batch / "config.json")
    require(source_hash() == config["runtime_source"], "Publication environment changed")
    for name, digest in config["source_hashes"].items():
        require(sha(ROOT / name) == digest, "Publication implementation changed")
    head_path = batch / "head.json"
    head = read(head_path) if head_path.exists() else config["base"]
    checkout = Path(config["checkout"])
    synced = False
    for evidence in sorted(Path(config["results"]).iterdir(), key=lambda p: int(p.name)):
        number = int(evidence.name)
        if number <= head["result_id"] or not (evidence / "posted.json").exists():
            continue
        if read(evidence / "posted.json")["disposition"] != "accepted":
            continue
        if not synced:
            sync_checkout(checkout, head)
            synced = True
        cycle = batch / f"result-{number}"
        prepare(cycle, config, head, evidence)
        publish(cycle, checkout)
        head = dict(result_id=number, ledger=str(cycle / "ledger"),
                    ledger_sha256=sha(cycle / "ledger/summary.json"),
                    snapshot_sha256=read(cycle / "publication/summary.json")["snapshot_sha256"])
        save(head_path, head, private=True)
    # Full snapshots remain in GitHub; retain receipts and compact scientific evidence locally.
    for cycle in batch.glob("result-*"):
        if int(cycle.name.split("-")[1]) <= head["result_id"] and (cycle / "publication/deployed.json").exists():
            for name in ("snapshot.json", "previous_snapshot.json"):
                (cycle / "publication" / name).unlink(missing_ok=True)
    save(batch / "status.json", dict(state="idle", checked_epoch=time.time(), last_result_id=head["result_id"]), private=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    batch = args.batch.resolve()
    require(batch.parent == ROOT / "outputs", "Publication batch must be an operator output directory")
    os.umask(0o077)
    with (batch / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (batch / "STOP").exists() or (batch / "hold.json").exists():
            return
        try:
            run(batch)
        except (publication.PublicationRetry, subprocess.TimeoutExpired) as error:
            save(batch / "status.json", dict(state="retry", checked_epoch=time.time(), reason=str(error)[:400]), private=True)
        except subprocess.CalledProcessError as error:
            if error.returncode == 255:
                save(batch / "status.json", dict(state="retry", checked_epoch=time.time(), reason="SSH connection interrupted"), private=True)
            else:
                save(batch / "hold.json", dict(state="held", checked_epoch=time.time(), reason=str(error)[:400]), private=True)
                raise
        except Exception as error:
            save(batch / "hold.json", dict(state="held", checked_epoch=time.time(), reason=str(error)[:400]), private=True)
            raise


if __name__ == "__main__":
    main()
