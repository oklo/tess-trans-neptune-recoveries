import copy
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import publish_travel_reviews as publisher
from p9search.storage import write_json as save


def test_reference_weight_requires_catalog_brightness_and_disjoint_ids():
    population = {12: dict(V="20.0", ZTF="0", DES="0", PS1="0")}
    task = {"injections": [dict(label="prior 12", flux=10 ** ((20.44 - 19.5) / 2.5))]}
    previous = dict(accounted_reference_ids=[], retained_unassessed_prior_ids=[])
    assert publisher.reference_models(task, population, previous)[0]["reference_index"] == 12
    for key in previous:
        with pytest.raises(ValueError, match="Overlapping"):
            publisher.reference_models(task, population, dict(previous, **{key: [12]}))
    wrong = copy.deepcopy(task)
    wrong["injections"][0]["flux"] *= 2
    with pytest.raises(ValueError, match="brightness"):
        publisher.reference_models(wrong, population, previous)
    wrong = {"injections": task["injections"] * 2}
    with pytest.raises(ValueError, match="Overlapping"):
        publisher.reference_models(wrong, population, previous)


def test_publication_retry_does_not_repeat_accounting_or_export(tmp_path, monkeypatch):
    cycle, checkout = tmp_path / "result-300", tmp_path / "checkout"
    folder = cycle / "publication"
    folder.mkdir(parents=True)
    old, new = b'{"value":1}\n', b'{"value":2}\n'
    (folder / "previous_snapshot.json").write_bytes(old)
    (folder / "snapshot.json").write_bytes(new)
    save(folder / "summary.json", dict(previous_sha256=publisher.publication.digest(old),
                                       snapshot_sha256=publisher.publication.digest(new)))
    site = checkout / publisher.publication.SNAPSHOT
    site.parent.mkdir(parents=True)
    site.write_bytes(new)
    export = Mock(side_effect=AssertionError("Must not export twice"))
    monkeypatch.setattr(publisher.science, "export", export)
    monkeypatch.setattr(publisher, "reviewed_git_publication", Mock(return_value="commit"))
    publish = Mock(side_effect=[publisher.publication.PublicationRetry("lost acknowledgement"), None])
    monkeypatch.setattr(publisher.publication, "publish_site", publish)
    with pytest.raises(publisher.publication.PublicationRetry):
        publisher.publish(cycle, checkout)
    assert not (folder / "deployed.json").exists()
    publisher.publish(cycle, checkout)
    assert (folder / "deployed.json").exists()
    assert not export.called
    site.write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="checkout changed"):
        publisher.publish(cycle, checkout)


def test_checkpoint_retries_same_result_then_skips_completed(tmp_path, monkeypatch):
    batch = tmp_path / "batch"
    batch.mkdir()
    results = tmp_path / "results"
    (results / "300").mkdir(parents=True)
    save(results / "300/posted.json", {"disposition": "accepted"})
    save(batch / "config.json", dict(runtime_source="source", source_hashes={}, results=str(results),
        checkout=str(tmp_path / "checkout"), base=dict(result_id=299, ledger="base", ledger_sha256="base",
                                                     snapshot_sha256="old")))
    monkeypatch.setattr(publisher, "source_hash", lambda: "source")
    monkeypatch.setattr(publisher, "sync_checkout", Mock())
    prepared = []

    def prepare(cycle, config, head, evidence):
        prepared.append(head["result_id"])
        (cycle / "ledger").mkdir(parents=True, exist_ok=True)
        save(cycle / "ledger/summary.json", {"value": 2})
        (cycle / "publication").mkdir(exist_ok=True)
        save(cycle / "publication/summary.json", {"snapshot_sha256": "new"})

    monkeypatch.setattr(publisher, "prepare", prepare)
    publish = Mock(side_effect=[publisher.publication.PublicationRetry("transport"), None])
    monkeypatch.setattr(publisher, "publish", publish)
    with pytest.raises(publisher.publication.PublicationRetry):
        publisher.run(batch)
    assert not (batch / "head.json").exists()
    publisher.run(batch)
    publisher.run(batch)
    assert prepared == [299, 299]
    assert publish.call_count == 2


def test_snapshot_publication_uses_pull_request_never_pushes_main(tmp_path, monkeypatch):
    calls = []
    old, new = b"old", b"new"

    def command(args, **kwargs):
        calls.append(args)
        if args[:3] == ["git", "branch", "--show-current"]:
            return b"main\n"
        if args[:3] == ["git", "show", "HEAD:" + publisher.publication.SNAPSHOT]:
            return new
        if args[:3] == ["git", "show", "origin/main:" + publisher.publication.SNAPSHOT]:
            return old
        if args[:3] == ["git", "diff", "--name-only"]:
            return (publisher.publication.SNAPSHOT + "\n").encode()
        if args[:3] == ["gh", "pr", "list"]:
            return b"[]"
        return b""

    monkeypatch.setattr(publisher.subprocess, "check_output", command)
    with pytest.raises(publisher.publication.PublicationRetry, match="required checks"):
        publisher.reviewed_git_publication(tmp_path, new, old)
    pushes = [c for c in calls if c[:2] == ["git", "push"]]
    assert len(pushes) == 1 and pushes[0][-1].startswith("HEAD:refs/heads/atlas-publication-")
    assert any(c[:3] == ["gh", "pr", "create"] for c in calls)


def accounting_example():
    from datetime import datetime
    from test_direct_yield_ledger import example
    previous, ids, models, outcomes, threshold, clocks = example()
    previous.update(accounted_reference_ids=ids, retained_unassessed_prior_ids=[],
                    retained_followup_searches=[], unresolved_flags_in_other_fields=0,
                    expectation_by_validation={"full_replay": previous["cumulative_expected_planets_proxy"]})
    task = dict(task_id="a" * 64, sectors=[1, 2, 3],
                injections=[dict(label=f"prior {m['reference_index']}") for m in models])
    result = dict(state="searched", candidate_count=0, exclusion_credit=0, candidates=[],
                  control_maximum=threshold, source_sha256="b" * 64,
                  quality_method="usable-pixels-v1",
                  quality=[dict(sector=i, masked_fraction=.2) for i in task["sectors"]],
                  injections=[dict(label=f"prior {r['reference_index']}", assessed=r["assessed"],
                              recovered=r["recovered"], best_score=r.get("strongest_nearby_score"))
                              for r in outcomes])
    review = dict(review_method="consistency", full_replay_verified=False, audit_probability=0,
                  replay_seconds=0)
    epoch = lambda k: int(datetime.fromisoformat(clocks[k]).timestamp())
    server = dict(task=dict(started=epoch("field_start_utc")),
                  result=dict(submitted=epoch("field_end_utc")), review_time=epoch("review_end_utc"))
    return previous, task, result, review, server, models


def test_partial_calibration_adds_only_demonstrated_recovery():
    args = accounting_example()
    ledger, outcomes = publisher.make_ledger(*args)
    assert ledger["added_expected_planets_proxy"] == pytest.approx(.2 / 21740)
    assert ledger["direct_calibration"]["unassessed_ids"] == [5]
    assert outcomes[1]["recovered"] is None
    assert ledger["review_method"] == "consistency"
    assert not ledger["full_replay_verified"]


def test_candidate_retention_needs_no_automatic_replay():
    args = list(accounting_example())
    args[2].update(state="needs_followup", candidate_count=1, candidates=[{}], injections=[])
    ledger, outcomes = publisher.make_ledger(*args)
    assert ledger["added_expected_planets_proxy"] == 0
    assert ledger["retained_unassessed_prior_ids"] == [4, 5]
    assert ledger["physical_followup_flags"] == 1
    assert not ledger["full_replay_verified"]
    assert all(r["recovered"] is None for r in outcomes)


def test_revisit_releases_only_that_fields_unassessed_samples():
    import json
    previous, task, result, _, _, _ = accounting_example()
    previous.update(retained_unassessed_prior_ids=[4, 5, 7], quality_rejected_fields=2)
    old = dict(result, task_id=task['task_id'], state='quality_rejected', injections=[])
    raw = json.dumps(old)
    record = dict(task=dict(id=task['task_id'], spec=json.dumps(task)),
                  result=dict(result=raw, digest=publisher.publication.digest(raw.encode()),
                              status='accepted', verified=1))
    revised = publisher.revisit_predecessor(previous, task, record)
    assert revised['retained_unassessed_prior_ids'] == [7]
    assert revised['quality_rejected_fields'] == 1
    assert previous['retained_unassessed_prior_ids'] == [4, 5, 7]
    assert revised['cumulative_expected_planets_proxy'] == previous['cumulative_expected_planets_proxy']
    wrong = copy.deepcopy(previous)
    wrong['accounted_reference_ids'].append(4)
    with pytest.raises(ValueError, match='overlap'):
        publisher.revisit_predecessor(wrong, task, record)
    wrong = copy.deepcopy(previous)
    wrong['retained_followup_searches'] = [dict(unassessed_prior_ids=[4])]
    with pytest.raises(ValueError, match='candidate'):
        publisher.revisit_predecessor(wrong, task, record)


def test_revisited_field_becomes_measured_and_keeps_rejection_history(tmp_path, monkeypatch):
    import account_planet_nine_quality_rejection as rejection
    args = accounting_example()
    ledger, outcomes = publisher.make_ledger(*args)
    task = args[1]
    ledger.update(sha256={}, center=[100., 20.], source_task=task['task_id'],
                  revisit=dict(previous_result_id=7, previous_state='preflight_failed',
                               footprints=[dict(sector=s, ra_dec_deg=[[1, 2]]) for s in task['sectors']]))
    cycle = tmp_path/'cycle'; (cycle/'ledger').mkdir(parents=True)
    save(cycle/'ledger/summary.json', ledger)
    save(cycle/'ledger/outcomes.json', outcomes)
    save(cycle/'credit-plan.json', dict(field='N1', previous_state='preflight_failed'))
    snapshot = dict(sensitivity_proxy=dict(value=ledger['previous_cumulative_proxy']),
                    searched_fields=0, physical_followup_flags=0,
                    fields=[dict(code='N1', task_id=task['task_id'], state='preflight_failed',
                                 ra_deg=100., dec_deg=20., planned_sectors=task['sectors'],
                                 result='Stopped at mask cutoff', footprints=[])])
    site = tmp_path/'snapshot.json';save(site,snapshot)
    monkeypatch.setattr(rejection,'verify_tree',lambda *a:None)
    publisher.science.export(cycle,site=site)
    actual=publisher.read(cycle/'publication/snapshot.json')
    field=actual['fields'][0]
    assert field['state']=='complete'
    assert field['injections'][1]['recovered'] is None
    assert field['depth_samples'][0]['tested']==1
    assert field['search_history'][0]['state']=='preflight_failed'
    assert len(field['footprints'])==3
    assert actual['searched_fields']==1
    assert actual['sensitivity_proxy']['value']==ledger['cumulative_expected_planets_proxy']


def test_anomaly_keeps_signed_candidate_and_withheld_calibration():
    args = list(accounting_example())
    sectors = [dict(sector=i, coverage=1., flux=f, sigma=.1) for i, f in enumerate([5., 4., .3])]
    args[2].update(state="needs_followup", candidate_count=1, candidates=[dict(sectors=sectors)], injections=[])
    ledger, outcomes = publisher.make_ledger(*args)
    entry = ledger["retained_followup_searches"][-1]
    assert args[2]["candidate_count"] == 1
    assert entry["candidate_flags"] == [] and entry["historical_candidate_flags"] == [1]
    assert entry["flux_anomalies"]["1"]["disposition"] == "anomaly"
    assert ledger["physical_followup_flags"] == 0
    assert ledger["added_expected_planets_proxy"] == 0 and not any(r["assessed"] for r in outcomes)
