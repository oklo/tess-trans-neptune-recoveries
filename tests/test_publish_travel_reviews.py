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
    monkeypatch.setattr(publisher.publication, "confirm_git_publication", Mock(return_value="commit"))
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
