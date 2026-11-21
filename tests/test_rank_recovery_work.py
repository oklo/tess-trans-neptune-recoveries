import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rank_recovery_work import fit_recovery, fit_runtime, score


def test_brightness_model_is_monotone_and_repeated_ids_do_not_add_trials():
    rows = [
        {"reference_index": i, "V": 18 + i / 10, "recovered": i < 20} for i in range(40)
    ]
    field = {"state": "complete", "injections": rows}
    once = fit_recovery({"fields": [field]})
    twice = fit_recovery({"fields": [field, field]})
    assert once == twice
    assert once["coefficients"][1] < 0
    runtime = {"coefficients": [100, 100, 100]}
    task = lambda v: {
        "sectors": [1, 2, 3],
        "injections": [{"flux": 10 ** ((20.94 - v) / 2.5)}],
    }
    assert (
        score(task(19), once, runtime)["priority_score"]
        > score(task(23), once, runtime)["priority_score"]
    )
    assert score(task(23), once, runtime)["priority_score"] > 0


def test_runtime_fit_remains_positive_and_robust_to_one_slow_download():
    rows = [
        {"sectors": s, "injections": n, "seconds": 50 + 80 * s + 100 * n}
        for s in [3, 4, 5]
        for n in range(1, 6)
    ]
    rows.append({"sectors": 3, "injections": 1, "seconds": 7000})
    fitted = fit_runtime(rows)
    predicted = score(
        {"sectors": [1, 2, 3], "injections": [{"flux": 1}]},
        {"coefficients": [0, -1]},
        fitted,
    )
    assert predicted["estimated_seconds"] == pytest.approx(390, rel=0.15)
    assert all(v >= 0 for v in fitted["coefficients"])


def test_insufficient_data_does_not_silently_rank_everything_zero():
    with pytest.raises(ValueError):
        fit_recovery({"fields": []})
    with pytest.raises(ValueError):
        fit_runtime([])
