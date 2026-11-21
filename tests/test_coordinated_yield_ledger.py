import copy

import pytest
from test_direct_yield_ledger import example

from kbo_tess.coordinated_yield_ledger import append_reviewed_field
from kbo_tess.direct_yield_ledger import append_reviewed_field as sequential_append


def test_sequential_credit_and_rates_are_unchanged():
    previous, ids, models, outcomes, threshold, clocks = example()
    actual = append_reviewed_field(previous, ids, models, outcomes, threshold, **clocks)
    expected = sequential_append(previous, ids, models, outcomes, threshold, **clocks)
    actual.pop("timing")
    expected.pop("timing")
    assert actual == expected


@pytest.mark.parametrize("start", ["00:00:00", "01:00:00", "02:00:00"])
def test_overlapping_acquisition_keeps_real_times_and_one_batch_clock(start):
    previous, ids, models, outcomes, threshold, clocks = example()
    clocks["field_start_utc"] = f"2026-11-04T{start}+00:00"
    actual = append_reviewed_field(previous, ids, models, outcomes, threshold, **clocks)
    assert actual["field_started_utc"] == clocks["field_start_utc"]
    assert actual["field_completed_utc"] == clocks["field_end_utc"]
    assert actual["field_reviewed_utc"] == clocks["review_end_utc"]
    assert actual["batch_elapsed_hours"] == 6
    assert actual["new_batch_proxy_per_hour"] == pytest.approx(0.4 / 21740 / 6)
    assert actual["accounted_reference_ids"] == [1, 2, 3, 4, 5]


def test_late_arriving_earlier_review_does_not_shorten_clock():
    previous, ids, models, outcomes, threshold, clocks = example()
    previous["batch_end_utc"] = "2026-11-04T10:00:00+00:00"
    actual = append_reviewed_field(previous, ids, models, outcomes, threshold, **clocks)
    assert actual["batch_end_utc"] == previous["batch_end_utc"]
    assert actual["batch_elapsed_hours"] == 10
    assert actual["new_batch_proxy_per_hour"] == pytest.approx(0.4 / 21740 / 10)
    assert actual["field_verified_hours"] == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("field_start_utc", "2026-11-03T23:59:59+00:00"),
        ("field_start_utc", "2026-11-04T04:00:00"),
        ("field_start_utc", "2026-11-04T05:00:00+00:00"),
        ("review_end_utc", "2026-11-04T04:30:00+00:00"),
    ],
)
def test_invalid_acquisition_or_review_time_still_fails(key, value):
    previous, ids, models, outcomes, threshold, clocks = example()
    clocks[key] = value
    with pytest.raises(ValueError):
        append_reviewed_field(previous, ids, models, outcomes, threshold, **clocks)


def test_overlap_does_not_allow_double_credit_or_failed_checks():
    previous, ids, models, outcomes, threshold, clocks = example()
    clocks["field_start_utc"] = "2026-11-04T01:00:00+00:00"
    for before in ([1, 2, 4], [1, 1, 2]):
        with pytest.raises(ValueError):
            append_reviewed_field(
                previous, before, models, outcomes, threshold, **clocks
            )
    with pytest.raises(ValueError):
        append_reviewed_field(
            previous, ids, models, outcomes, threshold, **dict(clocks, physical_flags=1)
        )
    wrong = copy.deepcopy(outcomes)
    wrong[0]["native_responses_passed"] = False
    with pytest.raises(ValueError):
        append_reviewed_field(previous, ids, models, wrong, threshold, **clocks)
    with pytest.raises(ValueError):
        append_reviewed_field(previous, ids, models, outcomes[:1], threshold, **clocks)
