"""An explicit follow-up cut; not a calibrated artifact or exclusion test."""

import math

POLICY = {
    "version": "repeat-support-v1",
    "minimum_coverage": 0.95,
    "minimum_anchor_sectors": 3,
    "minimum_measured_snr": 3.0,
    "minimum_predicted_snr": 3.0,
    "minimum_failed_sectors": 2,
}


def assess(sectors):
    if len({row["sector"] for row in sectors}) != len(sectors):
        raise ValueError("Repeated sector measurements")
    rows = []
    for row in sectors:
        flux, sigma, coverage = (float(row[key]) for key in ("flux", "sigma", "coverage"))
        if not all(map(math.isfinite, [flux, sigma, coverage])) or sigma <= 0 or not 0 <= coverage <= 1:
            raise ValueError("Finite flux, positive uncertainty and valid coverage required")
        rows.append(dict(sector=int(row["sector"]), flux=flux, sigma=sigma,
                         coverage=coverage, measured_snr=flux / sigma))
    anchors = [row for row in rows if row["coverage"] >= POLICY["minimum_coverage"]]
    if len(anchors) < POLICY["minimum_anchor_sectors"]:
        return dict(disposition="unassessed", reason="insufficient_coverage", sectors=rows)
    reference = min(anchors, key=lambda row: (row["sigma"], row["sector"]))
    if reference["measured_snr"] < POLICY["minimum_measured_snr"]:
        return dict(disposition="unassessed", reason="weak_reference_sector", sectors=rows)
    failed = []
    for row in rows:
        row["predicted_snr"] = reference["flux"] / row["sigma"]
        row["flux_deficit_sigma"] = (reference["flux"] - row["flux"]) / math.hypot(reference["sigma"], row["sigma"])
        row["fails_repeat_support"] = (
            row["sector"] != reference["sector"]
            and row["coverage"] >= POLICY["minimum_coverage"]
            and row["predicted_snr"] >= POLICY["minimum_predicted_snr"]
            and row["measured_snr"] < POLICY["minimum_measured_snr"]
        )
        if row["fails_repeat_support"]:
            failed.append(row["sector"])
    rest = [row for row in anchors if row["sector"] != reference["sector"]]
    weight = sum(row["sigma"]**-2 for row in rest)
    flux = sum(row["flux"] / row["sigma"]**2 for row in rest) / weight
    sigma = weight**-0.5
    dismissed = len(failed) >= POLICY["minimum_failed_sectors"]
    return dict(
        disposition="dismissed_repeat_support" if dismissed else "retained",
        reason="weak_in_multiple_capable_sectors" if dismissed else "repeat_support_cut_not_triggered",
        reference_sector=reference["sector"], failed_sectors=failed, sectors=rows,
        other_sectors_flux=flux, other_sectors_sigma=sigma, other_sectors_snr=flux / sigma,
        reference_flux_tension=(reference["flux"] - flux) / math.hypot(reference["sigma"], sigma),
        confirmed_artifact=False, sensitivity_credit=0,
    )
