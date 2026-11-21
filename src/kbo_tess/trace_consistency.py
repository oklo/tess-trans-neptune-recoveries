"""Follow-up triage for a positive, slowly varying TESS source."""

from itertools import permutations
from math import hypot, isfinite

POLICY = {
    "version": "sector-flux-contrast-v1",
    "minimum_coverage": 0.95,
    "minimum_anchor_sectors": 3,
    "allowed_flux_ratio": 2.0,
    "minimum_discrepancy": 5.0,
}


def assess_flux_contrast(sectors):
    """Return the strongest disagreement after allowing twofold brightness variation."""
    if len({s["sector"] for s in sectors}) != len(sectors):
        raise ValueError("Sector measurements must be distinct")
    for s in sectors:
        if (not all(isfinite(s[k]) for k in ("coverage", "flux", "sigma"))
                or not 0 <= s["coverage"] <= 1 or s["sigma"] <= 0):
            raise ValueError("Invalid sector photometry")
    anchors = [s for s in sectors if s["coverage"] >= POLICY["minimum_coverage"]
               and s.get("anchor", True)]
    if len(anchors) < POLICY["minimum_anchor_sectors"]:
        return {"disposition": "insufficient_coverage"}
    ratio = POLICY["allowed_flux_ratio"]
    pairs = [
        ((a["flux"] - ratio * b["flux"]) / hypot(a["sigma"], ratio * b["sigma"]), a, b)
        for a, b in permutations(anchors, 2)
        if a["flux"] >= POLICY["minimum_discrepancy"] * a["sigma"]
        and a["flux"] > b["flux"]
    ]
    if not pairs:
        return {"disposition": "retain"}
    z, high, low = max(pairs, key=lambda item: item[0])
    return dict(disposition="anomaly" if z >= POLICY["minimum_discrepancy"] else "retain",
                discrepancy=z, high_sector=high["sector"], low_sector=low["sector"])
