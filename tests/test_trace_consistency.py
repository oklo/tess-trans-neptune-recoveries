import pytest
from kbo_tess.trace_consistency import assess_flux_contrast


def test_brightness_range_and_uncertain_miss():
    sectors = [dict(sector=i, flux=f, sigma=.01, coverage=1.) for i,f in enumerate([2., 1., 1.5])]
    assert assess_flux_contrast(sectors)["disposition"] == "retain"
    sectors[1].update(flux=0., sigma=1.)
    assert assess_flux_contrast(sectors)["disposition"] == "retain"
    sectors[1].update(sigma=.05)
    assert assess_flux_contrast(sectors)["disposition"] == "anomaly"
    sectors[1].update(coverage=.94)
    assert assess_flux_contrast(sectors)["disposition"] == "insufficient_coverage"
    with pytest.raises(ValueError):
        assess_flux_contrast(sectors + sectors[:1])
