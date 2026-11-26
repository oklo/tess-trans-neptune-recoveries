import unittest

from kbo_tess.repeat_support import assess


def sector(number, flux, sigma, coverage=1.0):
    return dict(sector=number, flux=flux, sigma=sigma, coverage=coverage)


class RepeatSupportTests(unittest.TestCase):
    def test_n38_is_an_operational_dismissal_without_exclusion_credit(self):
        rows = [sector(1, 1.1595397813527883, .05319724028265711),
                sector(28, .5984343991361589, .3647706271123781),
                sector(68, .5923355006524018, .2616105661577373),
                sector(95, .7889403863958956, .2491628889180137)]
        result = assess(rows)
        self.assertEqual(result["disposition"], "dismissed_repeat_support")
        self.assertEqual(result["failed_sectors"], [28, 68])
        self.assertAlmostEqual(result["reference_flux_tension"], 2.838090311491229)
        self.assertFalse(result["confirmed_artifact"])
        self.assertEqual(result["sensitivity_credit"], 0)

    def test_shallow_followup_does_not_count_as_failed_repeat_support(self):
        result = assess([sector(1, 1, .05), sector(2, .2, .5), sector(3, .1, .5)])
        self.assertEqual(result["disposition"], "retained")
        self.assertEqual(result["failed_sectors"], [])

    def test_partial_sectors_do_not_create_a_dismissal(self):
        rows = [sector(1, 1, .05), sector(2, 1, .2), sector(3, .1, .2), sector(4, 0, .2, .5)]
        self.assertEqual(assess(rows)["failed_sectors"], [3])
        self.assertEqual(assess(rows)["disposition"], "retained")

    def test_strict_snr_boundary_and_negative_flux(self):
        rows = [sector(1, 3, .05), sector(2, 3, 1), sector(3, -.1, 1)]
        self.assertEqual(assess(rows)["failed_sectors"], [3])
        rows[1]["flux"] = 2.999
        self.assertEqual(assess(rows)["disposition"], "dismissed_repeat_support")

    def test_insufficient_coverage_and_bad_measurements_are_not_rejections(self):
        self.assertEqual(assess([sector(1, 1, .05), sector(2, 0, .2)])["disposition"], "unassessed")
        for bad in [sector(1, 1, 0), sector(1, float("nan"), 1), sector(1, 1, 1, 2)]:
            with self.assertRaises(ValueError):
                assess([bad])
        with self.assertRaises(ValueError):
            assess([sector(1, 1, .1), sector(1, 1, .1)])


if __name__ == "__main__":
    unittest.main()
