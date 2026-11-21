import copy
import unittest

from kbo_tess.direct_yield_ledger import append_reviewed_field


def example():
    unit=.2/21740
    previous=dict(state='complete',assumed_existence_probability=.2,assumed_V_minus_T=.5,reference_models=21740,
        cumulative_expected_planets_proxy=3*unit,new_batch_expected_planets_proxy=unit,
        batch_start_utc='2026-11-04T00:00:00+00:00',batch_end_utc='2026-11-04T02:00:00+00:00')
    models=[dict(reference_index=i,reference_V=20.,assumed_V_minus_T=.5,flux=1.) for i in [4,5]]
    outcomes=[dict(models[0],assessed=True,recovered=True,strongest_nearby_score=11.,
        control_maximum=10.,native_responses_passed=True,all_nearby_alternatives_scored=True,baseline_above_control=False),
        dict(models[1],assessed=False,recovered=None,control_maximum=10.,reason='native_response_or_fidelity_failed')]
    clocks=dict(field_start_utc='2026-11-04T04:00:00+00:00',field_end_utc='2026-11-04T05:00:00+00:00',
        review_end_utc='2026-11-04T06:00:00+00:00',physical_flags=0)
    return previous,[1,2,3],models,outcomes,10.,clocks


class DirectYieldLedgerTests(unittest.TestCase):
    def test_clock_includes_waits_development_review_but_not_historical_numerator(self):
        previous,ids,models,outcomes,threshold,clocks=example()
        actual=append_reviewed_field(previous,ids,models,outcomes,threshold,**clocks)
        self.assertAlmostEqual(actual['cumulative_expected_planets_proxy'],.8/21740)
        self.assertAlmostEqual(actual['new_batch_proxy_per_hour'],.4/21740/6)
        self.assertAlmostEqual(actual['field_and_calibration_proxy_per_hour'],.2/21740)
        self.assertAlmostEqual(actual['field_verified_proxy_per_hour'],.1/21740)
        self.assertEqual(actual['accounted_reference_ids'],[1,2,3,4,5])
        self.assertEqual(actual['direct_calibration']['unassessed_ids'],[5])
        self.assertIsNone(actual['calibrated_excluded_probability'])
        self.assertIsNone(actual['paired_algorithm_speedup'])

    def test_overlaps_duplicates_candidate_flags_and_changed_assumptions_fail(self):
        previous,ids,models,outcomes,threshold,clocks=example()
        for before in [[1,2,4],[1,1,2]]:
            with self.subTest(ids=before),self.assertRaises(ValueError):
                append_reviewed_field(previous,before,models,outcomes,threshold,**clocks)
        with self.assertRaises(ValueError):
            append_reviewed_field(previous,ids,models,outcomes,threshold,**dict(clocks,physical_flags=1))
        for key,value in [('assumed_existence_probability',.3),('assumed_V_minus_T',1.),('reference_models',100000)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                append_reviewed_field(dict(previous,**{key:value}),ids,models,outcomes,threshold,**clocks)

    def test_incomplete_or_unassessed_miss_cannot_increase_credit(self):
        previous,ids,models,outcomes,threshold,clocks=example()
        with self.assertRaises(ValueError):append_reviewed_field(previous,ids,models,outcomes[:1],threshold,**clocks)
        wrong=copy.deepcopy(outcomes);wrong[1]['recovered']=False
        with self.assertRaises(ValueError):append_reviewed_field(previous,ids,models,wrong,threshold,**clocks)
        wrong=copy.deepcopy(outcomes);wrong[0]['strongest_nearby_score']=10.
        with self.assertRaises(ValueError):append_reviewed_field(previous,ids,models,wrong,threshold,**clocks)

    def test_invalid_clock_and_numerator_cannot_make_rate_look_better(self):
        previous,ids,models,outcomes,threshold,clocks=example()
        for key,value in [('field_start_utc','2026-11-04T01:00:00+00:00'),
                          ('review_end_utc','2026-11-04T04:30:00+00:00'),('field_start_utc','2026-11-04T04:00:00')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                append_reviewed_field(previous,ids,models,outcomes,threshold,**dict(clocks,**{key:value}))
        for value in [-1.,float('nan'),previous['cumulative_expected_planets_proxy']+1.]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                append_reviewed_field(dict(previous,new_batch_expected_planets_proxy=value),ids,models,outcomes,threshold,**clocks)


if __name__=='__main__':unittest.main()
