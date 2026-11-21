"""Add disjoint, reviewed direct-prior trials without dropping elapsed work."""
from datetime import datetime
import math

from .candidate_evidence import integer
from .direct_prior_calibration import direct_yield


def append_reviewed_field(previous,previous_ids,models,outcomes,threshold,*,
                          field_start_utc,field_end_utc,review_end_utc,physical_flags):
    """Separate the new-batch clock from historical cumulative sensitivity."""
    if type(physical_flags) is not int or physical_flags!=0:
        raise ValueError('Physical candidate follow-up takes priority over yield credit')
    if (previous['state']!='complete' or previous['assumed_existence_probability']!=.2
            or previous['assumed_V_minus_T']!=.5 or previous['reference_models']!=21740):
        raise ValueError('Previous population and brightness assumptions differ')
    before=[integer(v) for v in previous_ids];after=[integer(m['reference_index']) for m in models]
    if (len(before)!=len(set(before)) or len(after)!=len(set(after))
            or set(before)&set(after) or any(v<0 for v in before+after)):
        raise ValueError('Overlapping or duplicate orbit IDs require joint reassessment')
    if any(m['assumed_V_minus_T']!=.5 for m in models):
        raise ValueError('Direct trials use a different color assumption')
    times=[datetime.fromisoformat(value) for value in [previous['batch_start_utc'],
        previous['batch_end_utc'],field_start_utc,field_end_utc,review_end_utc]]
    if any(t.utcoffset() is None for t in times) or not (times[0]<times[1]<=times[2]<times[3]<=times[4]):
        raise ValueError('Timezone-aware acquisition, completion and review ordering required')
    cumulative=previous['cumulative_expected_planets_proxy'];batch=previous['new_batch_expected_planets_proxy']
    if (not all(math.isfinite(v) for v in [cumulative,batch])
            or not 0<=batch<=cumulative<=.2*len(before)/21740+1e-18
            or len(before)+len(after)>21740):
        raise ValueError('Invalid previous expectation ledger')
    direct=direct_yield(models,outcomes,threshold)
    added=direct['expected_planets_proxy']
    batch_hours=(times[4]-times[0]).total_seconds()/3600
    production_hours=(times[3]-times[2]).total_seconds()/3600
    verified_hours=(times[4]-times[2]).total_seconds()/3600
    return dict(state='complete',assumed_existence_probability=.2,assumed_V_minus_T=.5,reference_models=21740,
        previous_cumulative_proxy=cumulative,added_expected_planets_proxy=added,
        cumulative_expected_planets_proxy=cumulative+added,new_batch_expected_planets_proxy=batch+added,
        batch_start_utc=previous['batch_start_utc'],batch_end_utc=review_end_utc,
        batch_elapsed_hours=batch_hours,new_batch_proxy_per_hour=(batch+added)/batch_hours,
        field_started_utc=field_start_utc,field_completed_utc=field_end_utc,field_reviewed_utc=review_end_utc,
        field_and_calibration_hours=production_hours,field_and_calibration_proxy_per_hour=added/production_hours,
        field_verified_hours=verified_hours,field_verified_proxy_per_hour=added/verified_hours,
        direct_calibration=direct,accounted_reference_ids=sorted(before+after),
        newly_accounted_reference_ids=after,physical_followup_flags=0,actual_discoveries=0,
        calibrated_excluded_probability=None,cumulative_expected_planets_per_hour=None,paired_algorithm_speedup=None,
        timing='Batch includes every elapsed hour from its original acquisition start through this field review, including rejected fields, waits, development, controls and calibration. Historical expectation is excluded from the batch rate numerator.',
        interpretation='Adds only disjoint finite-prior recovery indicators at fixed catalog brightness/color. Earlier heterogeneous estimates are preserved. These are engineering sensitivity proxies, not calibrated exclusion probability, general completeness or a measured paired algorithm speedup.')
