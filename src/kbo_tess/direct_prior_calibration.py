"""Prospective calibration at every eligible finite-prior orbit sample.

These conditional finite-population diagnostics do not establish a magnitude
completeness curve, a false-alarm probability or a calibrated exclusion.
"""
import math
import numpy as np
from .reference_orbits import elements_state, sky_coordinates

ELEMENTS = ('a', 'e', 'inc', 'varpi', 'Omega', 'M')


def declared_models(population, coverage, *, color=.5, minimum_sectors=3):
    """Select solely by declared geometry and inclination, never by flux."""
    if not math.isfinite(color) or minimum_sectors < 1:
        raise ValueError('Finite color and positive sector minimum required')
    lookup = {int(row['index']): row for row in population}
    if len(lookup) != len(population):
        raise ValueError('Duplicate prior orbit ID')
    selected = []; seen = set()
    for item in coverage:
        index = int(item['reference_index'])
        if index in seen:
            raise ValueError('Duplicate footprint membership')
        seen.add(index)
        if index not in lookup:
            raise ValueError('Unknown prior orbit ID')
        row = lookup[index]
        if any(int(row[name]) != 0 for name in ['ZTF', 'DES', 'PS1']):
            raise ValueError('Footprint includes an already flagged prior orbit')
        if int(item['sectors_nominal']) < minimum_sectors or float(row['inc']) > 60:
            continue
        v = float(row['V'])
        if not math.isfinite(v) or float(item['reference_V']) != v:
            raise ValueError('Prior brightness provenance differs')
        flux = 10**((20.44-(v-color))/2.5)
        if not math.isfinite(flux) or flux <= 0:
            raise ValueError('Invalid prior injection flux')
        selected.append(dict(reference_index=index, reference_V=v, assumed_V_minus_T=color,
            assumed_T=v-color, flux=flux, **{name: float(row[name]) for name in ELEMENTS}))
    return sorted(selected, key=lambda row: row['reference_index'])


def reference_search_state(model, frame, sun):
    """Convert rounded heliocentric elements into the existing search state."""
    position, velocity = elements_state(*[model[name] for name in ELEMENTS],
        days=frame.reference_time+2457000.-2458270.)
    position = position+np.asarray(sun[0]); velocity = velocity+np.asarray(sun[1])
    relative = position-frame.observer
    distance = np.linalg.norm(relative)
    ray = relative/distance
    xy = frame.wcs.all_world2pix(sky_coordinates(ray)[None], 1,
        tolerance=1e-10, maxiter=50)[0]-frame.origin
    # The ray is not generally perpendicular to the center-frame tangents.
    coefficients = np.linalg.solve(np.column_stack([frame.tangent.T, ray]), velocity)
    parameters = np.r_[xy, coefficients[:2]/distance, 1/distance]
    reconstructed_ray = frame.ray(xy)[0]
    np.testing.assert_allclose(reconstructed_ray/parameters[4]+frame.observer,
        position, rtol=0, atol=2e-10)
    np.testing.assert_allclose(parameters[2:4]@frame.tangent/parameters[4]+
        coefficients[2]*reconstructed_ray, velocity, rtol=0, atol=1e-14)
    if not np.isfinite(parameters).all() or not np.isfinite(coefficients[2]):
        raise ValueError('Nonfinite prior search state')
    return parameters, float(coefficients[2])


def validate_response(*, flux, baseline_flux, injected_flux, identifiable,
                      template_rms_error, relative_template_error):
    """At faint flux require relative fidelity as well as the original limits."""
    values = [flux, baseline_flux, injected_flux, template_rms_error, relative_template_error]
    if not all(math.isfinite(v) for v in values) or flux <= 0:
        return dict(assessed=False, gain=None, reason='nonfinite_response')
    gain = (injected_flux-baseline_flux)/flux
    passed = (identifiable is True and .9 <= gain <= 1.1
              and 0 <= template_rms_error <= .002 and 0 <= relative_template_error <= .1)
    return dict(assessed=passed, gain=gain,
        reason='passed' if passed else 'native_response_or_fidelity_failed')


def direct_yield(models, rows, threshold, *, denominator=21740, existence=.2):
    """Require one outcome per declared model; retain failures as unassessed."""
    if (type(denominator) is not int or denominator <= 0 or not math.isfinite(threshold)
            or not math.isfinite(existence) or not 0 <= existence <= 1):
        raise ValueError('Invalid frozen threshold or population weight')
    expected = {int(m['reference_index']): m for m in models}
    if len(expected) != len(models) or len(models) > denominator:
        raise ValueError('Distinct declared prior models required')
    observed = {}; assessed = 0; recovered = 0; unassessed = []; ambiguous = []
    for row in rows:
        index = int(row['reference_index'])
        if index in observed or index not in expected:
            raise ValueError('Duplicate or undeclared prior recovery')
        observed[index] = row
        model = expected[index]
        for key in ['reference_V', 'assumed_V_minus_T', 'flux']:
            if float(row[key]) != float(model[key]):
                raise ValueError('Prior recovery properties changed')
        if row['control_maximum'] != threshold:
            raise ValueError('Recovery threshold changed')
        if type(row['assessed']) is not bool:
            raise ValueError('Explicit assessment state required')
        if not row['assessed']:
            if row.get('recovered') is not None:
                raise ValueError('Unassessed orbit cannot be a measured miss')
            unassessed.append(index)
            continue
        if row.get('all_nearby_alternatives_scored') is not True or row.get('native_responses_passed') is not True:
            raise ValueError('Complete native validation and blind matching required')
        score = row.get('strongest_nearby_score')
        if score is not None and not math.isfinite(score):
            raise ValueError('Invalid matched score')
        detection = score is not None and score > threshold
        if type(row.get('recovered')) is not bool or row['recovered'] != detection:
            raise ValueError('Recovery differs from the strict frozen threshold')
        if type(row.get('baseline_above_control')) is not bool:
            raise ValueError('Baseline chance diagnostic required')
        if row['baseline_above_control']:
            ambiguous.append(index)
            continue
        assessed += 1; recovered += int(detection)
    if set(observed) != set(expected):
        raise ValueError('Every declared prior model must have an outcome')
    unmeasured = len(unassessed)+len(ambiguous)
    return dict(declared_models=len(models), assessed_models=assessed, recovered_models=recovered,
        unassessed_ids=unassessed, baseline_confounded_ids=ambiguous,
        conditional_finite_population_yield=recovered/denominator,
        expected_planets_proxy=existence*recovered/denominator,
        unassessed_expectation_upper_bound=existence*unmeasured/denominator,
        calibrated_excluded_probability=None,
        meaning='One isolated recovery per equal-weight prior sample, conditional on the declared brightness/color and fixed observed images. Sparse sampling and raw-injection validation limit interpretation.')
