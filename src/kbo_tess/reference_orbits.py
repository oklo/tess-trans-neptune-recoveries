"""Solar two-body propagation of the rounded P9 reference orbital elements.

These are planning coordinates, not precise candidate ephemerides. Element
rounding must be assessed against the catalog's reference sky positions.
Angles are J2000 ecliptic degrees; outputs are equatorial AU and AU/day.
"""
import numpy as np

GAUSS_K = .01720209895
OBLIQUITY = np.deg2rad(23.439291111)


def elements_state(a, e, inclination, longitude_perihelion, node, mean_anomaly, days=0.):
    a, e, inclination, longitude_perihelion, node, mean_anomaly, days = np.broadcast_arrays(
        *[np.asarray(x, float) for x in [a, e, inclination, longitude_perihelion, node, mean_anomaly, days]])
    if (np.any(a <= 0) or np.any((e < 0) | (e >= 1))
            or not all(np.isfinite(x).all() for x in [a, e, inclination, longitude_perihelion, node, mean_anomaly, days])):
        raise ValueError('Finite elliptic elements and time offsets required')
    n = GAUSS_K/a**1.5
    mean = (np.deg2rad(mean_anomaly)+n*days+np.pi) % (2*np.pi)-np.pi
    eccentric = np.where(e < .8, mean, np.where(mean >= 0, np.pi, -np.pi))
    for _ in range(50):
        correction = (eccentric-e*np.sin(eccentric)-mean)/(1-e*np.cos(eccentric))
        eccentric -= correction
        if np.max(abs(correction)) < 1e-13:
            break
    else:
        raise ArithmeticError('Kepler equation did not converge')
    c, s = np.cos(eccentric), np.sin(eccentric); b = np.sqrt(1-e*e)
    plane_position = np.stack([a*(c-e), a*b*s], axis=-1)
    rate = n/(1-e*c)
    plane_velocity = np.stack([-a*s*rate, a*b*c*rate], axis=-1)
    inc = np.deg2rad(inclination); asc = np.deg2rad(node)
    arg = np.deg2rad(longitude_perihelion-node)
    ca, sa = np.cos(asc), np.sin(asc); cw, sw = np.cos(arg), np.sin(arg)
    ci, si = np.cos(inc), np.sin(inc)
    x_axis = np.stack([ca*cw-sa*sw*ci, sa*cw+ca*sw*ci, sw*si], axis=-1)
    y_axis = np.stack([-ca*sw-sa*cw*ci, -sa*sw+ca*cw*ci, cw*si], axis=-1)
    rotation = np.array([[1., 0., 0.], [0., np.cos(OBLIQUITY), -np.sin(OBLIQUITY)],
                         [0., np.sin(OBLIQUITY), np.cos(OBLIQUITY)]])
    basis = np.stack([x_axis@rotation.T, y_axis@rotation.T], axis=-1)
    return (np.einsum('...ij,...j->...i', basis, plane_position),
            np.einsum('...ij,...j->...i', basis, plane_velocity))


def sky_coordinates(vector):
    vector = np.asarray(vector, float)
    return np.stack([np.rad2deg(np.arctan2(vector[..., 1], vector[..., 0])) % 360,
        np.rad2deg(np.arctan2(vector[..., 2], np.linalg.norm(vector[..., :2], axis=-1)))], axis=-1)


def sky_separation_arcsec(first, second):
    def rays(sky):
        ra, dec = np.moveaxis(np.deg2rad(sky), -1, 0)
        return np.stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)], axis=-1)
    chord = np.linalg.norm(rays(first)-rays(second), axis=-1)
    return np.rad2deg(2*np.arcsin(np.minimum(1, chord/2)))*3600
