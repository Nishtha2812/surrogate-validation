"""Loading SXS numerical-relativity waveforms and surrogate waveforms."""

import numpy as np

# Parameter range each surrogate was trained on. Points outside it are
# extrapolation, which is where we expect accuracy to degrade.
TRAINING_RANGE = {
    "NRHybSur3dq8": {"q_max": 8.0, "chi_max": 0.8, "precessing": False},
    "NRSur7dq4": {"q_max": 4.0, "chi_max": 0.8, "precessing": True},
}


def _meta(meta, *names, default=None):
    """Read the first metadata field that exists."""
    for name in names:
        try:
            value = meta[name]
        except (KeyError, TypeError):
            value = getattr(meta, name, None)
        if value is not None:
            return value
    return default


def load_nr(sim_id, ell_max=4):
    """Return t, modes, params for one SXS simulation."""
    import sxs

    sim = sxs.load(sim_id)
    h = sim.h
    t = np.asarray(h.t)
    modes = {}
    for ell in range(2, ell_max + 1):
        for m in range(-ell, ell + 1):
            modes[(ell, m)] = np.asarray(h[:, h.index(ell, m)])

    meta = sim.metadata
    q = float(_meta(meta, "reference_mass_ratio", "initial_mass_ratio"))
    chi1 = np.asarray(_meta(meta, "reference_dimensionless_spin1", "initial_dimensionless_spin1"), float)
    chi2 = np.asarray(_meta(meta, "reference_dimensionless_spin2", "initial_dimensionless_spin2"), float)
    t_ref = float(_meta(meta, "reference_time", default=t[0] + 300.0))
    ecc = _meta(meta, "reference_eccentricity", default=np.nan)
    try:
        ecc = float(ecc)
    except (TypeError, ValueError):
        ecc = np.nan

    if q < 1:
        q, chi1, chi2 = 1 / q, chi2, chi1

    return t, modes, {"sim_id": sim_id, "q": q, "chi1": chi1, "chi2": chi2,
                      "t_ref": t_ref, "ecc": ecc}


def load_surrogate(name):
    """Load a surrogate, downloading its data file the first time."""
    import gwsurrogate

    try:
        return gwsurrogate.LoadSurrogate(name)
    except Exception:
        gwsurrogate.catalog.pull(name)
        return gwsurrogate.LoadSurrogate(name)


def gw_frequency_at(t, modes, t0, span=200.0):
    """(2,2)-mode GW frequency (cycles per M) around t0."""
    h22 = modes[(2, 2)]
    omega = np.abs(np.gradient(np.unwrap(np.angle(h22)), t))
    sel = (t >= t0) & (t <= t0 + span)
    if not sel.any():
        sel = slice(0, 50)
    return float(np.median(omega[sel])) / (2 * np.pi)


def eval_surrogate(sur, name, q, chi1, chi2, f_low, dt=0.1):
    """Evaluate the surrogate starting at (2,2) frequency f_low.
    Never use f_low=0 for NRHybSur3dq8: it runs out of memory."""
    if not TRAINING_RANGE[name]["precessing"]:
        chi1 = [0.0, 0.0, float(chi1[2])]
        chi2 = [0.0, 0.0, float(chi2[2])]
    t, h, _ = sur(q, list(chi1), list(chi2), dt=dt, f_low=f_low)
    return np.asarray(t), {key: np.asarray(v) for key, v in h.items()}


def in_training_range(name, q, chi1, chi2):
    rng = TRAINING_RANGE[name]
    return (q <= rng["q_max"] + 0.01
            and np.linalg.norm(chi1) <= rng["chi_max"] + 0.01
            and np.linalg.norm(chi2) <= rng["chi_max"] + 0.01)


def catalog_dataframe():
    """The SXS catalog as a pandas DataFrame."""
    import sxs

    try:
        return sxs.load("dataframe")
    except Exception:
        return sxs.load("simulations").dataframe
