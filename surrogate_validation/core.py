"""Core math for comparing two sets of gravitational-wave modes.

Everything here is plain NumPy/SciPy, so it can be tested without
downloading any waveform data (see tests/test_core.py).

Conventions
-----------
* A waveform is a dict {(ell, m): complex array} sampled on a time array t,
  in geometric units (t in M, h = r h / M).
* The error measure is the time-domain error used in the NRSur papers
  (e.g. Varma et al. 2019, Phys. Rev. Research 1, 033015, Eq. 21):

      E = 1/2 * sum_lm int |h_NR - h_sur|^2 dt / sum_lm int |h_NR|^2 dt

  E = 0 is a perfect match. It is independent of total mass and of any
  detector noise curve, which keeps the first stage of the project simple.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


def complete_negative_m(modes):
    """Fill in missing m < 0 modes using h_{l,-m} = (-1)^l conj(h_{l,m}).

    This symmetry holds for non-precessing (aligned-spin) binaries only.
    Do not use it for precessing systems.
    """
    out = dict(modes)
    for (ell, m), h in modes.items():
        if m > 0 and (ell, -m) not in out:
            out[(ell, -m)] = (-1) ** ell * np.conj(h)
    return out


def total_amplitude(modes):
    """sqrt(sum_lm |h_lm|^2), the frame-independent amplitude."""
    return np.sqrt(sum(np.abs(h) ** 2 for h in modes.values()))


def peak_time(t, modes):
    """Time of maximum total amplitude, refined with a parabola through
    the three samples around the discrete maximum."""
    amp = total_amplitude(modes)
    i = int(np.argmax(amp))
    if 0 < i < len(t) - 1:
        y0, y1, y2 = amp[i - 1], amp[i], amp[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            offset = 0.5 * (y0 - y2) / denom
            return t[i] + offset * (t[i + 1] - t[i])
    return t[i]


def resample(t, modes, t_new):
    """Interpolate every mode onto t_new (cubic splines on real and
    imaginary parts). t_new must lie inside [t[0], t[-1]]."""
    out = {}
    for key, h in modes.items():
        re = CubicSpline(t, h.real)(t_new)
        im = CubicSpline(t, h.imag)(t_new)
        out[key] = re + 1j * im
    return out


def time_domain_error(t, h_ref, h_test, keys):
    """E from the module docstring, over the given mode keys."""
    num = 0.0
    den = 0.0
    for key in keys:
        num += np.trapezoid(np.abs(h_ref[key] - h_test[key]) ** 2, t)
        den += np.trapezoid(np.abs(h_ref[key]) ** 2, t)
    return 0.5 * num / den


def _best_phase(t, h_ref, h_test, keys):
    """Rotation angle phi about the z-axis (h_lm -> h_lm e^{i m phi})
    that minimizes E, for fixed time alignment.

    E(phi) = 1/2 [A + B - 2 Re sum_m c_m e^{i m phi}] / A, so we only
    need the overlaps c_m = sum_l int conj(h_ref) h_test dt. A dense
    grid plus a local refinement avoids the pi-ambiguity of the (2,2)
    mode picking the wrong branch.
    """
    c = {}
    for key in keys:
        m = key[1]
        c[m] = c.get(m, 0) + np.trapezoid(np.conj(h_ref[key]) * h_test[key], t)

    def neg_overlap(phi):
        return -np.real(sum(cm * np.exp(1j * m * phi) for m, cm in c.items()))

    grid = np.linspace(0, 2 * np.pi, 721)
    vals = [neg_overlap(p) for p in grid]
    p0 = grid[int(np.argmin(vals))]
    step = grid[1] - grid[0]
    res = minimize_scalar(neg_overlap, bounds=(p0 - step, p0 + step), method="bounded")
    return res.x


def rotate(modes, phi):
    """Rotate about the z-axis: h_lm -> h_lm e^{i m phi}."""
    return {key: h * np.exp(1j * key[1] * phi) for key, h in modes.items()}


def align_and_compare(t_ref, h_ref, t_test, h_test, window=(-np.inf, 100.0),
                      dt=0.1, max_time_shift=5.0, keys=None):
    """Align h_test to h_ref and return the time-domain error.

    Steps
    1. Shift both time axes so the peak of the total amplitude is t = 0.
    2. Keep the common time span, cut to `window` (in M, relative to peak).
    3. Optimize a small extra time shift (|tau| <= max_time_shift) and a
       z-rotation phi of h_test that minimize E.

    Parameters
    ----------
    window : (t_start, t_end) relative to the peak. Use t_start to drop the
        early part of the NR run (junk radiation), e.g. the NR reference time.
    keys : list of (ell, m) to include; default = modes present in both.

    Returns
    -------
    dict with E (all modes), E22 (only (2,+-2)), tau, phi, t_start, t_end.
    """
    if keys is None:
        keys = sorted(set(h_ref) & set(h_test))
    if not keys:
        raise ValueError("No common modes between the two waveforms.")

    t_ref = np.asarray(t_ref) - peak_time(t_ref, {k: h_ref[k] for k in keys})
    t_test = np.asarray(t_test) - peak_time(t_test, {k: h_test[k] for k in keys})

    t0 = max(t_ref[0], t_test[0] + max_time_shift, window[0])
    t1 = min(t_ref[-1], t_test[-1] - max_time_shift, window[1])
    if t1 - t0 < 10 * dt:
        raise ValueError(f"Common time window too short: [{t0:.1f}, {t1:.1f}] M")

    t = np.arange(t0, t1, dt)
    ref = resample(t_ref, {k: h_ref[k] for k in keys}, t)
    test_keys = {k: h_test[k] for k in keys}

    def error_at_shift(tau):
        base = resample(t_test, test_keys, t + tau)
        best = None
        for sign in (1, -1):  # codes can differ by an overall sign
            test = {k: sign * h for k, h in base.items()}
            phi = _best_phase(t, ref, test, keys)
            err = time_domain_error(t, ref, rotate(test, phi), keys)
            if best is None or err < best[0]:
                best = (err, phi, sign)
        return best

    res = minimize_scalar(lambda tau: error_at_shift(tau)[0],
                          bounds=(-max_time_shift, max_time_shift),
                          method="bounded", options={"xatol": 1e-3})
    tau = res.x
    err, phi, sign = error_at_shift(tau)

    test = rotate({k: sign * h for k, h in resample(t_test, test_keys, t + tau).items()}, phi)
    keys22 = [k for k in keys if k[0] == 2 and abs(k[1]) == 2]
    err22 = time_domain_error(t, ref, test, keys22) if keys22 else np.nan

    return {"E": err, "E22": err22, "tau": tau, "phi": phi % (2 * np.pi),
            "sign": sign, "t_start": t0, "t_end": t1, "n_modes": len(keys)}
