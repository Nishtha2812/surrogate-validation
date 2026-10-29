"""Checks for core.py using a synthetic chirp, so no SXS data is needed.

Run from the repository root:  python -m pytest tests   (or: python tests/test_core.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from surrogate_validation import core  # noqa: E402


def fake_waveform(t, amp_scale=1.0):
    """A chirp-like toy signal with (2,2), (2,1) and (3,3) modes.

    Amplitude rises to a peak at t = 0 and decays afterwards; the orbital
    phase speeds up towards the peak. Not physical, but it has the features
    the alignment code has to handle.
    """
    amp = amp_scale * 0.4 / (1 + (t / 30.0) ** 2) ** 0.5 * np.exp(-np.clip(t, 0, None) / 15)
    orbital_phase = -np.sign(t) * 2.0 * np.abs(t) ** 0.6
    modes = {
        (2, 2): amp * np.exp(-2j * orbital_phase),
        (2, 1): 0.1 * amp * np.exp(-1j * orbital_phase),
        (3, 3): 0.15 * amp * np.exp(-3j * orbital_phase),
    }
    return core.complete_negative_m(modes)


def test_identical_waveforms_give_zero_error():
    t = np.arange(-1500, 150, 0.1)
    h = fake_waveform(t)
    res = core.align_and_compare(t, h, t, h, window=(-1000, 100))
    assert res["E"] < 1e-8, res  # interpolation noise only


def test_recovers_time_shift_and_rotation():
    t = np.arange(-1500, 150, 0.1)
    h_ref = fake_waveform(t)
    # Same waveform, but on a shifted clock and rotated by phi = 0.7 rad.
    shift, phi = 37.3, 0.7
    h_test = core.rotate(fake_waveform(t - shift), phi)
    res = core.align_and_compare(t, h_ref, t, h_test, window=(-1000, 100))
    assert res["E"] < 1e-8, res


def test_amplitude_error_matches_formula():
    # A 1% amplitude offset should give E = 0.5 * 0.01^2 = 5e-5.
    t = np.arange(-1500, 150, 0.1)
    h_ref = fake_waveform(t)
    h_test = fake_waveform(t, amp_scale=1.01)
    res = core.align_and_compare(t, h_ref, t, h_test, window=(-1000, 100))
    assert abs(res["E"] - 5e-5) < 1e-6, res


def test_recovers_overall_sign_flip():
    t = np.arange(-1500, 150, 0.1)
    h_ref = fake_waveform(t)
    h_test = {k: -h for k, h in core.rotate(fake_waveform(t), 0.4).items()}
    res = core.align_and_compare(t, h_ref, t, h_test, window=(-1000, 100))
    assert res["E"] < 1e-8 and res["sign"] == -1, res


def test_peak_time_is_subsample_accurate():
    # Smooth peak (real merger peaks are smooth); true peak at t = 0.2,
    # between samples.
    t = np.arange(-100, 100, 0.5)
    amp = np.exp(-((t - 0.2) / 10.0) ** 2)
    h = {(2, 2): amp * np.exp(-0.3j * t)}
    assert abs(core.peak_time(t, h) - 0.2) < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"passed: {name}")
