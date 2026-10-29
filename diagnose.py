"""Per-mode error breakdown for one simulation: which modes dominate E?"""
import sys
import numpy as np
from surrogate_validation import core, data

sim_id = sys.argv[1] if len(sys.argv) > 1 else "SXS:BBH:0554"
name = "NRHybSur3dq8"
t_nr, h_nr, p = data.load_nr(sim_id, ell_max=4)
sur = data.load_surrogate(name)
f_low = 0.9 * data.gw_frequency_at(t_nr, h_nr, p["t_ref"])
t_sur, h_sur = data.eval_surrogate(sur, name, p["q"], p["chi1"], p["chi2"], f_low=f_low)
h_sur = core.complete_negative_m(h_sur)
keys = sorted(k for k in set(h_nr) & set(h_sur) if k[0] <= 4 and k[1] != 0)

t_start = p["t_ref"] - core.peak_time(t_nr, h_nr)
res = core.align_and_compare(t_nr, h_nr, t_sur, h_sur, window=(t_start, 100), keys=keys)
print(f"{sim_id}  q={p['q']:.2f}  E={res['E']:.2e}  E22={res['E22']:.2e}  sign={res['sign']:+d}\n")

tr = t_nr - core.peak_time(t_nr, {k: h_nr[k] for k in keys})
ts = t_sur - core.peak_time(t_sur, {k: h_sur[k] for k in keys})
t = np.arange(res["t_start"], res["t_end"], 0.1)
ref = core.resample(tr, {k: h_nr[k] for k in keys}, t)
test = core.rotate(core.resample(ts, {k: res["sign"] * h_sur[k] for k in keys}, t + res["tau"]), res["phi"])
den = sum(np.trapezoid(np.abs(ref[k]) ** 2, t) for k in keys)

rows = []
for k in keys:
    err = 0.5 * np.trapezoid(np.abs(ref[k] - test[k]) ** 2, t) / den
    power = np.trapezoid(np.abs(ref[k]) ** 2, t) / den
    rows.append((err, k, power))
print(f"{'mode':>8} {'share of E':>11} {'error':>10} {'power':>10}")
for err, k, power in sorted(rows, reverse=True):
    print(f"{str(k):>8} {100*err/res['E']:10.1f}% {err:10.2e} {power:10.2e}")
