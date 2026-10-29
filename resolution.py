"""NR error floor: compare each simulation's two highest resolutions."""
import os, sys
import numpy as np
import pandas as pd
from surrogate_validation import core

OUT = "results/resolution.csv"
RESULTS = "results/NRHybSur3dq8.csv"
FIELDS = ["sim_id", "lev_hi", "lev_lo", "E_res", "E_sur", "ratio", "error"]


def modes_of(h, ell_max=4):
    t = np.asarray(h.t)
    modes = {}
    for ell in range(2, ell_max + 1):
        for m in range(-ell, ell + 1):
            if m != 0:
                modes[(ell, m)] = np.asarray(h[:, h.index(ell, m)])
    return t, modes


def two_highest_levels(sim_id):
    import sxs
    sim = sxs.load(sim_id)
    levs = sorted(getattr(sim, "lev_numbers", []) or [])
    if len(levs) < 2:
        raise RuntimeError(f"only resolutions {levs} available")
    hi, lo = levs[-1], levs[-2]
    sim_hi = sxs.load(f"{sim_id}/Lev{hi}")
    sim_lo = sxs.load(f"{sim_id}/Lev{lo}")
    t_ref = float(sim.metadata.get("reference_time", sim_hi.h.t[0] + 300.0))
    return hi, lo, sim_hi.h, sim_lo.h, t_ref


def main():
    args = sys.argv[1:]
    if args and args[0] == "--from-results":
        n = int(args[1]) if len(args) > 1 else 30
        d = pd.read_csv(RESULTS).dropna(subset=["E"]).sort_values("E")
        idx = np.unique(np.linspace(0, len(d) - 1, min(n, len(d))).round().astype(int))
        ids = list(d.iloc[idx]["sim_id"])
    else:
        ids = args or ["SXS:BBH:2786", "SXS:BBH:2680", "SXS:BBH:1432", "SXS:BBH:3144"]
    sur = pd.read_csv(RESULTS).set_index("sim_id")["E"]
    done = set(pd.read_csv(OUT)["sim_id"]) if os.path.exists(OUT) and os.path.getsize(OUT) else set()
    for i, sim_id in enumerate(ids, 1):
        if sim_id in done:
            continue
        print(f"[{i}/{len(ids)}] {sim_id} ... ", end="", flush=True)
        try:
            hi, lo, h_hi, h_lo, t_ref = two_highest_levels(sim_id)
            t1, m1 = modes_of(h_hi)
            t2, m2 = modes_of(h_lo)
            t_start = t_ref - core.peak_time(t1, m1)
            res = core.align_and_compare(t1, m1, t2, m2, window=(t_start, 100.0))
            E_sur = float(sur.get(sim_id, np.nan))
            row = {"sim_id": sim_id, "lev_hi": hi, "lev_lo": lo, "E_res": res["E"],
                   "E_sur": E_sur, "ratio": E_sur / res["E"]}
            print(f"Lev{hi} vs Lev{lo}: E_res={res['E']:.2e}  E_sur={E_sur:.2e}  ratio={row['ratio']:.1f}")
        except Exception as exc:
            row = {"sim_id": sim_id, "error": f"{type(exc).__name__}: {exc}"}
            print("FAILED:", row["error"])
        pd.DataFrame([row]).reindex(columns=FIELDS).to_csv(
            OUT, mode="a", index=False, header=not (os.path.exists(OUT) and os.path.getsize(OUT)))
    print(f"Results in {OUT}")


if __name__ == "__main__":
    main()
