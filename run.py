"""Compare a surrogate model against SXS numerical-relativity waveforms.

Examples
--------
  # 30 non-precessing, quasi-circular simulations spread across mass ratio
  python run.py --auto 30 --q-max 12

  # specific simulations
  python run.py --sims SXS:BBH:0001 SXS:BBH:0002

Results are appended to a CSV (default results/NRHybSur3dq8.csv) one row at
a time, so an interrupted run loses nothing; rerunning skips finished IDs.
"""

import argparse
import csv
import os
import sys
import traceback

import numpy as np
import pandas as pd

from surrogate_validation import core, data

FIELDS = ["sim_id", "q", "chi1z", "chi2z", "chi_eff", "chi1_mag", "chi2_mag",
          "ecc", "in_training_range", "E", "E22", "tau", "phi", "sign",
          "t_start", "t_end", "n_modes", "error"]


def find_column(df, *candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def select_nonprecessing(n, q_min, q_max, max_ecc, max_chi_perp=1e-3):
    """Pick n non-precessing, quasi-circular BBH simulations, spread
    evenly in mass ratio."""
    df = data.catalog_dataframe()
    cols = {
        "q": find_column(df, "reference_mass_ratio"),
        "chi1_perp": find_column(df, "reference_chi1_perp"),
        "chi2_perp": find_column(df, "reference_chi2_perp"),
        "ecc": find_column(df, "reference_eccentricity"),
        "objects": find_column(df, "object_types"),
        "deprecated": find_column(df, "deprecated"),
    }
    missing = [k for k in ("q", "chi1_perp", "chi2_perp") if cols[k] is None]
    if missing:
        sys.exit(f"Catalog columns not found for {missing}. Available columns:\n"
                 f"{list(df.columns)}\nEdit find_column(...) in run.py, or pass --sims.")

    keep = pd.Series(True, index=df.index)
    if cols["objects"]:
        keep &= df[cols["objects"]].astype(str).str.upper().eq("BHBH")
    if cols["deprecated"]:
        keep &= ~df[cols["deprecated"]].fillna(False).astype(bool)
    q = pd.to_numeric(df[cols["q"]], errors="coerce")
    keep &= q.between(q_min, q_max)
    for c in ("chi1_perp", "chi2_perp"):
        keep &= pd.to_numeric(df[cols[c]], errors="coerce").abs() < max_chi_perp
    if cols["ecc"]:
        ecc = pd.to_numeric(df[cols["ecc"]], errors="coerce")
        keep &= ecc < max_ecc  # unknown (NaN) eccentricity is excluded

    pool = q[keep].sort_values()
    if pool.empty:
        sys.exit("No simulations matched the selection.")
    picks = np.unique(np.linspace(0, len(pool) - 1, min(n, len(pool))).round().astype(int))
    ids = [str(i) for i in pool.index[picks]]
    print(f"Selected {len(ids)} of {len(pool)} matching simulations "
          f"(q from {pool.iloc[0]:.2f} to {pool.iloc[-1]:.2f}).")
    return ids


def compare_one(sim_id, sur, sur_name, ell_max, window_end, include_m0=False):
    t_nr, h_nr, p = data.load_nr(sim_id, ell_max=ell_max)
    q, chi1, chi2 = p["q"], p["chi1"], p["chi2"]
    row = {"sim_id": sim_id, "q": q, "chi1z": chi1[2], "chi2z": chi2[2],
           "chi_eff": (q * chi1[2] + chi2[2]) / (1 + q),
           "chi1_mag": np.linalg.norm(chi1), "chi2_mag": np.linalg.norm(chi2),
           "ecc": p["ecc"],
           "in_training_range": data.in_training_range(sur_name, q, chi1, chi2)}

    f_low = 0.9 * data.gw_frequency_at(t_nr, h_nr, p["t_ref"])
    t_sur, h_sur = data.eval_surrogate(sur, sur_name, q, chi1, chi2, f_low=f_low)
    if not data.TRAINING_RANGE[sur_name]["precessing"]:
        h_sur = core.complete_negative_m(h_sur)

    # Start the comparison at the NR reference time (after junk radiation).
    t_start = p["t_ref"] - core.peak_time(t_nr, h_nr)
    # m = 0 modes carry memory, which the surrogate does not model: skip by default.
    keys = sorted(k for k in set(h_nr) & set(h_sur)
                  if k[0] <= ell_max and (include_m0 or k[1] != 0))
    row.update(core.align_and_compare(t_nr, h_nr, t_sur, h_sur,
                                      window=(t_start, window_end), keys=keys))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surrogate", default="NRHybSur3dq8", choices=sorted(data.TRAINING_RANGE))
    ap.add_argument("--sims", nargs="*", help="SXS IDs, e.g. SXS:BBH:0305")
    ap.add_argument("--auto", type=int, help="pick N non-precessing simulations automatically")
    ap.add_argument("--q-min", type=float, default=1.0)
    ap.add_argument("--q-max", type=float, default=12.0)
    ap.add_argument("--max-ecc", type=float, default=1e-3)
    ap.add_argument("--ell-max", type=int, default=4)
    ap.add_argument("--window-end", type=float, default=100.0,
                    help="end of comparison window, in M after peak")
    ap.add_argument("--include-m0", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if data.TRAINING_RANGE[args.surrogate]["precessing"]:
        sys.exit("Precessing surrogates need frame alignment (stage 2); not implemented yet.")

    ids = list(args.sims or [])
    if args.auto:
        ids += select_nonprecessing(args.auto, args.q_min, args.q_max, args.max_ecc)
    if not ids:
        sys.exit("Give --sims or --auto N.")

    out = args.out or os.path.join("results", f"{args.surrogate}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) == 0:
        os.remove(out)
    done = set(pd.read_csv(out)["sim_id"]) if os.path.exists(out) else set()
    new_file = not os.path.exists(out)

    sur = data.load_surrogate(args.surrogate)
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, sim_id in enumerate(ids, 1):
            if sim_id in done:
                continue
            print(f"[{i}/{len(ids)}] {sim_id} ... ", end="", flush=True)
            try:
                row = compare_one(sim_id, sur, args.surrogate, args.ell_max, args.window_end, args.include_m0)
                print(f"q={row['q']:.2f}  chi_eff={row['chi_eff']:+.2f}  E={row['E']:.2e}")
            except Exception as exc:
                row = {"sim_id": sim_id, "error": f"{type(exc).__name__}: {exc}"}
                print("FAILED:", row["error"])
                traceback.print_exc(limit=2)
            writer.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
