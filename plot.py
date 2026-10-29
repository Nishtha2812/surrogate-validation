"""Plot surrogate error against mass ratio and effective spin.

  python plot.py results/NRHybSur3dq8.csv
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from surrogate_validation.data import TRAINING_RANGE


def main(path, surrogate="NRHybSur3dq8"):
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df["E"], errors="coerce").notna()].copy()
    df["E"] = df["E"].astype(float)
    inside = df["in_training_range"].astype(str).str.lower().eq("true")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    vmax = max(0.8, df["chi_eff"].abs().max())

    ax = axes[0]
    for mask, marker, label in ((inside, "o", "inside training range"),
                                (~inside, "X", "outside (extrapolation)")):
        sc = ax.scatter(df.loc[mask, "q"], df.loc[mask, "E"], c=df.loc[mask, "chi_eff"],
                        cmap="coolwarm", vmin=-vmax, vmax=vmax, marker=marker,
                        s=45, edgecolor="k", linewidth=0.4, label=label)
    ax.axvline(TRAINING_RANGE[surrogate]["q_max"], ls="--", c="gray", lw=1)
    ax.set_xlabel("mass ratio $q = m_1/m_2$")
    ax.set_ylabel("time-domain error $\\mathcal{E}$")
    ax.set_yscale("log")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="inside training range"),
        Line2D([], [], marker="X", ls="", mfc="white", mec="k", label="outside (extrapolation)"),
    ], frameon=False, fontsize=9)
    fig.colorbar(sc, ax=ax, label="$\\chi_{\\rm eff}$")

    ax = axes[1]
    ax.scatter(df["chi_eff"], df["E"], c=np.where(inside, "C0", "C3"),
               s=40, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("effective spin $\\chi_{\\rm eff}$")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="C0", mec="k", label="inside training range"),
        Line2D([], [], marker="o", ls="", mfc="C3", mec="k", label="outside (extrapolation)"),
    ], frameon=False, fontsize=9)

    fig.suptitle(f"{surrogate} vs. SXS numerical relativity ({len(df)} simulations)")
    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")
    print(f"Median E inside training range:  {df.loc[inside, 'E'].median():.2e}")
    if (~inside).any():
        print(f"Median E outside training range: {df.loc[~inside, 'E'].median():.2e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/NRHybSur3dq8.csv")
