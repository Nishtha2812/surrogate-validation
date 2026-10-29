import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "results/NRHybSur3dq8.csv"
df = pd.read_csv(path)
df = df[pd.to_numeric(df["E"], errors="coerce").notna()].copy()
df["E"] = df["E"].astype(float)
inside = df["in_training_range"].astype(str).str.lower().eq("true")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for mask, marker in ((inside, "o"), (~inside, "X")):
    sc = ax.scatter(df.loc[mask, "q"], df.loc[mask, "chi_eff"], c=np.log10(df.loc[mask, "E"]),
                    cmap="viridis", vmin=np.log10(df["E"].min()), vmax=np.log10(df["E"].max()),
                    marker=marker, s=70, edgecolor="k", linewidth=0.4)
ax.axvline(8, ls="--", c="gray", lw=1)
ax.set_xlabel("mass ratio $q$")
ax.set_ylabel("effective spin $\\chi_{\\rm eff}$")
ax.set_title("NRHybSur3dq8 error (o inside, x outside training range)", fontsize=10)
fig.colorbar(sc, ax=ax, label="$\\log_{10}\\,\\mathcal{E}$")
fig.tight_layout()
out = path.rsplit(".", 1)[0] + "_map.png"
fig.savefig(out, dpi=200)
print(f"Saved {out}")
