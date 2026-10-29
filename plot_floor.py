import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

r = pd.read_csv("results/resolution.csv").dropna(subset=["E_res", "E_sur"])
s = pd.read_csv("results/NRHybSur3dq8.csv").set_index("sim_id")
r["q"] = r.sim_id.map(s["q"]); r["chi_eff"] = r.sim_id.map(s["chi_eff"])
n = r.sim_id.str.split(":").str[-1].astype(int)
r["trained_on"] = n.between(1419, 1509)

fig, ax = plt.subplots(figsize=(6, 5))
for mask, marker, label in ((r.trained_on, "o", "trained on"), (~r.trained_on, "s", "unseen")):
    sc = ax.scatter(r.E_res[mask], r.E_sur[mask], c=r.q[mask], cmap="plasma", vmin=1, vmax=10,
                    marker=marker, s=60, edgecolor="k", linewidth=0.4, label=label)
lo, hi = 1e-7, 1e-1
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="surrogate error = NR error")
ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
       xlabel="NR resolution error $\\mathcal{E}_{\\rm res}$ (Lev N vs N-1)",
       ylabel="surrogate error $\\mathcal{E}$")
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.colorbar(sc, ax=ax, label="mass ratio $q$")
fig.tight_layout(); fig.savefig("results/error_floor.png", dpi=200)
print(f"{len(r)} simulations. Above the line (genuine surrogate failures):")
print(r[r.E_sur > r.E_res][["sim_id", "q", "chi_eff", "E_res", "E_sur", "ratio"]].sort_values("ratio", ascending=False).to_string(index=False))
print("\nFailed to load:", pd.read_csv("results/resolution.csv").error.notna().sum())
