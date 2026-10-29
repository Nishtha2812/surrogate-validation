"""Compare surrogate error on its own training simulations vs unseen ones.
NRHybSur3dq8 was trained on SXS:BBH:1419-1509 (Varma et al. 2019, PRD 99, 064045)."""
import pandas as pd
from scipy.stats import mannwhitneyu

d = pd.read_csv("results/NRHybSur3dq8.csv").dropna(subset=["E"])
n = d.sim_id.str.split(":").str[-1].astype(int)
d["trained_on"] = n.between(1419, 1509)
inside = d.in_training_range.astype(str).eq("True")

print("All simulations:")
print(d.groupby("trained_on").E.describe()[["count", "50%", "max"]], "\n")
print("Only simulations INSIDE the training range (fair comparison):")
print(d[inside].groupby("trained_on").E.describe()[["count", "50%", "max"]], "\n")

a, b = d.E[inside & ~d.trained_on], d.E[inside & d.trained_on]
print("Unseen vs trained-on, inside range:", mannwhitneyu(a, b, alternative="greater"), "\n")
print("Unseen simulations inside the range, worst first:")
print(d[inside & ~d.trained_on].sort_values("E", ascending=False)[["sim_id", "q", "chi_eff", "E"]].head(10).to_string(index=False))
d.to_csv("results/NRHybSur3dq8_labeled.csv", index=False)
