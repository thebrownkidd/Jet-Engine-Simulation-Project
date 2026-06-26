"""Generate the cross-dataset summary figure from cross_dataset_results.csv."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "results", "tables", "cross_dataset_results.csv"))
out = os.path.join(ROOT, "results", "figures")
os.makedirs(out, exist_ok=True)

ds = df["dataset"].tolist()
x = np.arange(len(ds))
col = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# (1) RUL R2 vs baseline
ax[0, 0].bar(x, df["rul_r2"], color=col, alpha=0.9)
ax[0, 0].axhline(0, color="k", lw=0.8)
for i, v in enumerate(df["rul_r2"]):
    ax[0, 0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(ds)
ax[0, 0].set_ylabel("test RUL  $R^2$")
ax[0, 0].set_title("(a) RUL prediction on official test set")
ax[0, 0].set_ylim(0, 1)
ax[0, 0].grid(alpha=0.3, axis="y")

# (2) RUL RMSE vs baseline
w = 0.38
ax[0, 1].bar(x - w / 2, df["rul_rmse"], w, label="health->RUL", color="#1b7837")
ax[0, 1].bar(x + w / 2, df["base_rmse"], w, label="mean baseline", color="#b2182b", alpha=0.8)
ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(ds)
ax[0, 1].set_ylabel("RUL RMSE (cycles)")
ax[0, 1].set_title("(b) RMSE vs mean-RUL baseline")
ax[0, 1].legend(fontsize=9)
ax[0, 1].grid(alpha=0.3, axis="y")

# (3) reconstruction R2 + PCA rho2
ax[1, 0].bar(x - w / 2, df["recon_mean_r2"], w, label="AE recon mean $R^2$", color="#4c72b0")
ax[1, 0].bar(x + w / 2, df["pca_rho2"], w, label=r"PCA $\rho_2$ (2-D var.)", color="#dd8452")
ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(ds)
ax[1, 0].set_ylabel("fraction")
ax[1, 0].set_title("(c) 2-D health manifold identifiability")
ax[1, 0].set_ylim(0, 1)
ax[1, 0].legend(fontsize=9)
ax[1, 0].grid(alpha=0.3, axis="y")

# (4) VAR free-run growth (log) -> stability
ax[1, 1].bar(x, df["var_freerun_growth"], color="#b2182b", alpha=0.85, label="VAR free-run growth")
ax[1, 1].axhline(1.0, color="#1b7837", lw=2, label="manifold (bounded ~1x)")
ax[1, 1].set_yscale("log")
ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(ds)
ax[1, 1].set_ylabel(r"$\|z_{400}\| / \|z_0\|$  (log)")
ax[1, 1].set_title("(d) Free-run divergence: VAR explodes, manifold bounded")
for i, v in enumerate(df["var_freerun_growth"]):
    ax[1, 1].text(i, v * 1.2, f"{v:.0f}x", ha="center", fontsize=9)
ax[1, 1].legend(fontsize=9)
ax[1, 1].grid(alpha=0.3, axis="y")

fig.suptitle("Physics-constrained health manifold across C-MAPSS FD001-FD004",
             fontsize=14, y=1.0)
fig.tight_layout()
p = os.path.join(out, "SUMMARY_cross_dataset.png")
fig.savefig(p, dpi=140, bbox_inches="tight")
print("saved", p)
