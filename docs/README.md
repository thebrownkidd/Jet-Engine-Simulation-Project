# Health-Manifold Theory & Skeptical Experiments

This folder documents — at the level of a skeptical reviewer — the claims made
about the physics-constrained **health manifold**, with formal theorems,
proofs, and falsifiable experiments backed by plots. The theory and the
per-experiment writeups use **FD001** as the running example; the method is
validated across **all four** C-MAPSS datasets in the
[cross-dataset section](#cross-dataset-results-fd001fd004) below and, in full,
in the research-paper [root README](../README.md).

The headline questions and their answers:

| Question | Document | Verdict |
|---|---|---|
| Is the rollout actually **stable**? | [ROLLOUT_STABILITY.md](ROLLOUT_STABILITY.md) | **Yes** — provably bounded (free-run), unlike the VAR ($\rho=1.02$). |
| Can the **health state** be forecast? | [HEALTH_FORECASTING.md](HEALTH_FORECASTING.md) | **Yes** — beats persistence by skill $+0.66$–$0.77$; polynomial error. |
| Does it predict **RUL**? | [RUL_PREDICTION.md](RUL_PREDICTION.md) | **Yes** — FD001 RMSE $14.5$, $R^2=0.88$ on the official test set (all four datasets beat baseline). |
| What is the **math** behind all this? | [THEORY.md](THEORY.md) | 5 theorems + proofs, each tied to an experiment. |

---

## The logical chain

```mermaid
flowchart LR
    ID["Identifiable<br/>2-D health<br/>(Lemma 1, rho2=0.90)"]
    ST["Stable rollout<br/>(Exp A: bounded vs VAR rho=1.02)"]
    FC["Forecastable<br/>(Exp B: skill > 0, error ~ k^2)"]
    RUL["RUL prediction<br/>(Exp C: RMSE 13.7, R2 0.89)"]
    ID --> ST --> FC --> RUL
```

1. **Identifiable** — a single operating condition collapses the degradation to
   a 2-D manifold (PCA $\rho_2 = 0.901$); the autoencoder recovers it
   (mean test $R^2 = 0.969$).
2. **Stable** — the logistic latent + Lipschitz decoder make the rollout
   *bounded for all horizons* (Theorem 2), whereas a sensor-space VAR is
   non-contractive ($\rho(A)=1.02$) and diverges (Theorem 1). The free-run
   divergence plot is the decisive evidence.
3. **Forecastable** — because the health coordinate is smooth/monotone (small
   curvature $\kappa$), constant-velocity extrapolation has polynomial error
   $\le \tfrac{\kappa}{2}k^2$ (Theorem 3) and beats persistence.
4. **Useful** — the forecastable state (+ its velocity) predicts RUL with
   RMSE 13.7 on the official FD001 test set (Proposition 5).

---

## Reproducing the experiments

```powershell
# from the project root, using the project venv
cd experiments
..\.venv\Scripts\python.exe exp_rollout_stability.py    # Exp A  -> figures A1..A4
..\.venv\Scripts\python.exe exp_health_forecasting.py   # Exp B  -> figures B1..B3
..\.venv\Scripts\python.exe exp_rul_prediction.py       # Exp C  -> figures C1..C3
cd ..
```

All three scripts share [`../experiments/manifold_core.py`](../experiments/manifold_core.py),
which trains the $k=2$ manifold **once** and caches it to
`experiments/artifacts/` so every experiment uses an identical encoder,
decoder, standardization, and denoising convention. Figures are written to
[`figures/`](figures/); numeric artifacts to `experiments/artifacts/`.

---

## Figure index

Figures are now stored **per dataset** under `figures/FD00<fd>/`. The table
below lists the FD001 set; the identical set exists for FD002, FD003, FD004.

| File (under `figures/FD001/`) | Shows |
|---|---|
| `D1_health_trajectories.png` | Discovered monotone health trajectories |
| `D2_manifold.png` | PCA scree + 2-D health scatter colored by life fraction |
| `A1_rollout_r2_vs_horizon.png` | Rollout NRMSE / cross-engine $R^2$ vs horizon |
| `A2_var_eigenvalues.png` | VAR eigenvalues outside the unit circle ($\rho=1.02$) |
| `A3_example_trajectories.png` | True vs manifold vs VAR for one engine |
| `A4_free_run_divergence.png` | **Decisive**: VAR diverges, manifold bounded |
| `B1_health_forecasts.png` | Example health forecasts from 50 % life |
| `B2_error_vs_horizon.png` | Forecast error under the $\tfrac{\kappa}{2}k^2$ envelope |
| `B3_skill_vs_persistence.png` | Forecast skill vs persistence (h0, h1) |
| `C1_rul_scatter.png` | Predicted vs true RUL + error histogram |
| `C2_examples.png` | Example test engines: causal health + RUL readout |
| `C3_health_vs_rul.png` | Monotone health → RUL relationship |
| `figures/SUMMARY_cross_dataset.png` | 4-panel cross-dataset comparison (FD001–FD004) |

---

## Cross-dataset results (FD001–FD004)

The FD001 method generalizes to the multi-regime / multi-fault datasets via an
**operating-condition normalization** (KMeans regime clustering + per-regime
mean removal + within-regime standardization). The single driver
[`../experiments/run_all.py`](../experiments/run_all.py) reproduces the grid
(`../experiments/artifacts/cross_dataset_results.csv`).

![Cross-dataset summary](figures/SUMMARY_cross_dataset.png)

| Metric | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| Operating regimes | 1 | 6 | 1 | 6 |
| Fault modes | 1 | 1 | 2 | 2 |
| Train / test units | 100 / 100 | 260 / 259 | 100 / 100 | 249 / 248 |
| Discovery PCA $\rho_2$ | 0.965 | 0.795 | 0.851 | 0.849 |
| Recon mean $R^2$ | 0.930 | 0.865 | 0.960 | 0.930 |
| VAR $\rho(A)$ | 1.020 | 1.016 | 1.018 | 1.015 |
| VAR free-run growth | $760\times$ | $468\times$ | $58\times$ | $25\times$ |
| Manifold bounded? | ✓ | ✓ | ✓ | ✓ |
| Forecast skill ($k{=}20$) | $+0.751$ | $+0.680$ | $+0.522$ | $+0.157$ |
| **RUL RMSE** | **14.53** | **27.02** | **16.31** | **27.58** |
| **RUL $R^2$** | **0.878** | **0.748** | **0.845** | **0.744** |
| Mean baseline RMSE | 43.07 | 54.08 | 45.07 | 54.90 |

**Takeaways:** health→RUL beats the mean baseline on all four datasets; the
VAR free-runs to $25$–$760\times$ ($\rho(A)>1$) everywhere while the manifold
rollout stays bounded everywhere; difficulty tracks the
regime$\times$fault grid (easy corner FD001 → hard corner FD004).
