# Results Section — Artifact Guide for an Algorithmic Study

This guide maps every usable artifact in the repository to a concrete place in a
Results section, and states **exactly what each artifact contains/shows**. It
also proposes the consolidated table you described (final RUL + boundedness +
ablation + baselines) and explains where a single mega-table helps vs. where to
split for readability.

All numbers below are quoted from the saved CSVs (production run at 4000 epochs;
ACML sweeps at 1500 epochs — orderings identical). Sources are linked per row.

---

## 1. Recommended Results section structure

A representation-learning / algorithmic study reads best as a sequence of
falsifiable claims, each with one table or figure. Proposed layout:

| § | Title | Core question | Primary table | Primary figure |
|---|-------|---------------|---------------|----------------|
| 5.1 | Datasets & protocol | what/how evaluated | (text) Table P | — |
| 5.2 | Latent manifold quality | does a compact latent emerge? | intrinsic-dim | `D2_manifold` (2×2) |
| 5.3 | Main cross-dataset result | does the final model work? | **Table 1 (main)** | `SUMMARY_cross_dataset` |
| 5.4 | Rollout stability vs baselines | is it stable vs VAR/VAR2/LSTM/GRU/TCN? | **Table 2 (baselines)** | `fig_baselines` |
| 5.5 | Latent dimension study | bottleneck or regularizer? | **Table 3 (K-sweep)** | `fig_k_tradeoff` |
| 5.6 | Ablation / mechanism | what causes boundedness vs forecastability? | **Table 4 (ablation)** | `boundedness_mechanism` |
| 5.7 | Seed robustness | robust to init? | **Table 5 (seeds)** | `seed_robustness_summary` |
| 5.8 | Transfer | specialist vs generalized? | **Table 6 (transfer)** | `specialist_vs_generalized` |

Put §5.4–5.8 under one banner "Comparative study", and §5.3 as "Final model".

---

## 2. The consolidated table you imagined

You asked for one table combining **final-model RUL + boundedness + ablation +
baselines**. That is three different row-types (datasets, variants, models), so a
single literal table is wide. Two good options:

### Option A — One "headline" table (recommended for the main text)

A compact table that fuses the **final model** with the **other-model
comparison**, with boundedness and RUL side by side. One row per (dataset,
model); the proposed method is one of the models.

| Dataset | Model | ρ(VAR) | Free-run growth | Bounded | NRMSE@50 | RUL RMSE | RUL R² |
|---|---|---:|---:|:--:|---:|---:|---:|
| FD001 | **Manifold (ours)** | — | **1.5×** | **✓** | 0.467 | **14.53** | **0.878** |
| FD001 | VAR | 1.020 | 760× | ✗ | 0.424 | — | — |
| FD001 | VAR2 | 1.020 | 748× | ✗ | 0.425 | — | — |
| FD001 | LSTM | — | 2.9× | ✓ | 0.853 | — | — |
| FD001 | GRU | — | 2.8× | ✓ | 0.907 | — | — |
| FD001 | TCN | — | 3.2× | ✓ | 0.476 | — | — |
| FD002 | **Manifold (ours)** | — | **1.5×** | **✓** | 1.453 | **27.02** | **0.748** |
| FD002 | VAR | 1.016 | 468× | ✗ | 0.510 | — | — |
| FD002 | VAR2 | 1.016 | 476× | ✗ | 0.513 | — | — |
| FD002 | LSTM | — | 4.7× | ✓ | 2.125 | — | — |
| FD002 | GRU | — | 13.0× | ✗ | 0.875 | — | — |
| FD002 | TCN | — | 1.27e6× | ✗ | 1.376 | — | — |
| FD003 | **Manifold (ours)** | — | **1.3×** | **✓** | 0.312 | **16.31** | **0.845** |
| FD003 | VAR / VAR2 | 1.018 | 58× / 57× | ✗ | 0.381 | — | — |
| FD003 | LSTM / GRU / TCN | — | 0.9× / 1.5× / 2.8× | ✓ | 0.531 / 1.173 / 0.384 | — | — |
| FD004 | **Manifold (ours)** | — | **0.8×** | **✓** | 0.561 | **27.58** | **0.744** |
| FD004 | VAR / VAR2 | 1.015 | 25× / 27× | ✗ | 0.385 | — | — |
| FD004 | LSTM / GRU / TCN | — | 3.4× / 1.9× / 1.4× | ✓ | 0.619 / 0.715 / 0.525 | — | — |

- RUL columns only populated for the proposed method (baselines are sensor-space
  rollouts, not RUL models) — that asymmetry is itself a point: only the bounded
  latent yields *both* stable rollout *and* a usable RUL state.
- Sources: free-run/NRMSE from
  [baselines_extended.csv](../results/acml/tables/baselines_extended.csv);
  RUL from [cross_dataset_results.csv](../results/tables/cross_dataset_results.csv).
- Ready-made LaTeX: combine
  [baselines_extended.tex](../results/acml/tables/baselines_extended.tex) +
  [tab_main_cross_dataset.tex](../results/acml/tables/tab_main_cross_dataset.tex).

### Option B — Keep ablation separate (recommended)

The ablation answers a *different* question (mechanism), so keep it as its own
table rather than forcing it into the headline. Use **Table 4** below. Trying to
cram dataset-rows, model-rows and variant-rows into one grid hurts readability
and reviewers dislike it.

**Recommendation:** Option A as the headline Table 1–2 fusion, and a separate
ablation Table 4. Do *not* merge all four into one literal table.

---

## 3. Every results table — location and exact contents

### Table 1 (main, final model) — production cross-dataset
- File: [results/tables/cross_dataset_results.csv](../results/tables/cross_dataset_results.csv)
  (+ `.json` mirror), LaTeX skeleton
  [tab_main_cross_dataset.tex](../results/acml/tables/tab_main_cross_dataset.tex).
- One row per dataset. Columns include: `n_regimes`, `pca_rho2`,
  `recon_mean_r2`, `rho_var`, `var_freerun_growth`, `man_freerun_norm`,
  `kappa`, `skill_cv_k20`, `rul_rmse`, `rul_mae`, `rul_r2`, `rul_nasa`,
  `base_rmse`, `base_nasa`.
- Shows: the final bounded k=2 model beats the mean-RUL baseline on every
  dataset (RUL RMSE 14.53/27.02/16.31/27.58 vs base 43.07/54.08/45.07/54.90),
  VAR diverges (ρ>1, 25–760×), manifold bounded (norm ≈2–6).

### Table 2 (baselines) — rollout comparison incl. TCN
- File: [results/acml/tables/baselines_extended.csv](../results/acml/tables/baselines_extended.csv)
  + [.tex](../results/acml/tables/baselines_extended.tex).
- 24 rows = 4 datasets × {manifold, var1, var2, lstm, gru, tcn}. Columns:
  `rho`, `freerun_norm`, `freerun_growth`, `bounded`, `nrmse_h1/h10/h25/h50`.
- Shows: manifold bounded on all 4 datasets; VAR/VAR2 diverge everywhere
  (ρ>1); neural baselines inconsistent — **TCN diverges 1.27×10⁶ on FD002**,
  **GRU 13× on FD002**. Short-horizon NRMSE@1 favors VAR (0.08–0.18) — honest
  caveat that the claim is stability, not one-step accuracy.

### Table 3 (K-sweep) — capacity vs stability
- File: [results/acml/tables/k_aware_dim_sweep.csv](../results/acml/tables/k_aware_dim_sweep.csv)
  + [.tex](../results/acml/tables/k_aware_dim_sweep.tex).
- 24 rows = 4 datasets × k∈{1..6}. Columns: `recon_mean_r2`, `recon_min_r2`,
  `freerun_norm`, `freerun_growth`, `freerun_bounded`, `nrmse_h1/h10/h25/h50`,
  `kappa`, `rul_rmse`, `rul_r2`.
- Shows (the central claim): reconstruction rises with K (FD002 0.808→0.932)
  but **FD002 becomes unbounded at k≥3** (growth 5.2/8.6/5.8/5.8×,
  `bounded=False`), and RUL never improves over k=2. **All K coordinates are
  used downstream** (feature vector `[h0..h(K-1), v0..v(K-1)]`), fixing the
  earlier confound.

### Table 4 (ablation/mechanism) — what causes what
- Files:
  [ablation_extended.csv](../results/acml/tables/ablation_extended.csv) (7
  variants incl. unbounded AE; FD001+FD002) and
  [boundedness_mechanism.csv](../results/acml/tables/boundedness_mechanism.csv)
  (5 configs, adds the **latent-norm growth** metric). LaTeX:
  [ablation_extended.tex](../results/acml/tables/ablation_extended.tex),
  [boundedness_mechanism.tex](../results/acml/tables/boundedness_mechanism.tex).
- ablation columns: `bounded_latent`, `recon_mean_r2`, `recon_min_r2`,
  `mono_viol_frac`, `kappa`, `freerun_growth`, `freerun_bounded`,
  `nrmse_h*`, `cv_skill_k20`, `rul_rmse/mae/r2`, `base_rmse`.
- Shows: (i) **reconstruction ≠ prognosis** — FD002 `no_regime_norm` hits recon
  R²=0.999 but skill −0.13 and RUL 40.5; (ii) penalties cut curvature κ
  (FD002 0.0346 unbounded → 0.0019 full) and lift skill (+0.28→+0.68); (iii)
  every bounded variant stays bounded regardless of penalties.

### Table 5 (seed robustness)
- Files: [seed_robustness.csv](../results/acml/tables/seed_robustness.csv)
  (per-seed), [seed_robustness_summary.csv](../results/acml/tables/seed_robustness_summary.csv)
  (mean±std), [.tex](../results/acml/tables/seed_robustness.tex).
- Shows: bounded `full` model bounded in 100% of seeds; FD001 RUL 14.39±0.45,
  FD002 27.40±0.13; unbounded-AE skill collapses and is unstable
  (FD002 −0.73±0.82). Establishes conclusions are not seed artifacts.

### Table 6 (specialist vs generalized)
- File: [specialist_vs_generalized.csv](../results/acml/tables/specialist_vs_generalized.csv)
  + [.tex](../results/acml/tables/specialist_vs_generalized.tex).
- 8 rows = 4 datasets × {specialist, generalized}. Columns: `recon_mean_r2`,
  `freerun_growth`, `freerun_bounded`, `rul_rmse`, `rul_r2`.
- Shows: a single pooled (generalized) model preserves bounded rollout on 3/4
  datasets and near-identical RUL (FD002 26.92 vs 26.87) — geometry is reusable.

### Supporting tables (appendix / methods)
- [intrinsic_dim.csv](../results/tables/research/latent_dim/intrinsic_dim.csv):
  PCA ρ1/ρ2/ρ3 and #components for 90/95/99% var — justifies low intrinsic dim
  (FD001 ρ2=0.965 vs FD002 0.795). Use in §5.2.
- [recon_vs_k.csv](../results/tables/research/latent_dim/recon_vs_k.csv):
  reconstruction-only AE vs k (capacity probe, no penalties). Complements
  Table 3; appendix.
- [regime_mining.csv](../results/tables/research/latent_dim/regime_mining.csv):
  heuristic vs silhouette-mined regime counts (1/6/1/6 confirmed). Methods/appendix.
- [rollout_stability.csv](../results/tables/rollout_stability.csv): horizon-wise
  manifold vs VAR R²/NRMSE (FD001) — backs the §5.4 narrative.
- [health_forecasting.csv](../results/tables/health_forecasting.csv): forecast
  skill vs persistence per horizon/model — backs the forecastability claim.
- [rul_predictions.csv](../results/tables/rul_predictions.csv) +
  [rul_metrics.json](../results/tables/rul_metrics.json): per-engine predictions
  and the honest threshold-crossing failure (RMSE 49 vs robust 13.7). Appendix.
- [theory_impl_rollout_compare.csv](../results/acml/tables/theory_impl_rollout_compare.csv):
  legacy `h0_clip` vs theory-matched `full_box` projection — for the
  implementation note that Theorem 2 now holds verbatim.
- [ablation_summary.csv](../results/tables/research/ablation/ablation_summary.csv),
  [rollout_ablation_summary.csv](../results/tables/research/ablation/rollout_ablation_summary.csv),
  [baselines_summary.csv](../results/tables/research/rollout_baselines/baselines_summary.csv),
  [per_file_dim_summary.csv](../results/tables/research/per_file_dim/per_file_dim_summary.csv),
  [unified_per_file.csv](../results/tables/research/unified/unified_per_file.csv):
  earlier production versions; cite only if you want the 4000-epoch numbers.

---

## 4. Every results figure — location and exact contents

### Headline figures (main text)
- [fig_pipeline.png](../results/acml/figures/fig_pipeline.png): method schematic
  (regime ID → normalise → denoise → select → bounded AE → rollout/RUL). §5.1/Method.
- [SUMMARY_cross_dataset.png](../results/figures/SUMMARY_cross_dataset.png):
  4-panel final-model overview. §5.3.
- [fig_baselines.png](../results/acml/figures/fig_baselines.png): compact 2×2
  free-run growth bars per model incl. TCN; manifold below threshold, VAR/VAR2/TCN
  above. §5.4.
- [fig_k_tradeoff.png](../results/acml/figures/fig_k_tradeoff.png): (a) recon vs K,
  (b) RUL vs K — capacity up, prognosis flat. §5.5.
- [k_vs_reconstruction_and_stability.png](../results/acml/figures/k_vs_reconstruction_and_stability.png):
  recon↑ vs free-run growth↑ with K, bounded threshold marked. §5.5 (alt/expanded).
- [boundedness_mechanism.png](../results/acml/figures/boundedness_mechanism.png):
  latent-norm growth vs curvature κ across the 5 configs — separates boundedness
  from forecastability. §5.6.
- [seed_robustness_summary.png](../results/acml/figures/seed_robustness_summary.png):
  mean±std bars over seeds 0–4. §5.7.
- [specialist_vs_generalized.png](../results/acml/figures/specialist_vs_generalized.png):
  recon/growth/RUL specialist vs generalized. §5.8.

### Per-dataset mechanism figures (use the combined 2×2 panels in main text)
- [combined_fd/D2_manifold_FD001_FD004_combined.png](../results/figures/combined_fd/D2_manifold_FD001_FD004_combined.png):
  PCA scree + 2-D latent colored by life fraction, all 4 datasets. §5.2.
- [combined_fd/D1_health_trajectories_FD001_FD004_combined.png](../results/figures/combined_fd/D1_health_trajectories_FD001_FD004_combined.png):
  learned monotone health trajectories. §5.2.
- [combined_fd/A4_free_run_divergence_FD001_FD004_combined.png](../results/figures/combined_fd/A4_free_run_divergence_FD001_FD004_combined.png):
  **the decisive plot** — VAR explodes, manifold flat (log-scale ‖z‖). §5.4.
- [combined_fd/A2_var_eigenvalues_FD001_FD004_combined.png](../results/figures/combined_fd/A2_var_eigenvalues_FD001_FD004_combined.png):
  VAR eigenvalues outside unit circle (ρ>1). §5.4.
- [combined_fd/A1_rollout_r2_vs_horizon_FD001_FD004_combined.png](../results/figures/combined_fd/A1_rollout_r2_vs_horizon_FD001_FD004_combined.png):
  rollout NRMSE/R² vs horizon. §5.4.
- [combined_fd/A3_example_trajectories_FD001_FD004_combined.png](../results/figures/combined_fd/A3_example_trajectories_FD001_FD004_combined.png):
  true vs manifold vs VAR per sensor. Appendix.
- [combined_fd/B2_error_vs_horizon_FD001_FD004_combined.png](../results/figures/combined_fd/B2_error_vs_horizon_FD001_FD004_combined.png):
  forecast error under the (κ/2)t² envelope. §5.6 / forecastability.
- [combined_fd/B3_skill_vs_persistence_FD001_FD004_combined.png](../results/figures/combined_fd/B3_skill_vs_persistence_FD001_FD004_combined.png):
  forecast skill vs persistence. §5.6.
- [combined_fd/B1_health_forecasts_FD001_FD004_combined.png](../results/figures/combined_fd/B1_health_forecasts_FD001_FD004_combined.png):
  example health forecasts. Appendix.
- [combined_fd/C1_rul_scatter_FD001_FD004_combined.png](../results/figures/combined_fd/C1_rul_scatter_FD001_FD004_combined.png):
  predicted vs true RUL + error hist. §5.3.
- [combined_fd/C3_health_vs_rul_FD001_FD004_combined.png](../results/figures/combined_fd/C3_health_vs_rul_FD001_FD004_combined.png):
  monotone health→RUL relation. §5.3.
- [combined_fd/C2_examples_FD001_FD004_combined.png](../results/figures/combined_fd/C2_examples_FD001_FD004_combined.png):
  example test engines, causal health + RUL readout. Appendix.

### Per-dataset single figures (if you prefer FD001-only running example)
- [results/figures/FD001](../results/figures/FD001) … `FD004` each contain the
  full A1–A4, B1–B3, C1–C3, D1–D2 set described above for one dataset.

### Extra / appendix figures
- [k_vs_rul.png](../results/acml/figures/k_vs_rul.png): RUL RMSE vs K (per dataset).
- [ablation_extended_summary.png](../results/acml/figures/ablation_extended_summary.png):
  3-panel boundedness/curvature/RUL bars for all 7 variants.
- [fig_ablation_mechanism.png](../results/acml/figures/fig_ablation_mechanism.png):
  compact curvature + RUL by config.
- [baselines_extended_freerun.png](../results/acml/figures/baselines_extended_freerun.png)
  and per-dataset `baselines_extended_freerun_FD00x.png`: free-run traces incl. TCN.
- [research/latent_dim/recon_vs_k.png](../results/figures/research/latent_dim/recon_vs_k.png):
  reconstruction capacity + PCA scree (capacity probe).
- [research/rollout_baselines/freerun_FD00x.png](../results/figures/research/rollout_baselines):
  earlier (no-TCN) free-run baselines — superseded by `baselines_extended`.
- [research/ablation/rollout_ablation_FD001.png](../results/figures/research/ablation/rollout_ablation_FD001.png),
  `FD002`: earlier rollout ablation panels.

---

## 5. What else in the project belongs in Results (or near it)

- **Theory ↔ experiment cross-refs.** [docs/THEORY.md](THEORY.md) M.1–M.8 each
  pair a theorem with the experiment that tests it; the Results section should
  cite the matching figure/table per theorem (e.g. Theorem 1 ↔ A2/A4; Theorem 3
  ↔ B2/B3; Proposition 5 ↔ C3).
- **Implementation note.** [docs/theory_implementation_check.md](theory_implementation_check.md)
  + its CSV — one paragraph in Results/Method that Theorem 2 holds verbatim under
  full-box projection with immaterial metric change.
- **Honesty paragraph.** From [docs/ACML_READINESS_REPORT.md](ACML_READINESS_REPORT.md)
  §6: state plainly that the constant-velocity rollout keeps even the unbounded
  AE numerically bounded over finite horizons; the dramatic divergence is the
  closed-loop sensor-space/neural models. This pre-empts the obvious reviewer
  objection.
- **Datasets/protocol table (Table P).** Compose from
  [cross_dataset_results.csv](../results/tables/cross_dataset_results.csv)
  columns (`n_regimes`, train/test sizes from
  [docs/README.md](README.md)) — regimes 1/6/1/6, faults 1/1/2/2.
- **Reproducibility.** Cite the run logs
  [results/acml/log_*.txt](../results/acml) and the fixed-seed scripts under
  [experiments/acml](../experiments/acml) in an "Reproducibility" paragraph.
- **Blocked item.** Note Task 8 (external bearing dataset) as future work via
  [extra_dataset_BLOCKED.txt](../results/acml/tables/extra_dataset_BLOCKED.txt).

---

## 6. Minimal main-text set (if space is tight)

Tables: **Table 1** (main + baselines fused, Option A), **Table 3** (K-sweep),
**Table 4** (ablation). Figures: `fig_pipeline`,
`A4_free_run_divergence` (combined), `fig_k_tradeoff`, `boundedness_mechanism`.
Everything else → appendix. This set alone supports all four headline claims:
bounded guarantee, stability vs baselines, K-as-regularizer, penalties→forecastability.
