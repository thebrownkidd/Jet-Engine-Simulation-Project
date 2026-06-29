# ACML Results Bundle and LaTeX Integration Report

This report documents the consolidated paper artifact bundle, what each artifact means, how to cite/use it in LaTeX, and where each item belongs in your requested Results flow.

Bundle root:

- `paper_artifacts/acml_results_bundle`

Main objective:

- Keep all studies in scope.
- Keep main text focused and readable.
- Move depth and diagnostics to appendix.
- Provide direct copy-paste LaTeX usage.

## 1) Consolidated bundle structure

Main-text assets:

- `paper_artifacts/acml_results_bundle/main/tables`
- `paper_artifacts/acml_results_bundle/main/figures`
- `paper_artifacts/acml_results_bundle/main/data`

Appendix assets:

- `paper_artifacts/acml_results_bundle/appendix/tables`
- `paper_artifacts/acml_results_bundle/appendix/figures`

Documentation:

- `paper_artifacts/acml_results_bundle/docs`

## 2) Results section blueprint (exact order)

### 5.1 Experimental protocol

Required table:

- `results/acml/tables/tab_dataset_protocol.tex`

Source data:

- `results/acml/tables/dataset_protocol.csv`

What this table anchors:

- FD001-FD004 complexity ladder using regimes/fault modes.
- Unit counts per split.
- Constant sensor/operating-variable dimensionality.

Columns:

1. `Dataset`: FD001, FD002, FD003, FD004.
2. `Regimes`: operating-condition count (1 or 6).
3. `Fault modes`: fault-type count (1 or 2).
4. `Train units`: number of train engines.
5. `Test units`: number of test engines.
6. `Sensors`: 21.
7. `Operating vars`: 3.

Rows:

- One row per dataset.

### 5.2 Learned latent manifold quality

Required figure:

- `results/figures/combined_fd/D2_manifold_FD001_FD004_combined.png`

Required table/data:

- `results/tables/research/latent_dim/intrinsic_dim.csv`

Optional appendix figure:

- `results/figures/combined_fd/D1_health_trajectories_FD001_FD004_combined.png`

Key paragraph values to state (rho2):

- FD001: 0.965
- FD002: 0.795
- FD003: 0.851
- FD004: 0.849

How to explain the figure:

- 2D embeddings colored by degradation progression show whether trajectories organize into a compact monotone geometry.
- Use this before stability sections to motivate low-dimensional latent dynamics.

### 5.3 Main cross-dataset result

Required table:

- `results/acml/tables/tab_main_cross_dataset.tex`

Source data:

- `results/tables/cross_dataset_results.csv`

Optional supporting figure:

- `results/figures/SUMMARY_cross_dataset.png`

Main evidence to emphasize:

- Final model RUL RMSE across all FD subsets.
- Reconstruction quality.
- Sensor-space VAR spectral radius and growth.
- Bounded manifold rollout norm.

### 5.4 Rollout stability vs baselines

Required table:

- `results/acml/tables/baselines_extended.tex`

Required figure (main preference):

- `results/acml/figures/fig_baselines.png`

Appendix alternatives:

- `results/figures/combined_fd/A4_free_run_divergence_FD001_FD004_combined.png`
- `results/figures/combined_fd/A2_var_eigenvalues_FD001_FD004_combined.png`

Required claims:

- VAR/VAR2 diverge on all datasets.
- TCN diverges catastrophically on FD002 (growth about 1.27e6x).
- GRU diverges on FD002 (growth about 13.0x).
- Manifold remains bounded on all four datasets.
- VAR can look strong at very short horizon NRMSE; claim is stable free-run behavior, not one-step-only accuracy.

### 5.5 Latent dimension: bottleneck or regularizer

Required table:

- `results/acml/tables/k_aware_dim_sweep.tex`

Required figure:

- `results/acml/figures/fig_k_tradeoff.png`

Optional appendix/main-space figure:

- `results/acml/figures/k_vs_reconstruction_and_stability.png`

Required methodological sentence:

- All K coordinates and K velocities are used downstream; higher-dimensional models are not evaluated through only (h0, h1), so the sweep is not confounded.

### 5.6 Ablation and mechanism

Required tables:

- `results/acml/tables/boundedness_mechanism.tex`
- `results/acml/tables/ablation_extended.tex`

Required figure (pick one):

- `results/acml/figures/boundedness_mechanism.png`
- `results/acml/figures/fig_ablation_mechanism.png`

Required nuance sentence:

- The unbounded AE remains finite over evaluated constant-velocity open-loop horizons; empirical advantage is improved forecastability and downstream prognosis, not finite-horizon boundedness alone.

FD002 spotlight values:

- `no_regime_norm`: recon R2 = 0.999, skill = -0.13, RUL RMSE = 40.5.
- `unbounded_ae`: kappa = 0.0346, skill = 0.276.
- `full`: kappa = 0.0019, skill = 0.676.

### 5.7 Seed robustness

Required table:

- `results/acml/tables/seed_robustness.tex`

Optional figure:

- `results/acml/figures/seed_robustness_summary.png`

Headlines:

- Full model bounded in 100% of seeds.
- FD001 RUL: 14.39 +/- 0.45.
- FD002 RUL: 27.40 +/- 0.13.
- FD002 unbounded-AE skill: -0.73 +/- 0.82, full skill: +0.52 +/- 0.13.

### 5.8 Specialist vs generalized

Required table:

- `results/acml/tables/specialist_vs_generalized.tex`

Optional figure:

- `results/acml/figures/specialist_vs_generalized.png`

Main claim:

- Pooled model stays bounded on 3/4 datasets and keeps near-identical FD002 RUL (26.92 vs 26.87), suggesting transferable latent geometry.

### 5.9 Downstream RUL utility

Required source:

- RUL columns from `results/tables/cross_dataset_results.csv`

Optional figure:

- `results/figures/combined_fd/C3_health_vs_rul_FD001_FD004_combined.png`

Appendix alternative:

- `results/figures/combined_fd/C1_rul_scatter_FD001_FD004_combined.png`

## 3) Main table catalog: row/column dictionaries and interpretation

### Table P: dataset protocol

Files:

- LaTeX: `results/acml/tables/tab_dataset_protocol.tex`
- CSV: `results/acml/tables/dataset_protocol.csv`

Rows:

- 4 rows, one per FD dataset.

Columns and meaning:

1. `dataset`: subset identity.
2. `regimes`: operating-condition multiplicity.
3. `fault_modes`: degradation mechanism multiplicity.
4. `train_units`: train sample count in units.
5. `test_units`: test sample count in units.
6. `sensors`: input sensor dimension.
7. `operating_vars`: operating-setting dimension.

Paper use:

- Use at start of Results to define complexity progression and benchmarking context.

### Table 1: main cross-dataset

Files:

- LaTeX: `results/acml/tables/tab_main_cross_dataset.tex`
- CSV: `results/tables/cross_dataset_results.csv`

Rows:

- 4 rows (FD001-FD004).

Columns in LaTeX table and direct meaning:

1. `Dataset`: subset.
2. `Regimes`: operating condition count.
3. `rho2`: fraction of variance captured by first two principal components after preprocessing.
4. `ReconR2`: reconstruction R2 of bounded AE.
5. `rhoVAR`: spectral radius of sensor-space VAR transition matrix.
6. `VARgrowth`: free-run growth factor for VAR trajectory norm.
7. `ManifNorm`: manifold rollout norm at horizon endpoint.
8. `RUL_RMSE`: downstream error in cycles.
9. `RUL_R2`: explanatory power for RUL.

Key interpretive pattern:

- `rhoVAR > 1` and large VAR growth, while manifold norm stays small and RUL remains strong.

### Table 2: baseline rollout comparison

Files:

- LaTeX: `results/acml/tables/baselines_extended.tex`
- CSV: `results/acml/tables/baselines_extended.csv`

Rows:

- 24 rows = 4 datasets x 6 models.
- Model IDs: manifold, var1, var2, lstm, gru, tcn.

Columns:

1. `dataset`: FD subset.
2. `model`: baseline/model label.
3. `rho`: spectral radius (only meaningful for linear VAR models).
4. `freerun_norm`: final trajectory norm.
5. `freerun_growth`: growth ratio final/initial norm.
6. `bounded`: whether growth remains under boundedness criterion.
7. `nrmse_h1`, `nrmse_h10`, `nrmse_h25`, `nrmse_h50`: normalized forecast errors by horizon.

Interpretation guidance:

- Read `freerun_growth` and `bounded` first for stability claim.
- Use horizon NRMSE as caveat context (short horizon can favor VAR).

### Table 3: K-aware dimension sweep

Files:

- LaTeX: `results/acml/tables/k_aware_dim_sweep.tex`
- CSV: `results/acml/tables/k_aware_dim_sweep.csv`

Rows:

- 24 rows = 4 datasets x K in {1,2,3,4,5,6}.

Columns:

1. `dataset`, `k`.
2. `recon_mean_r2`, `recon_min_r2`: reconstruction quality.
3. `freerun_norm`, `freerun_growth`, `freerun_bounded`: stability summary.
4. `nrmse_h1`, `nrmse_h10`, `nrmse_h25`, `nrmse_h50`: horizon errors.
5. `kappa`: latent curvature proxy.
6. `rul_rmse`, `rul_r2`: downstream prognosis quality.

Interpretation guidance:

- Reconstruction generally improves with K.
- Stability/prognosis does not monotonically improve with K.
- FD002 is most diagnostic for instability onset at higher K.

### Table 4A: mechanism (compact)

Files:

- LaTeX: `results/acml/tables/boundedness_mechanism.tex`
- CSV: `results/acml/tables/boundedness_mechanism.csv`

Rows:

- 10 rows = 2 datasets (FD001, FD002) x 5 configs.

Configs:

- unbounded_no_pen
- bounded_no_pen
- bounded_smooth
- bounded_mono
- full

Columns:

1. `bounded_latent`, `mono_penalty`, `smooth_penalty`: mechanism switches.
2. `freerun_bounded`, `freerun_growth`: decoded trajectory boundedness behavior.
3. `latent_growth`, `latent_bounded`: latent-space trajectory behavior.
4. `kappa`: curvature regularity metric.
5. `cv_skill_k20`: forecast skill at horizon 20 against persistence.
6. `recon_mean_r2`: reconstruction quality.
7. `rul_rmse`: downstream RUL error.

Interpretation guidance:

- Separates boundedness from forecastability.
- Shows curvature penalties and monotonicity constraints improve skill even when finite-horizon decoded outputs are bounded in both cases.

### Table 4B: ablation (extended)

Files:

- LaTeX: `results/acml/tables/ablation_extended.tex`
- CSV: `results/acml/tables/ablation_extended.csv`

Rows:

- 14 rows = 2 datasets x 7 variants.

Variants:

- full
- no_regime_norm
- no_mono
- no_smooth
- no_mono_smooth
- bounded_no_pen
- unbounded_ae

Columns:

1. Design flags: `bounded_latent`.
2. Reconstruction: `recon_mean_r2`, `recon_min_r2`.
3. Monotonicity violation: `mono_viol_frac`.
4. Dynamics: `kappa`, `freerun_growth`, `freerun_bounded`.
5. Horizon errors: `nrmse_h1/h10/h25/h50`.
6. Forecasting: `cv_skill_k20`.
7. Prognosis: `rul_rmse`, `rul_mae`, `rul_r2`, with `base_rmse` reference.

Interpretation guidance:

- Demonstrates that high reconstruction alone is insufficient for prognosis quality.

### Table 5: seed robustness

Files:

- LaTeX: `results/acml/tables/seed_robustness.tex`
- CSV detail: `results/acml/tables/seed_robustness.csv`
- CSV summary: `results/acml/tables/seed_robustness_summary.csv`

Rows:

- Summary file has 6 rows = 2 datasets x 3 variants.

Variants represented:

- full
- no_smooth
- unbounded_ae

Columns:

1. `recon_mean`, `recon_std`.
2. `growth_mean`, `growth_std`.
3. `bounded_frac`.
4. `kappa_mean`, `kappa_std`.
5. `skill_mean`, `skill_std`.
6. `rul_rmse_mean`, `rul_rmse_std`.
7. `rul_r2_mean`, `rul_r2_std`.

Interpretation guidance:

- Use mean +/- std for each metric; this is your reproducibility shield.

### Table 6: specialist vs generalized

Files:

- LaTeX: `results/acml/tables/specialist_vs_generalized.tex`
- CSV: `results/acml/tables/specialist_vs_generalized.csv`

Rows:

- 8 rows = 4 datasets x {specialist, generalized}.

Columns:

1. `recon_mean_r2`.
2. `freerun_growth`.
3. `freerun_bounded`.
4. `rul_rmse`.
5. `rul_r2`.

Interpretation guidance:

- Compare within each dataset pair; do not average across datasets without weighting.

## 4) Figure catalog: what each visualization shows

### Main figures

1. `results/acml/figures/fig_pipeline.png`
- Visualization type: pipeline diagram.
- Shows: preprocessing, latent model, rollout, and RUL mapping stages.
- Use: opening figure for method-to-results linkage.

2. `results/figures/combined_fd/D2_manifold_FD001_FD004_combined.png`
- Visualization type: manifold projections for all FD subsets.
- Shows: geometric compactness and progression ordering.
- Use: latent quality section.

3. `results/acml/figures/fig_baselines.png`
- Visualization type: grouped/bar comparison of free-run growth by model and dataset.
- Shows: boundedness contrast across manifold vs VAR/VAR2/LSTM/GRU/TCN.
- Use: main baseline stability section.

4. `results/acml/figures/fig_k_tradeoff.png`
- Visualization type: dual-panel tradeoff (reconstruction vs K, RUL vs K).
- Shows: capacity gains do not guarantee stability/prognosis gains.
- Use: K-aware section.

5. `results/acml/figures/boundedness_mechanism.png`
- Visualization type: mechanism/tradeoff plot (growth, kappa, or skill by config).
- Shows: effect of penalties and bounded latent constraints.
- Use: ablation and mechanism section.

6. `results/figures/combined_fd/C3_health_vs_rul_FD001_FD004_combined.png` (if space allows)
- Visualization type: latent-health to RUL relationship.
- Shows: interpretability of learned health coordinate for downstream RUL.
- Use: downstream utility subsection.

### Appendix figure set

1. `results/figures/combined_fd/A2_var_eigenvalues_FD001_FD004_combined.png`
- Shows eigenvalue argument for VAR instability.

2. `results/figures/combined_fd/A4_free_run_divergence_FD001_FD004_combined.png`
- Shows norm divergence trajectories directly.

3. `results/figures/combined_fd/A1_rollout_r2_vs_horizon_FD001_FD004_combined.png`
- Shows horizon-wise fit/decay behavior.

4. `results/figures/combined_fd/A3_example_trajectories_FD001_FD004_combined.png`
- Shows representative sensor trajectories under rollouts.

5. `results/figures/combined_fd/B1_health_forecasts_FD001_FD004_combined.png`
- Shows health forecast examples.

6. `results/figures/combined_fd/B2_error_vs_horizon_FD001_FD004_combined.png`
- Shows error growth by horizon.

7. `results/figures/combined_fd/B3_skill_vs_persistence_FD001_FD004_combined.png`
- Shows forecast skill relative to persistence baseline.

8. `results/figures/combined_fd/C1_rul_scatter_FD001_FD004_combined.png`
- Shows predicted-vs-true RUL scatter and error spread.

9. `results/figures/combined_fd/C2_examples_FD001_FD004_combined.png`
- Shows per-engine examples.

10. `results/figures/combined_fd/D1_health_trajectories_FD001_FD004_combined.png`
- Shows monotone health trajectory behavior.

11. `results/acml/figures/seed_robustness_summary.png`
- Shows mean/std bars for seed sensitivity.

12. `results/acml/figures/specialist_vs_generalized.png`
- Shows pooled vs specialist comparison visually.

13. `results/acml/figures/fig_ablation_mechanism.png`
- Compact mechanism summary alternative.

14. `results/acml/figures/k_vs_reconstruction_and_stability.png`
- Expanded K tradeoff with stability axis.

15. `results/acml/figures/k_vs_rul.png`
- RUL-only view of K sweep.

## 5) Ready-to-paste LaTeX integration

Assume copied assets under `figures/` and `tables/` in your paper project.

### Suggested package setup

```tex
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{siunitx}
```

### 5.1 Protocol table

```tex
\input{tables/tab_dataset_protocol.tex}
```

### 5.2 Manifold quality

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{figures/D2_manifold_FD001_FD004_combined.png}
  \caption{Cross-dataset latent manifold projections showing compact degradation geometry.}
  \label{fig:d2_manifold_combined}
\end{figure}
```

### 5.3 Main cross-dataset table

```tex
\input{tables/tab_main_cross_dataset.tex}
```

### 5.4 Baselines

```tex
\input{tables/baselines_extended.tex}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\linewidth]{figures/fig_baselines.png}
  \caption{Free-run growth comparison across manifold and sensor-space baselines.}
  \label{fig:baselines_main}
\end{figure}
```

### 5.5 K-aware sweep

```tex
\input{tables/k_aware_dim_sweep.tex}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\linewidth]{figures/fig_k_tradeoff.png}
  \caption{Reconstruction-capacity gain versus stability/prognosis tradeoff across latent dimension K.}
  \label{fig:k_tradeoff}
\end{figure}
```

### 5.6 Ablation and mechanism

```tex
\input{tables/boundedness_mechanism.tex}
\input{tables/ablation_extended.tex}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\linewidth]{figures/boundedness_mechanism.png}
  \caption{Mechanism analysis separating boundedness from forecastability improvements.}
  \label{fig:mechanism}
\end{figure}
```

### 5.7 Seed robustness

```tex
\input{tables/seed_robustness.tex}
```

### 5.8 Specialist vs generalized

```tex
\input{tables/specialist_vs_generalized.tex}
```

### 5.9 Optional downstream RUL figure

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{figures/C3_health_vs_rul_FD001_FD004_combined.png}
  \caption{Learned latent health coordinate versus RUL across datasets.}
  \label{fig:health_vs_rul}
\end{figure}
```

## 6) Appendix package mapping

Appendix table/data items in bundle:

- `theory_impl_rollout_compare.csv`
- `rul_predictions.csv`
- `rul_metrics.json`
- `recon_vs_k.csv`
- `regime_mining.csv`
- `rollout_stability.csv`
- `health_forecasting.csv`
- `extra_dataset_BLOCKED.txt`

Appendix figures in bundle:

- A-series: A1/A2/A3/A4 combined FD panels.
- B-series: B1/B2/B3 combined FD panels.
- C-series: C1/C2 combined FD panels.
- D-series: D1 combined FD panel.
- Extra ACML diagnostics: seed robustness, specialist-generalized, alternate ablation, alternate K plots.

## 7) Newly added missing/underdeveloped artifacts

Created now (templates or diagnostics):

1. Dataset/protocol artifact
- `results/acml/tables/tab_dataset_protocol.tex`
- `results/acml/tables/dataset_protocol.csv`

2. Runtime/training-cost template
- `results/acml/tables/tab_runtime_training_cost_template.tex`
- `results/acml/tables/runtime_training_cost_template.csv`

3. Lambda sensitivity template
- `results/acml/tables/tab_lambda_sensitivity_template.tex`
- `results/acml/tables/lambda_sensitivity_template.csv`

4. Open-loop vs closed-loop clarification diagnostic
- `results/acml/tables/tab_open_vs_closed_loop_diagnostic.tex`
- `results/acml/tables/open_vs_closed_loop_diagnostic.csv`

5. Decoder saturation diagnostic template
- `results/acml/tables/tab_decoder_saturation_diagnostic_template.tex`
- `results/acml/tables/decoder_saturation_diagnostic_template.csv`

How to present these in paper:

- Runtime and lambda tables can go to appendix if space is tight.
- Open-vs-closed diagnostic is very useful in main text as a compact reviewer-clarification table.
- Decoder saturation can remain appendix unless you run the actual activation-trace extraction.

## 8) Limitations sentence to include

Recommended wording:

- All experiments are currently on C-MAPSS; external bearing or industrial datasets remain future work (see `extra_dataset_BLOCKED.txt`).

## 9) Final checklist before manuscript freeze

1. Confirm all table labels are unique and match references.
2. Keep one baseline figure in main text (`fig_baselines`), move A2/A4 to appendix unless page budget allows both.
3. Keep cross-dataset evidence table-first; use summary figure only as compact visual aid.
4. Include mean +/- std seed values directly in text near robustness table.
5. Include explicit claim boundary: stability and forecastability over long horizon, not strict one-step superiority.
