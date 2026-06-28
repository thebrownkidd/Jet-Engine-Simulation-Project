# Paper Bundle Index

This bundle centralizes comparative-study outputs and final algorithm results.

## Root Path

- results/paper_bundle

## A) Comparative Study Results

### Tables

- results/paper_bundle/comparative/tables/intrinsic_dim.csv  
  Use: intrinsic dimension and PCA energy by dataset (latent-capacity evidence).
- results/paper_bundle/comparative/tables/recon_vs_k.csv  
  Use: reconstruction quality vs latent dimension k.
- results/paper_bundle/comparative/tables/regime_mining.csv  
  Use: heuristic vs mined regime counts and silhouette scores.
- results/paper_bundle/comparative/tables/ablation_summary.csv  
  Use: full/no-regime/no-mono/no-smooth ablation impacts on recon + RUL.
- results/paper_bundle/comparative/tables/rollout_ablation_summary.csv  
  Use: rollout-focused ablation impacts on boundedness, growth, curvature, skill.
- results/paper_bundle/comparative/tables/baselines_summary.csv  
  Use: manifold vs VAR1/VAR2/LSTM/GRU rollout comparison.
- results/paper_bundle/comparative/tables/per_file_dim_summary.csv  
  Use: per-dataset chosen k and effect on recon/rollout/RUL.
- results/paper_bundle/comparative/tables/unified_per_file.csv  
  Use: unified model performance on each FD dataset.

### Figures

- results/paper_bundle/comparative/figures/recon_vs_k.png
- results/paper_bundle/comparative/figures/rollout_ablation_FD001.png
- results/paper_bundle/comparative/figures/rollout_ablation_FD002.png
- results/paper_bundle/comparative/figures/freerun_FD001.png
- results/paper_bundle/comparative/figures/freerun_FD002.png
- results/paper_bundle/comparative/figures/freerun_FD003.png
- results/paper_bundle/comparative/figures/freerun_FD004.png
- results/paper_bundle/comparative/figures/A1_rollout_r2_vs_horizon_FD001_FD004_combined.png
- results/paper_bundle/comparative/figures/A2_var_eigenvalues_FD001_FD004_combined.png
- results/paper_bundle/comparative/figures/A3_example_trajectories_FD001_FD004_combined.png
- results/paper_bundle/comparative/figures/A4_free_run_divergence_FD001_FD004_combined.png
- results/paper_bundle/comparative/figures/B1_health_forecasts_FD001_FD004_combined.png
- results/paper_bundle/comparative/figures/B2_error_vs_horizon_FD001_FD004_combined.png
- results/paper_bundle/comparative/figures/B3_skill_vs_persistence_FD001_FD004_combined.png
- results/paper_bundle/comparative/figures/C1_rul_scatter_FD001_FD004_combined.png
- results/paper_bundle/comparative/figures/C2_examples_FD001_FD004_combined.png
- results/paper_bundle/comparative/figures/C3_health_vs_rul_FD001_FD004_combined.png
- results/paper_bundle/comparative/figures/D1_health_trajectories_FD001_FD004_combined.png
- results/paper_bundle/comparative/figures/D2_manifold_FD001_FD004_combined.png

## B) Final Algorithm Results

### Tables

- results/paper_bundle/final_algorithm/tables/cross_dataset_results.csv
- results/paper_bundle/final_algorithm/tables/cross_dataset_results.json
- results/paper_bundle/final_algorithm/tables/rollout_stability.csv
- results/paper_bundle/final_algorithm/tables/health_forecasting.csv
- results/paper_bundle/final_algorithm/tables/rul_predictions.csv
- results/paper_bundle/final_algorithm/tables/rul_metrics.json

### Figures

- results/paper_bundle/final_algorithm/figures/SUMMARY_cross_dataset.png
- results/paper_bundle/final_algorithm/figures/FD001/
- results/paper_bundle/final_algorithm/figures/FD002/
- results/paper_bundle/final_algorithm/figures/FD003/
- results/paper_bundle/final_algorithm/figures/FD004/

Per-FD folders contain:
- A1_rollout_r2_vs_horizon.png
- A2_var_eigenvalues.png
- A3_example_trajectories.png
- A4_free_run_divergence.png
- B1_health_forecasts.png
- B2_error_vs_horizon.png
- B3_skill_vs_persistence.png
- C1_rul_scatter.png
- C2_examples.png
- C3_health_vs_rul.png
- D1_health_trajectories.png
- D2_manifold.png

## C) Headline Metrics (from final tables)

From cross_dataset_results.csv:
- FD001: RUL RMSE 14.53, R2 0.878, baseline RMSE 43.07
- FD002: RUL RMSE 27.02, R2 0.748, baseline RMSE 54.08
- FD003: RUL RMSE 16.31, R2 0.845, baseline RMSE 45.07
- FD004: RUL RMSE 27.58, R2 0.744, baseline RMSE 54.90

From baselines_summary.csv:
- VAR1 and VAR2 are unbounded on FD002-FD004 (rho > 1; large free-run growth).
- Manifold rollout remains bounded across datasets in the baseline sweep.
