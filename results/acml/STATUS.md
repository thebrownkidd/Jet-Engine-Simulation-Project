# ACML Upgrade — Status Tracker

Central claim under test: *Bounded low-dimensional latent representations act as
dynamical regularizers for stable long-horizon multivariate time-series rollout.
Higher-dimensional latent spaces can reconstruct better but their additional
degrees of freedom can weaken rollout stability. Boundedness is architectural;
monotonicity and smoothness improve forecastability.*

Artifact root: `results/acml/` (tables, figures, models) and `docs/` (reports).

Reproducibility: all retraining scripts accept a fixed seed. Global reduced
epoch budget `ACML_EPOCHS` is documented per task; production model uses 4000.

| # | Task | Status | Script / command | Output artifacts | Notes & caveats |
|---|------|--------|------------------|------------------|-----------------|
| 1 | Theory/implementation consistency | done | `experiments/acml/exp_theory_impl_check.py` | `docs/theory_implementation_check.md`, `theory_impl_rollout_compare.csv` | Full-box projection makes Thm 2 hold verbatim; metric change immaterial (FD003 growth 3.02→2.62) |
| 2 | Extended ablation (7 variants incl. unbounded AE) | done | `experiments/acml/exp_ablation_extended.py` | `ablation_extended.csv/.tex`, `ablation_extended_summary.png` | FD002 no_regime_norm: recon 0.999 but skill −0.13, RUL 40.5 (recon≠prognosis). Unbounded AE bounded over finite horizon (see caveat) |
| 3 | Multi-seed robustness (seeds 0–4) | done | `experiments/acml/exp_seed_robustness.py` | `seed_robustness*.csv`, `.png`, `.tex` | full bounded 100%; FD002 unbounded-AE skill −0.73±0.82 vs full +0.52±0.13 |
| 4 | K-aware latent dim sweep (k=1..6) | done | `experiments/acml/exp_k_aware_sweep.py` | `k_aware_dim_sweep.csv/.tex`, 2 figs | ALL k coords used. FD002 unbounded at k≥3; recon↑ but RUL flat |
| 5 | Boundedness mechanism | done | `experiments/acml/exp_boundedness_mechanism.py` | `boundedness_mechanism.csv/.tex`, `.png` | See KEY FINDING below |
| 6 | Modern neural baseline (TCN) | done | `experiments/acml/exp_baselines_extended.py` | `baselines_extended.csv/.tex`, freerun figs | TCN diverges 1.27e6× on FD002; manifold bounded all 4 |
| 7 | Specialist vs generalized | done | `experiments/acml/exp_specialist_vs_generalized.py` | `specialist_vs_generalized.csv/.tex`, `.png` | generalized preserves bounded rollout 3/4; RUL ~equal |
| 8 | External dataset (optional) | blocked | `experiments/acml/exp_external_dataset.py` | `extra_dataset_BLOCKED.txt` | No external dataset available; script ready via --data |
| 9 | ACML-ready figures & tables | done | `experiments/acml/make_acml_assets.py` | `fig_pipeline/k_tradeoff/ablation_mechanism/baselines.png`, `tab_main_cross_dataset.tex` | 300 DPI, one-column friendly |
| 10 | ACML readiness report | done | — | `docs/ACML_READINESS_REPORT.md` | Final honest synthesis |

Status legend: pending / running / done / blocked.

## KEY FINDING (honest, affects claim framing)

Under the proposed **local constant-velocity latent rollout**, the decoded
free-run stays bounded for *every* variant tested — including the **unbounded
latent AE** — over practical horizons. Two architectural facts over-determine
this: (i) the compact latent box, and (ii) the bounded-output tanh decoder
(tanh saturates even on a growing latent). The genuinely **unbounded**
behaviour belongs to the **closed-loop sensor-space VAR/VAR2** (spectral radius
ρ>1), which diverge 25–760×.

Therefore the central claim must be framed precisely:
- *Boundedness is architectural* — the bounded latent **guarantees** bounded
  rollout for all horizons (Theorem 2); the contrast that exhibits divergence is
  the closed-loop sensor-space recursion, not a finite-horizon unbounded-AE run.
- *Monotonicity and smoothness improve forecastability* — **strongly confirmed**:
  curvature κ drops several-fold (FD002: 0.0346 unbounded → 0.0019 full) and
  forecast skill rises (FD002: +0.276 → +0.676), with corresponding RUL gains.
- *Reconstruction ≠ prognosis* — **strongly confirmed**: FD002 `no_regime_norm`
  reaches recon R²=0.999 yet collapses to skill −0.13 and RUL 40.5.
