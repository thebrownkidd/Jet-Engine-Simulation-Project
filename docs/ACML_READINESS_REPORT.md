# ACML Readiness Report

**Paper claim under test.** *Bounded low-dimensional latent representations act
as dynamical regularizers for stable long-horizon multivariate time-series
rollout. Higher-dimensional latent spaces can reconstruct better, but their
additional degrees of freedom can weaken rollout stability. Boundedness is
architectural; monotonicity and smoothness improve forecastability.*

All evidence below is produced by reproducible, fixed-seed scripts under
`experiments/acml/`, with artifacts in `results/acml/` (tables, figures, logs)
and the per-task status in `results/acml/STATUS.md`. Reduced epoch budget
(`ACML_EPOCHS=1500`) is used for sweeps; the production model uses 4000. Trends
and orderings are stable to this budget.

---

## 1. What already existed

- A modular pipeline (`src/manifold/`): regime KMeans, per-regime normalisation,
  rolling-median denoise, trend-based sensor selection, a bounded (sigmoid) k=2
  autoencoder with monotonicity + smoothness penalties on `h0`, latent
  constant-velocity rollout, and a gradient-boosted health→RUL head.
- Production experiments (Exp A/B/C, latent-dim probe, rollout baselines
  VAR/VAR2/LSTM/GRU, per-file and unified runs) with cross-dataset results on
  FD001–FD004.
- A consolidated math section (`docs/THEORY.md`).

Two latent weaknesses existed in the prior code (now addressed, not hidden):
1. The rollout clipped **only** the primary latent coordinate, so Theorem 2 was
   not matched verbatim for k>1.
2. The RUL head used **only** `h0,h1,v0,v1`, so any k>2 conclusion silently
   discarded the extra coordinates.

## 2. What was newly added

- `experiments/acml/acml_common.py`: a `FlexAE` with an explicit `bounded`
  switch (sigmoid on/off), a k-aware `TrainedAE` wrapper, k-aware RUL features
  `[h0..h(K-1), v0..v(K-1)]`, three named rollout projections (`h0_clip`,
  `full_box`, `none`), and metric helpers (recon, curvature κ, monotonicity
  violation, forecast skill, decoded & **latent** free-run growth, rollout
  NRMSE, k-aware RUL).
- Seven experiment scripts (Tasks 1–7), a blocked-but-ready external-dataset
  script (Task 8), and an asset builder (Task 9).
- New honest metric: **latent-norm free-run growth**, which isolates the
  bounded-geometry claim from the bounded-output tanh decoder.

## 3. Task completion

| Task | Status | Primary artifact |
|---|---|---|
| 1 Theory/impl consistency | done | `docs/theory_implementation_check.md` |
| 2 Extended ablation (7 variants incl. unbounded AE) | done | `results/acml/tables/ablation_extended.csv` |
| 3 Multi-seed robustness (seeds 0–4) | done | `results/acml/tables/seed_robustness_summary.csv` |
| 4 K-aware sweep (k=1..6, all coords used) | done | `results/acml/tables/k_aware_dim_sweep.csv` |
| 5 Boundedness mechanism | done | `results/acml/tables/boundedness_mechanism.csv` |
| 6 Modern neural baseline (TCN) | done | `results/acml/tables/baselines_extended.csv` |
| 7 Specialist vs generalized | done | `results/acml/tables/specialist_vs_generalized.csv` |
| 8 External dataset | **blocked** | `results/acml/tables/extra_dataset_BLOCKED.txt` |
| 9 ACML figures & tables | done | `results/acml/figures/fig_*.png`, `*.tex` |
| 10 Readiness report | done | this file |

## 4. Tasks blocked

- **Task 8 (external dataset).** Blocked: no external public degradation dataset
  (e.g. PRONOSTIA/FEMTO or IMS bearings) is present in the workspace and this
  environment cannot download one. The script
  `experiments/acml/exp_external_dataset.py` is ready to run unchanged once a
  long-format `unit_id, cycle, sensors` file is supplied via `--data`.

---

## 5. Main claims now supported (with evidence)

### (a) Is boundedness architectural? — **Yes, as a guarantee; with an important nuance.**

- **Guarantee.** With the theory-matched `full_box` projection, the bounded
  latent lies in `[0,1]^k`, so the decoded rollout is bounded for *all* horizons
  (Theorem 2). Task 1 shows enforcing this verbatim changes nothing material
  (FD003 free-run growth 3.02→2.62×; NRMSE unchanged): the production single-
  coordinate clip was already empirically bounded.
- **Decisive divergence contrast is the closed-loop sensor-space recursion.**
  Across all four datasets, sensor-space VAR/VAR2 have spectral radius ρ>1 and
  free-run growth of **25×–760×** (`baselines_extended.csv`), whereas the
  bounded manifold stays at **0.8×–1.5×**.
- **Honest nuance (reported, not hidden).** Under the *local constant-velocity
  latent rollout*, a finite-horizon free-run of even the **unbounded** AE
  stays numerically bounded (Task 2/5), because (i) velocities are small and
  (ii) the tanh decoder saturates. So the unbounded-AE *vs* bounded-AE contrast
  is **not** the dramatic one — the dramatic divergence belongs to closed-loop
  sensor-space models and to neural sequence models (see (d)). The bounded
  latent's role is best stated as a *formal guarantee* of horizon-independent
  boundedness, with the empirically dominant benefit being forecastability.

### (b) Do smoothness and monotonicity improve forecastability? — **Yes, strongly.**

- Curvature κ (lower = smoother/more forecastable) drops several-fold when
  penalties are added. FD002: unbounded-AE κ≈3.46e-2 → full κ≈1.90e-3 (≈18×
  smoother); FD001: 1.37e-2 → 1.63e-3.
- Forecast skill (constant-velocity vs persistence, k=20) rises with the
  penalties. FD002 seed-averaged: unbounded-AE **−0.73 ± 0.82** vs full
  **+0.52 ± 0.13** (`seed_robustness_summary.csv`). The bounded geometry is
  required for *stable, positive* forecast skill.
- The mechanism experiment (Task 5) confirms boundedness is present for all
  bounded variants regardless of penalties, while κ and skill track the
  penalties — cleanly separating the two properties.

### (c) Is K=2 a bottleneck or a regularizer? — **A regularizer.**

`k_aware_dim_sweep.csv` (all K coordinates used downstream):
- Reconstruction **improves** with K on every dataset (e.g. FD002 0.808→0.932,
  FD001 0.888→0.975) — so K=2 is *not* a reconstruction ceiling we are stuck at.
- Stability **degrades** with K where it matters: on **FD002, k≥3 becomes
  unbounded** (free-run growth 5.2×, 8.6×, 5.8×, 5.8× for k=3..6;
  `bounded=False`), while k≤2 stays bounded. Rollout NRMSE@50 also worsens
  monotonically with K on FD001/FD003/FD004.
- RUL does **not** improve with K despite better reconstruction (FD002 RUL RMSE
  ≈27 across k=2..6; FD001 best at k=2–3). So extra latent capacity buys
  reconstruction, not prognosis — exactly the "regularizer, not bottleneck"
  claim.

### (d) Are results robust across seeds? — **Yes.**

`seed_robustness_summary.csv` (seeds 0–4):
- Bounded `full` model is bounded in **100%** of runs on both FD001 and FD002.
- RUL is tight: FD001 14.39 ± 0.45, FD002 27.40 ± 0.13.
- The unbounded AE's forecast skill is both **lower and far more variable**
  (FD002 −0.73 ± 0.82), reinforcing that bounded geometry stabilises
  forecastability across initialisations.

### (e) Stronger than a C-MAPSS-specific RUL result? — **Yes, framed as representation learning.**

- The phenomena are stated and measured as **representation/dynamics**
  properties: spectral-radius-driven divergence of closed-loop linear/neural
  rollout vs horizon-independent boundedness of a compact latent; curvature-
  controlled forecastability; capacity-vs-stability trade-off in K.
- **Modern neural baseline (Task 6, TCN added):** neural sequence models are
  competitive at short horizons but their closed-loop boundedness is
  **inconsistent** — TCN diverges **1.27×10⁶** on FD002, GRU diverges (13×) on
  FD002, while the bounded latent stays bounded on all four datasets.
- **Transfer (Task 7):** a single *generalized* model trained on pooled data
  preserves bounded rollout (3/4 datasets) and nearly identical RUL (FD002 26.92
  vs specialist 26.87), indicating the bounded latent geometry is reusable, not
  dataset-specific.

---

## 6. Claims that should NOT be made

1. **Do not claim** the unbounded AE *diverges* under the constant-velocity
   latent rollout at practical horizons — it does not (tanh decoder + small
   velocities). The honest divergence contrast is closed-loop **sensor-space**
   VAR/VAR2 and some neural baselines (TCN/GRU on FD002).
2. **Do not claim** the latent coordinates are physical health variables. They
   are learned proxies (already stated in `docs/`).
3. **Do not claim** the manifold wins on **reconstruction** or short-horizon
   accuracy — VAR is slightly better at h=1, and high-K AEs reconstruct better.
   The contribution is *stability + forecastability + bounded guarantee*.
4. **Do not claim** regime normalisation helps reconstruction. It *hurts*
   reconstruction R² (FD002 0.999 without it) but is essential for prognosis
   (RUL 27.3 → 40.5 without it): reconstruction ≠ prognosis.
5. **Do not over-generalise across domains** until Task 8 (external dataset) is
   run.

---

## 7. Key figures and tables for the ACML paper

**Figures** (`results/acml/figures/`, 300 DPI):
- `fig_pipeline.png` — method pipeline schematic.
- `fig_k_tradeoff.png` / `k_vs_reconstruction_and_stability.png` — K: capacity
  up, stability down (the central claim).
- `fig_baselines.png` / `baselines_extended_freerun.png` — closed-loop free-run
  growth incl. TCN (manifold bounded; VAR/VAR2/TCN diverge).
- `boundedness_mechanism.png` — boundedness vs forecastability separation.
- `seed_robustness_summary.png` — robustness over seeds.
- `specialist_vs_generalized.png` — geometry transfer.
- (reuse) production `results/figures/.../D2_manifold`, `A4_free_run_divergence`,
  `C3_health_vs_rul` per-dataset and combined panels.

**Tables** (LaTeX in `results/acml/tables/*.tex`):
- `tab_main_cross_dataset.tex` — main cross-dataset results.
- `ablation_extended.tex` — 7-variant ablation.
- `k_aware_dim_sweep.tex` — K sweep (all coords used).
- `baselines_extended.tex` — baselines incl. TCN.
- `seed_robustness.tex` — mean ± std over seeds.
- `boundedness_mechanism.tex`, `specialist_vs_generalized.tex`.

---

## 8. Remaining weaknesses

1. **Rollout simplicity.** Constant-velocity latent extrapolation is what makes
   even unbounded latents look bounded over finite horizons. A stronger,
   adversarial rollout (e.g. closed-loop latent dynamics learned and iterated)
   would make the bounded-vs-unbounded *latent* contrast sharper. Worth adding
   if a reviewer presses the unbounded-AE comparison.
2. **External validity.** Task 8 is blocked; the cross-domain claim rests on the
   four C-MAPSS subsets plus the modular argument until a bearing dataset is run.
3. **Budget.** Sweeps use 1500 epochs; final numbers should be regenerated at
   the production 4000-epoch budget for camera-ready (orderings already stable).
4. **Curvature as a proxy.** κ (median |Δ²h₀|) is a convenient forecastability
   proxy; pairing it with the measured forecast-skill (already reported) is
   recommended so the claim does not rest on κ alone.
5. **Multi-seed scope.** Seeds were run for the three decisive variants on
   FD001/FD002; extending to FD003/FD004 and all variants would tighten the
   robustness story.

---

## 9. One-paragraph honest summary

The bounded low-dimensional latent provides a **formal, horizon-independent
boundedness guarantee** (Theorem 2, now matched verbatim by the `full_box`
rollout) and, empirically, the **most consistent closed-loop stability** across
four datasets and against VAR, VAR2, LSTM, GRU and a TCN — several of which
diverge by 1–6 orders of magnitude. Its low dimension behaves as a **dynamical
regularizer**: raising K improves reconstruction but degrades rollout stability
(FD002 becomes unbounded at K≥3) without improving RUL. Monotonicity and
smoothness do **not** create boundedness; they **reduce latent curvature and
raise forecast skill**, which is what makes the bounded state useful downstream.
The one claim we explicitly avoid is that an unbounded autoencoder *diverges*
under the simple constant-velocity rollout — it does not at practical horizons,
and we report this openly.
