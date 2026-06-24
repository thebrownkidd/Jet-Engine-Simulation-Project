# A Physics-Constrained Health Manifold for Turbofan Remaining-Useful-Life Prediction: Discovery, Provably-Bounded Rollout, and Cross-Dataset Validation on C-MAPSS

> **One-line summary.** We learn a 2-dimensional, monotone, physics-constrained
> *health manifold* from raw turbofan sensor streams, prove that rolling it out
> forward in time is *bounded for all horizons* (whereas the natural
> sensor-space linear model provably diverges), and show that the health state
> predicts Remaining Useful Life (RUL) on **all four** C-MAPSS sub-datasets
> (FD001–FD004), beating a mean-RUL baseline on every one.

---

## Abstract

Data-driven prognostics for jet engines typically train a black-box regressor
to map a sliding window of sensors to RUL. Such models are accurate but offer
no guarantee that the *internal dynamics they imply* are physically sensible —
roll them forward and they can diverge. We take the opposite route: we first
**discover** a low-dimensional degradation state, constrain it to be bounded
and monotone (the two properties a wear coordinate must have), and only then
read RUL off it. The pipeline has four stages — *discovery*, *rollout
stability*, *health forecasting*, and *RUL prediction* — each tied to a formal
theorem. The decisive structural result is a **boundedness dichotomy**: a
sensor-space vector autoregression (VAR) has spectral radius
$\rho(A) > 1$ on every dataset and free-runs to $25\text{–}760\times$ its
initial norm, while the manifold rollout is provably bounded by the decoder's
Lipschitz constant and stays flat. We generalize the originally FD001-only
method to the multi-regime, multi-fault datasets FD002–FD004 using an
**operating-condition normalization** (regime clustering + per-regime
standardization) so that a common degradation residual is exposed regardless of
flight condition. Across FD001–FD004 the health$\to$RUL map attains test
$R^2 \in [0.744, 0.878]$ and RMSE below the mean baseline on all four datasets.

---

## 1. Introduction

### 1.1 Problem

Given multivariate sensor histories from a fleet of turbofan engines run to
failure, predict the **Remaining Useful Life (RUL)** of a partially-degraded
engine: how many operating cycles remain before it crosses a failure threshold.
This is the canonical NASA C-MAPSS benchmark.

### 1.2 Why a physics-grounded latent state

A direct sensors-$\to$-RUL regressor cannot answer two questions a maintenance
engineer actually cares about:

1. **Is the implied degradation trajectory stable?** If you ask the model to
   *simulate* the engine forward, does it stay in a physically plausible
   envelope, or does it blow up?
2. **Is the latent it uses monotone in wear?** Wear is a one-way process; a
   health coordinate that wanders is not interpretable.

We therefore impose the two non-negotiable properties of a wear coordinate —
**boundedness** (via a logistic bottleneck) and **monotonicity** (via a
penalty) — and make them *theorems*, not hopes. RUL then becomes a smooth
read-out of an interpretable state.

### 1.3 Contributions

1. **A physics-constrained 2-D health autoencoder** with monotonicity and
   smoothness penalties whose latent is bounded by construction.
2. **A boundedness dichotomy** (Theorems 1–2): the natural sensor-space VAR
   provably diverges ($\rho(A)>1$), the manifold rollout provably cannot.
   Verified empirically on all four datasets.
3. **A polynomial forecast-error bound** (Theorem 3) for the smooth, monotone
   health coordinate, explaining why constant-velocity extrapolation in
   *health* space works where autoregression in *sensor* space explodes.
4. **An operating-condition normalization** that generalizes the whole method
   from the single-regime FD001 to the 6-regime / 2-fault datasets FD002–FD004
   with no architecture change.
5. **A cross-dataset evaluation** (Section 8) on FD001–FD004 with a single
   reproducible driver.

```mermaid
flowchart LR
    ID["<b>Discovery</b><br/>identifiable 2-D health<br/>(PCA + constrained AE)"]
    ST["<b>Stable rollout</b><br/>bounded vs VAR rho>1"]
    FC["<b>Forecastable</b><br/>error ~ (kappa/2) t^2"]
    RUL["<b>RUL</b><br/>health -> RUL regressor"]
    ID --> ST --> FC --> RUL
```

---

## 2. Data: NASA C-MAPSS (FD001–FD004)

Each sub-dataset is a fleet of engines with 21 sensors and 3 operating-condition
settings, run from healthy to failure (training) or truncated before failure
(test, with ground-truth RUL provided separately). The four sub-datasets form a
$2\times2$ difficulty grid in **# operating regimes** $\times$ **# fault modes**:

| Dataset | Operating regimes | Fault modes | Train units | Test units |
|---|---|---|---|---|
| **FD001** | 1 | 1 (HPC degradation) | 100 | 100 |
| **FD002** | 6 | 1 (HPC degradation) | 260 | 259 |
| **FD003** | 1 | 2 (HPC + Fan) | 100 | 100 |
| **FD004** | 6 | 2 (HPC + Fan) | 249 | 248 |

The diagonal stress test is the whole point: FD001 is the easy corner (one
condition, one fault); FD004 is the hard corner (six conditions, two faults).
A method that only works on FD001 has overfit to an unusually clean regime.

### 2.1 The multi-regime problem

In FD002/FD004 the engine cycles through six discrete flight conditions. Raw
sensors therefore jump in a *staircase* that has nothing to do with wear — it
is pure operating point. Any degradation signal is buried under this staircase.
Section 4.1 removes it.

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| $x_t \in \mathbb{R}^{m}$ | sensor vector at cycle $t$ ($m$ = # dynamic sensors kept) |
| $c_t \in \{1,\dots,R\}$ | operating-regime label at cycle $t$ ($R$ regimes) |
| $z_t$ | condition-normalized sensor residual (Section 4.1) |
| $\bar{x}_t$ | denoised (rolling-median) trend |
| $h_t = E_\theta(x_t) \in (0,1)^k$ | health state, $k=2$ |
| $\hat x_t = D_\phi(h_t)$ | decoder reconstruction |
| $d_t \in [0,1]$ | life fraction (cycle / failure cycle) |
| $\rho(A)$ | spectral radius of VAR matrix $A$ |
| $\kappa$ | bound on $|\Delta^2 h_0|$ (health curvature) |

---

## 4. Methodology

### 4.0 Pipeline overview

```mermaid
flowchart TD
    A["Raw sensors x_t + settings"] --> B["<b>4.1 Condition normalization</b><br/>KMeans regimes -> per-regime<br/>mean removal + within-regime std"]
    B --> C["<b>4.2 Sensor selection</b><br/>trend |corr(sensor, cycle)|"]
    C --> D["<b>4.3 Denoise</b><br/>rolling median, window 15"]
    D --> E["<b>4.4 k=2 health autoencoder</b><br/>sigmoid bottleneck +<br/>monotonicity + smoothness"]
    E --> F["<b>4.5 Rollout</b><br/>velocity extrapolation in (0,1)^k"]
    E --> G["<b>4.6 RUL</b><br/>HistGBR on (h0,h1,h0',h1')"]
```

### 4.1 Operating-condition normalization (the key generalization)

For multi-regime data we expose a *common-scale degradation residual*:

1. **Regime clustering.** Fit `KMeans(n_clusters = R, n_init = 10,
   random_state = 42)` on the three operating settings
   $(\text{setting}_1, \text{setting}_2, \text{setting}_3)$. $R = 1$ for
   FD001/FD003, $R = 6$ for FD002/FD004 (chosen from the known C-MAPSS regime
   count; the six clusters are cleanly separated).
2. **Per-regime mean removal.** For each sensor $j$ and regime $r$, subtract the
   regime-specific healthy mean $\mu_{j,r}$. This *kills the staircase*: what
   remains is deviation-from-nominal *within* the current flight condition.
3. **Pooled within-regime standardization.** Divide by the pooled
   within-regime standard deviation $s_j$, putting every sensor on a comparable
   scale.

$$
z_{t,j} \;=\; \frac{x_{t,j} - \mu_{j,\,c_t}}{s_j}.
$$

The autoencoder operates entirely in this normalized frame, so a single
architecture handles 1- and 6-regime datasets identically.

### 4.2 Trend-based sensor selection

Rather than hard-coding sensor lists per dataset, we score each sensor by the
mean over engines of $|\,\mathrm{corr}(\bar x_j, \text{cycle})\,|$ (Pearson, on
the denoised channel). Sensors above $\text{TREND}_\text{dynamic}=0.20$ are
*dynamic* (carry degradation); above $\text{TREND}_\text{informative}=0.50$ are
*informative*. A safety net guarantees $\ge 3$ dynamic sensors. This selected
**14 sensors for FD001 and 15 for FD002–FD004** automatically.

### 4.3 Denoising

Per-engine rolling **median**, window $w = 15$ (causal/trailing variant
wherever temporal leakage must be excluded, e.g. forecasting and test-time
features). Median is used over mean for robustness to sensor spikes.

### 4.4 The $k=2$ physics-constrained health autoencoder

A small MLP autoencoder with a **logistic bottleneck**:

$$
E_\theta:\; \mathbb{R}^m \xrightarrow{\,\text{Lin}(32)\,\tanh\,\text{Lin}(16)\,\tanh\,\text{Lin}(2)\,} \xrightarrow{\;\sigma\;} (0,1)^2,
\qquad
D_\phi:\; (0,1)^2 \xrightarrow{\,\text{Lin}(16)\,\tanh\,\text{Lin}(32)\,\tanh\,\text{Lin}(m)\,} \mathbb{R}^m .
$$

The sigmoid makes the latent **bounded by construction** — the cornerstone of
Theorem 2. Training minimizes a ceiling-weighted reconstruction loss plus two
physics penalties on the first health coordinate $h_0$:

$$
\mathcal{L} \;=\; \underbrace{\sum_j w_j\,\big\|x_{\cdot,j}-\hat x_{\cdot,j}\big\|^2}_{\text{weighted reconstruction}}
\;+\; \underbrace{\lambda_{\text{mono}}\sum_t \mathrm{ReLU}(-\Delta h_{0,t})}_{\text{monotonicity, }\lambda=5.0}
\;+\; \underbrace{\lambda_{\text{smooth}}\sum_t (\Delta h_{0,t})^2}_{\text{smoothness, }\lambda=2.0},
$$

where $\Delta h_{0,t}=h_{0,t}-h_{0,t-1}$. The monotonicity penalty pushes $h_0$
to increase with wear; the smoothness penalty bounds its curvature $\kappa$ —
exactly the quantity Theorem 3 needs. $h_0$ is oriented (flipped if needed) so
that it *increases* with cycle. Optimizer: Adam, $\text{lr}=5\!\times\!10^{-3}$,
full-batch, $4000$ epochs. The manifold is trained **once** per dataset and
cached to `experiments/artifacts/FD00<fd>/`.

### 4.5 Rollout (forward simulation)

From an anchor cycle $c$ with health velocity $\nu = h_c - h_{c-1}$, the
manifold rollout is a clipped constant-velocity extrapolation **in latent
space**, decoded back to sensors:

$$
h_t = \operatorname{clip}_{[0,1]}\!\big(h_c + (t-c)\,\nu\big), \qquad \hat x_t = D_\phi(h_t).
$$

We compare against the natural alternative — a **sensor-space VAR**
$z_{t+1}=A z_t + b$ fit by least squares. Section 5 shows why the VAR is
unstable and the manifold rollout is not.

### 4.6 Health$\to$RUL regression

The final estimator is a gradient-boosted tree
(`HistGradientBoostingRegressor`, `max_depth=3`, `learning_rate=0.05`,
`max_iter=400`, `l2_regularization=1.0`) mapping the four-dimensional feature
$(h_0, h_1, \dot h_0, \dot h_1)$ — health levels and their causal velocities —
to RUL, with the standard $\text{RUL}_\text{cap}=125$ piecewise-linear target.
We contrast it with a **naive threshold-crossing** estimator (forecast $h_0$ to
a learned failure threshold) to show that the *regression on the state*, not
the threshold heuristic, is what works.

---

## 5. Theory

All proofs are in [docs/THEORY.md](docs/THEORY.md); we state the results and
their cross-dataset empirical hooks here.

### Lemma 1 — Identifiability of a 2-D health manifold

If the denoised, condition-normalized sensors lie (up to per-coordinate noise
$\varepsilon^2$) on a $d$-dimensional curve, the cumulative PCA variance obeys

$$
\rho_d \;\ge\; 1 - \frac{(m-d)\,\varepsilon^2}{\sum_i \lambda_i}.
$$

With $d=2$ this is the identifiability premise. **Empirically**, $\rho_2$
ranges $0.795$ (FD002) to $0.965$ (FD001) — a 2-D health state is sufficient
even in the 6-regime case once the staircase is normalized away.

### Theorem 1 — Sensor-space autoregression is unstable (negative result)

For a free-running predictor $\hat z_{t+1}=F(\hat z_t)$ with Jacobian $J$, the
rollout error grows as

$$
\|e_t\| \sim \rho(J)^{\,t}\,\|e_0\|.
$$

If $\rho(J)>1$ the error is **unbounded**. **Empirically**, the fitted VAR has
$\rho(A) = 1.014\text{–}1.020 > 1$ on *all four* datasets, and free-runs to
$25\times$ (FD004) up to $760\times$ (FD001) its initial norm. (See the
free-run growth panel in the summary figure.)

### Theorem 2 — Manifold rollout is bounded (positive result)

With a logistic latent $h_t\in(0,1)^k$ and an $L_D$-Lipschitz decoder,

$$
\|\hat x_t\| \;\le\; \big\|D_\phi(\tfrac12\mathbf 1)\big\| + L_D\frac{\sqrt k}{2} \;=:\; B < \infty \quad \forall t.
$$

The decoded trajectory **cannot blow up at any horizon**. The contrast with
Theorem 1 is the central structural claim: *the VAR provably can diverge, the
manifold provably cannot.* Confirmed on all four datasets (the manifold
free-run trace stays flat while the VAR's explodes).

### Theorem 3 — Polynomial forecast error for a smooth, monotone health state

If $|\Delta^2 h_0|\le\kappa$, the constant-velocity forecast error obeys

$$
\big|h_{c+t}-\hat h_{c+t}\big| \;\le\; \frac{\kappa}{2}\,t^2 .
$$

Polynomial, never exponential. The smoothness penalty (Section 4.4) is what
makes $\kappa$ small: measured $\kappa \approx 2.9\text{–}3.7\times10^{-4}$
across datasets. This is why constant-velocity extrapolation **in health
space** beats persistence, while autoregression **in sensor space** explodes.

> **Corroborating evidence.** An unconstrained AR(2) forecaster (no curvature
> control) numerically explodes on the harder datasets — its skill collapses to
> $\sim\!-10^{13}$ on FD003. This is Theorem 1 playing out in miniature and is
> exactly the pathology the bounded, smooth health coordinate avoids.

### Theorem 4 — Irreducible $R^2$ ceiling

For $x_i=\bar x_i+\eta_i$ with noise variance $\sigma_i^2$,

$$
R^2_i \le 1 - \frac{\sigma_i^2}{\operatorname{Var}(x_i)}.
$$

This justifies dropping noise-limited channels during sensor selection
(Section 4.2) rather than forcing the autoencoder to model pure noise.

### Proposition 5 — A monotone health$\to$RUL map exists

If $h_0$ is strictly monotone in wear and degradation is stochastically
monotone toward $\{h_0\ge\tau\}$, then $g(\eta)=\mathbb E[\text{RUL}\mid h_0=\eta]$
is non-increasing, and RUL is identifiable from $(h_0,\dot h_0)$. **Empirically**
the binned $\mathbb E[\text{RUL}\mid h_0]$ curve is cleanly monotone, and the
supervised map (Section 4.6) realizes it.

### Logical chain

```mermaid
flowchart TD
    L1["Lemma 1: 2-D health identifiable (rho_2 = 0.80-0.97)"] --> T2
    T1["Theorem 1: VAR error ~ rho^t, rho>1 (UNBOUNDED)"]
    T2["Theorem 2: manifold rollout bounded by B (sigmoid + Lipschitz)"]
    T1 -. "decisive contrast" .-> T2
    T2 --> T3["Theorem 3: const-velocity error <= (kappa/2) t^2 (POLYNOMIAL)"]
    T3 --> P5["Proposition 5: monotone health -> RUL map exists"]
    T4["Theorem 4: R2 ceiling = 1 - sigma^2/Var"] --> P5
    P5 --> C["RUL: R2 = 0.74-0.88 on all four datasets"]
```

---

## 6. Experiments

Four experiments, one per pipeline stage, run identically on each dataset:

| Exp | Question | Script | Key output |
|---|---|---|---|
| **0 — Discovery** | Is a 2-D health state identifiable? | `exp_discovery.py` | PCA $\rho_2$, recon $R^2$; figs D1–D2 |
| **A — Rollout stability** | Is forward simulation bounded? | `exp_rollout_stability.py` | $\rho(A)$, free-run growth; figs A1–A4 |
| **B — Health forecasting** | Can health be forecast? | `exp_health_forecasting.py` | skill vs persistence; figs B1–B3 |
| **C — RUL prediction** | Does it predict RUL? | `exp_rul_prediction.py` | RMSE, $R^2$, NASA score; figs C1–C3 |

All four share `experiments/manifold_core.py`, which performs condition
normalization, sensor selection, and trains/caches the manifold once per
dataset. The single driver `experiments/run_all.py` runs the whole grid.

---

## 7. Results — per stage

The figures below are shown for FD001 (the cleanest case); the identical figure
set for every dataset lives in `docs/figures/FD00<fd>/`.

### 7.1 Discovery

A single normalized operating condition collapses degradation onto a 2-D
manifold; the constrained autoencoder recovers a monotone health trajectory.

![Health trajectories](docs/figures/FD001/D1_health_trajectories.png)
![Manifold + PCA scree](docs/figures/FD001/D2_manifold.png)

### 7.2 Rollout stability (the headline structural result)

The VAR eigenvalues sit outside the unit circle ($\rho>1$) and the free run
diverges geometrically; the manifold free-run stays flat — exactly the
Theorem 1 vs Theorem 2 dichotomy.

![VAR eigenvalues](docs/figures/FD001/A2_var_eigenvalues.png)
![Free-run divergence](docs/figures/FD001/A4_free_run_divergence.png)

> **Honest caveat (read Section 9).** At the *short, scored* horizons
> ($h\le 25$) the VAR is a near-perfect 1-step predictor on the smooth denoised
> trend and is comparable to (sometimes better than) the manifold; the
> autoencoder pays a fixed $\sim$5–7% reconstruction-floor NRMSE at every
> horizon. The manifold's advantage is **provable boundedness over long
> horizons**, not short-horizon accuracy.

### 7.3 Health forecasting

Constant-velocity extrapolation in health space stays under the
$\tfrac{\kappa}{2}t^2$ envelope and beats persistence.

![Error vs horizon](docs/figures/FD001/B2_error_vs_horizon.png)
![Skill vs persistence](docs/figures/FD001/B3_skill_vs_persistence.png)

### 7.4 RUL prediction

The binned $\mathbb E[\text{RUL}\mid h_0]$ map is monotone; the supervised
regressor on $(h_0,h_1,\dot h_0,\dot h_1)$ predicts RUL well below the baseline,
while the naive threshold-crossing estimator fails.

![RUL scatter](docs/figures/FD001/C1_rul_scatter.png)
![Health vs RUL](docs/figures/FD001/C3_health_vs_rul.png)

---

## 8. Cross-dataset results (FD001–FD004)

The complete grid, produced by `experiments/run_all.py`
(`experiments/artifacts/cross_dataset_results.csv`):

![Cross-dataset summary](docs/figures/SUMMARY_cross_dataset.png)

### 8.1 Master results table

| Metric | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| Operating regimes $R$ | 1 | 6 | 1 | 6 |
| Fault modes | 1 | 1 | 2 | 2 |
| Train / test units | 100 / 100 | 260 / 259 | 100 / 100 | 249 / 248 |
| Informative sensors | 14 | 15 | 15 | 15 |
| **Discovery** PCA $\rho_2$ | 0.965 | 0.795 | 0.851 | 0.849 |
| **Discovery** recon mean $R^2$ | 0.930 | 0.865 | 0.960 | 0.930 |
| **Rollout** VAR $\rho(A)$ | 1.020 | 1.016 | 1.018 | 1.015 |
| **Rollout** VAR free-run growth | $760\times$ | $468\times$ | $58\times$ | $25\times$ |
| **Rollout** manifold bounded? | ✓ | ✓ | ✓ | ✓ |
| **Forecast** skill vs persistence ($k{=}20$) | $+0.751$ | $+0.680$ | $+0.522$ | $+0.157$ |
| **RUL** RMSE | **14.53** | **27.02** | **16.31** | **27.58** |
| **RUL** MAE | 11.02 | 18.61 | 12.44 | 20.19 |
| **RUL** $R^2$ | **0.878** | **0.748** | **0.845** | **0.744** |
| **RUL** NASA score | 380.8 | 8692 | 615 | 6088 |
| Mean-RUL baseline RMSE | 43.07 | 54.08 | 45.07 | 54.90 |
| Naive threshold-crossing RMSE | 39.46 | 39.64 | 87.14 | 102.29 |

### 8.2 What the table says

1. **Health$\to$RUL beats the mean baseline on every dataset** — by $3\times$ on
   the easy corner (FD001: $14.5$ vs $43.1$) and still by $2\times$ on the hard
   corner (FD004: $27.6$ vs $54.9$).
2. **The boundedness dichotomy holds universally.** The VAR has $\rho(A)>1$ and
   free-runs to $25$–$760\times$ on every dataset; the manifold rollout is
   bounded on every dataset.
3. **Difficulty tracks the regime$\times$fault grid.** $R^2$ degrades
   monotonically from the easy corner (FD001, $0.878$) to the hard corner
   (FD004, $0.744$), and forecasting skill drops from $+0.75$ to $+0.16$ —
   exactly as expected when six flight conditions and two fault modes are
   superimposed. The method still works; it is simply harder.
4. **The threshold heuristic is not what carries the result.** Naive
   threshold-crossing is *worse than the mean baseline* on FD003/FD004
   ($87$–$102$ RMSE) — the supervised regression on the state is doing the work.

---

## 9. Discussion

**Where the manifold wins, and where it doesn't.** Our strongest, *provable*
claim is structural: the manifold rollout is bounded for all horizons while the
sensor-space VAR is not. At the short horizons used for scoring, however, the
VAR is a strong one-step predictor and the autoencoder's reconstruction floor
(a $15\!\to\!2\!\to\!15$ bottleneck leaves $\sim$7% residual variance) costs a
roughly constant NRMSE offset at every horizon. So the honest framing is:
*the manifold buys interpretability and long-horizon stability, not a free
lunch on 1-step accuracy.*

**Multi-regime difficulty.** FD002/FD004 are harder because the degradation
residual must be extracted from six superimposed flight conditions. Condition
normalization (Section 4.1) is what makes the single architecture work at all
here; without it the staircase dominates PCA and the 2-D assumption fails.

**AR(2) blow-up corroborates the theory.** The unconstrained AR(2) forecaster's
numerical explosion on FD003/FD004 is not a bug to hide — it is Theorem 1 in
miniature, and it is precisely the failure mode the bounded, smoothness-penalized
health coordinate is designed to avoid.

---

## 10. Limitations and future work

- **Reconstruction floor.** The 2-D bottleneck caps short-horizon rollout
  accuracy. An optional **anchoring/bias correction** — adding the constant
  reconstruction residual $r_c = x_c - D_\phi(h_c)$ to every decoded step —
  would remove the fixed offset without touching the boundedness guarantee.
  (Not implemented here; left as the obvious next experiment.)
- **Regime count is supplied.** We use the known $R$ (1 or 6). A model-selection
  pass (e.g. silhouette / BIC over $R$) would make the pipeline fully
  unsupervised in the operating dimension.
- **NASA score sensitivity.** The asymmetric NASA score is dominated by a few
  late-prediction engines on the multi-regime sets; quantile or
  uncertainty-aware RUL heads are a natural extension of Proposition 5.

---

## 11. Reproducibility

```powershell
# from the project root, using the project venv
cd experiments
..\.venv\Scripts\python.exe run_all.py            # FD001..FD004: discovery + A + B + C
..\.venv\Scripts\python.exe make_summary_figure.py  # docs/figures/SUMMARY_cross_dataset.png
```

`run_all.py` calls `manifold_core.configure(fd, retrain=True)` for each dataset,
runs all four experiments, and writes
`experiments/artifacts/cross_dataset_results.{csv,json}`. Every figure is
regenerated into `docs/figures/FD00<fd>/`. Fixed seed `SEED = 42` throughout;
key constants `WINDOW = 15`, `K = 2`, `RUL_CAP = 125`,
`TREND_dynamic = 0.20`, `TREND_informative = 0.50`,
$\lambda_\text{mono}=5.0$, $\lambda_\text{smooth}=2.0$.

### Repository layout

| Path | Contents |
|---|---|
| `experiments/manifold_core.py` | shared core: normalization, selection, manifold |
| `experiments/exp_discovery.py` | Experiment 0 (discovery) |
| `experiments/exp_rollout_stability.py` | Experiment A (stability) |
| `experiments/exp_health_forecasting.py` | Experiment B (forecasting) |
| `experiments/exp_rul_prediction.py` | Experiment C (RUL) |
| `experiments/run_all.py` | cross-dataset driver |
| `experiments/make_summary_figure.py` | summary figure |
| `docs/THEORY.md` | full theorems + proofs |
| `docs/{ROLLOUT_STABILITY,HEALTH_FORECASTING,RUL_PREDICTION}.md` | per-experiment writeups |
| `docs/figures/FD00<fd>/` | per-dataset figures |
| `docs/figures/SUMMARY_cross_dataset.png` | cross-dataset summary |

---

## 12. Conclusion

We built a turbofan prognostic around a *discovered, physics-constrained* health
manifold rather than a black-box regressor. The latent is bounded and monotone
by design, which turns "stable forward simulation" into a theorem (Theorem 2)
that the natural sensor-space alternative provably fails (Theorem 1). The smooth
health coordinate is forecastable with polynomial error (Theorem 3) and carries
a monotone map to RUL (Proposition 5). Crucially, the whole pipeline generalizes
from the single-regime FD001 to the six-regime, two-fault FD002–FD004 through a
single operating-condition normalization, and **beats a mean-RUL baseline on all
four C-MAPSS datasets** ($R^2 = 0.744$–$0.878$). The contribution is not a new
state-of-the-art RMSE; it is a prognostic whose internal dynamics are
*provably* well-behaved and *empirically* portable across operating regimes and
fault modes.
