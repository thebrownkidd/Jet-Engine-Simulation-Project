    # cmapss_FD001 — Experiment Report

    > **Dataset:** NASA C-MAPSS FD001 — 100 simulated jet engines, single flight condition, monotone degradation to failure.

    ---

    ## 1. What this experiment does

    We apply a **Context-Conditioned Bounded Latent Dynamics** model to this
    dataset. The core idea:

    1. An **autoencoder** squeezes the 14-variable sensor stream into
       a small **k=2** latent state.  Each latent number is forced to stay
       between 0 and 1 (bounded), so predictions can never blow up.
    2. A small **residual MLP** then learns how the latent state steps forward in
       time.  If context columns are available (calendar time, operating settings)
       it conditions on those too.
    3. Every prediction is projected back onto the [0,1] box, then decoded back
       into the original sensor space so we can compare predicted vs actual sensor
       values directly.

    Five forecasting methods are compared:

    | Label | What it does |
    |---|---|
    | **persistence** | Just repeats the last observed value forever. The laziest possible forecast. Beats it = good. |
    | **cv** | Constant-velocity rollout: extrapolates the recent trend in the latent space. The old method. |
    | **var_sensor** | Fits a linear autoregression directly on the sensors (no latent compression). |
    | **mlp_noctx** | Bounded residual MLP in latent space, no context/season/regime information. |
    | **mlp_ctx** | Bounded residual MLP in latent space, **conditioned on context features** (proposed). |

    ---

    ## 2. Methodology and parameters

    | Parameter | Value |
    |---|---|
    | Latent dimension k | **2** (chosen by k-sweep, see Section 3) |
    | AE epochs | 800 |
    | Dynamics epochs | 400 |
    | Seeds for MLP heads | [0, 1, 2] |
    | Monotonicity penalty λ_mono | 1.0 |
    | Smoothness penalty λ_smooth | 0.5 |
    | Forecast horizons evaluated | [1, 10, 25, 50] (in cycles/steps) |
    | Train / test split | 70 / 30 units |
    | Context columns used | ['ctx_setting_1', 'ctx_setting_2', 'ctx_setting_3'] |
    | Sensors (raw → selected) | 15 → 14 |
    | Denoising window | 15 cycles (causal rolling median) |
    | Sensor selection threshold | trend |corr(sensor,cycle)| ≥ 0.2 |

    **Preprocessing pipeline (applied before any model training):**
    1. **Train/test split** — units split first so no preprocessing leaks across the boundary.
    2. **Denoising** — causal rolling-median (window=15 cycles) applied per unit per sensor.
       Removes cycle-to-cycle noise without looking ahead; matches the validated `manifold` pipeline.
    3. **Sensor selection** — sensors with mean |corr(sensor, cycle)| < 0.2 across training units are dropped. These sensors carry no degradation signal and only add noise to the latent space. Selected: 14/15 sensors.

    **Why k=2?**
    The k-sweep trains the autoencoder at each dimension and applies the **PCA elbow method**
    to find the latent dimension that captures the data's intrinsic structure without overfitting.
    The elbow is found using two criteria:
      1. The smallest k where cumulative explained variance ≥ 85%
      2. The k where incremental variance gain drops below 5%

    We take the **more conservative choice** (smaller k). For k=2, the reconstruction
    R² (mean across sensors) = **0.955** (worst single sensor = 0.888).

    **Why PCA elbow, not max reconstruction R²?**
    Reconstruction R² always rises with k, so choosing max R² would lead to overfitting
    (using many more latent dimensions than necessary). The PCA elbow method finds the
    "natural" dimensionality of the problem, a much more principled approach.

    **Why λ_mono=1.0?**
    The dataset has a degradation trend, so we encourage the primary latent coordinate to increase monotonically with time.  This is the *health progressing toward failure* assumption.

    ---

    ## 3. K-sweep results

    This table shows how reconstruction quality changes as we give the latent
    space more dimensions.  **recon_r2** is the mean R² across all sensors;
    **recon_r2_min** is the worst single sensor.

    |   k |   recon_r2 |   recon_r2_min |
|----:|-----------:|---------------:|
|   1 |     0.805  |         0.032  |
|   2 |     0.9554 |         0.8878 |
|   3 |     0.9581 |         0.892  |
|   4 |     0.957  |         0.8918 |
|   5 |     0.9579 |         0.8929 |
|   6 |     0.9579 |         0.8931 |

    A higher recon_r2 means the compressed representation is more faithful.
    Diminishing returns usually set in around k=3–5; we pick the elbow point
    (best recon without overfitting capacity).

    ---

    ## 4. Forecasting comparison

    **Skill vs persistence** measures "how much better than just repeating the
    last value are we?"  A skill of +0.5 means our error is half of the
    persistence error.  A skill of 0 means we tied persistence.  Negative means
    we were *worse* than doing nothing.

    **NRMSE** is the root-mean-squared forecast error expressed in standard
    deviation units of the training data, so a value of 0.5 means the average
    error is about half a standard deviation.

    **freerun_growth** is how many times larger the forecast output got over a
    long free run starting from the test cutoff.  A value near 1 is good (stable
    forecast).  Large values mean the forecast exploded.

    **bounded** = True means the growth stayed below the threshold of
    5.0×, which we define as "bounded" for reporting.

    | model       |   skill_h1 |   nrmse_h1 |   skill_h10 |   nrmse_h10 |   skill_h25 |   nrmse_h25 |   skill_h50 |   nrmse_h50 |   freerun_growth | bounded   |
|:------------|-----------:|-----------:|------------:|------------:|------------:|------------:|------------:|------------:|-----------------:|:----------|
| persistence |     0      |     0.1169 |      0      |      0.2784 |      0      |      0.5082 |      0      |      1.1583 |           1      | True      |
| cv          |    -1.9907 |     0.2022 |      0.2965 |      0.2335 |      0.6351 |      0.307  |      0.66   |      0.6754 |           1.7765 | True      |
| var_sensor  |     0.0606 |     0.1133 |      0.3536 |      0.2239 |      0.5983 |      0.3221 |      0.6526 |      0.6827 |           8.894  | False     |
| mlp_noctx   |    -1.988  |     0.2021 |      0.3199 |      0.2296 |      0.6594 |      0.2966 |      0.7448 |      0.5849 |           5.9177 | False     |
| mlp_ctx     |    -1.9874 |     0.2021 |      0.3216 |      0.2293 |      0.652  |      0.2998 |      0.7383 |      0.5925 |           6.0689 | False     |

    **Winner:** `mlp_noctx` (highest mean skill at horizons [10, 25, 50]).

    ---

    ## 5. Observation-space accuracy (best head: `mlp_noctx`)

    This table shows the per-sensor forecast accuracy in the **original feature
    scale** (regime-normalized z-scores or raw feature units depending on the
    dataset).  Averaged over the three anchor points (50%, 65%, 80% of each
    test trajectory) and all test units.

    **rmse**: root-mean-squared error in original feature units.
    **r2**: coefficient of determination; 1.0 = perfect, 0 = no better than the
    mean, negative = worse than the mean.

    Values are averaged over all forecast horizons ([1, 10, 25, 50]).

    | sensor   |   rmse |     r2 |
|:---------|-------:|-------:|
| s11      | 0.2365 | 0.8279 |
| s12      | 0.2548 | 0.8085 |
| s13      | 0.2595 | 0.8477 |
| s14      | 0.2298 | 0.9104 |
| s15      | 0.2492 | 0.7996 |
| s17      | 0.2998 | 0.7065 |
| s2       | 0.2487 | 0.7902 |
| s20      | 0.2438 | 0.8153 |
| s21      | 0.2583 | 0.7827 |
| s3       | 0.2818 | 0.7275 |
| s4       | 0.2385 | 0.8312 |
| s7       | 0.2551 | 0.7964 |
| s8       | 0.2697 | 0.8353 |
| s9       | 0.2226 | 0.9094 |

    ---

    ## 6. Figures: Reconstruction and Forecast

    ### Figure (a): Reconstruction — `recon_plot.png`

    ![Reconstruction figure](recon_plot.png)

    This figure shows how faithfully the bounded autoencoder can **rebuild** the
    sensor values from the k=2 latent numbers, using the full observed
    trajectory of the longest test unit.

    | Element | What it means |
    |---|---|
    | **Blue solid line** | Ground truth — actual recorded sensor values |
    | **Green dashed line** | AE reconstruction — what the bounded AE rebuilds from the 2 latent numbers |

    If the green line follows the blue line closely, the k=2 latent space
    captures the data well. Large gaps mean information was lost in compression.
    This is a "no-forecast" check: can the model even represent the data?

    ---

    ### Figure (b): Forecast — `forecast_plot.png`

    ![Forecast figure](forecast_plot.png)

    This figure shows the **predictions** made by the best-head model (`mlp_noctx`)
    starting from a mid-to-late point through the trajectory of the longest test unit.

    | Element | What it means |
    |---|---|
    | **Blue solid line** | Ground truth — the actual sensor values after the cutoff (what really happened) |
    | **Red solid line** | Forecast — the model's predicted future values |

    **Cutoff strategy:** The forecast starts at the later of (1) 50% through the trajectory
    or (2) 2× the maximum forecast horizon before the end. This ensures you see a
    substantial forecast window (at least 2× max_horizon steps) to evaluate where
    predictions diverge from reality. If the red line stays close to the blue line,
    the model can predict well. If they diverge sharply, explode, or flatten, the
    forecast is unreliable.

    ---

    ## 7. Observation-space accuracy (best head: `mlp_noctx`)

    This table shows the per-sensor forecast accuracy in the **original feature
    scale** (regime-normalized z-scores or raw feature units depending on the
    dataset).  Averaged over the three anchor points (50%, 65%, 80% of each
    test trajectory) and all test units.

    **rmse**: root-mean-squared error in original feature units.
    **r2**: coefficient of determination; 1.0 = perfect, 0 = no better than the
    mean, negative = worse than the mean.

    Values are averaged over all forecast horizons ([1, 10, 25, 50]).

    | sensor   |   rmse |     r2 |
|:---------|-------:|-------:|
| s11      | 0.2365 | 0.8279 |
| s12      | 0.2548 | 0.8085 |
| s13      | 0.2595 | 0.8477 |
| s14      | 0.2298 | 0.9104 |
| s15      | 0.2492 | 0.7996 |
| s17      | 0.2998 | 0.7065 |
| s2       | 0.2487 | 0.7902 |
| s20      | 0.2438 | 0.8153 |
| s21      | 0.2583 | 0.7827 |
| s3       | 0.2818 | 0.7275 |
| s4       | 0.2385 | 0.8312 |
| s7       | 0.2551 | 0.7964 |
| s8       | 0.2697 | 0.8353 |
| s9       | 0.2226 | 0.9094 |

    ---

    ## 8. Honest summary

    ### What worked
    - Reconstruction R² at the chosen k = **0.955**.
- Best head (`mlp_noctx`) achieves positive forecast skill at h=10: **+0.320** (beats persistence).

    ### What did not work / limitations
    - **Short-horizon reconstruction tax:** at h=1, skill = -1.988.  Squeezing to k latent numbers discards small-scale detail that matters for very-next-step predictions.  Persistence wins here.

    ---

    *Generated automatically by experiments/acml/exp_all_datasets.py*
