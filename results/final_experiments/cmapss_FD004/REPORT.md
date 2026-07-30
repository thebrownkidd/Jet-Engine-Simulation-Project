    # cmapss_FD004 — Experiment Report

    > **Dataset:** NASA C-MAPSS FD004 — 249 simulated jet engines, six flight conditions, two fault modes.

    ---

    ## 1. What this experiment does

    We apply a **Context-Conditioned Bounded Latent Dynamics** model to this
    dataset. The core idea:

    1. An **autoencoder** squeezes the 16-variable sensor stream into
       a small **k=3** latent state.  Each latent number is forced to stay
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
    | Latent dimension k | **3** (chosen by k-sweep, see Section 3) |
    | AE epochs | 800 |
    | Dynamics epochs | 400 |
    | Seeds for MLP heads | [0, 1, 2] |
    | Monotonicity penalty λ_mono | 1.0 |
    | Smoothness penalty λ_smooth | 0.5 |
    | Forecast horizons evaluated | [1, 10, 25, 50] (in cycles/steps) |
    | Train / test split | 174 / 75 units |
    | Context columns used | ['ctx_setting_1', 'ctx_setting_2', 'ctx_setting_3'] |
    | Sensors (raw → selected) | 17 → 16 |
    | Denoising window | 15 cycles (causal rolling median) |
    | Sensor selection threshold | trend |corr(sensor,cycle)| ≥ 0.2 |

    **Preprocessing pipeline (applied before any model training):**
    1. **Train/test split** — units split first so no preprocessing leaks across the boundary.
    2. **Denoising** — causal rolling-median (window=15 cycles) applied per unit per sensor.
       Removes cycle-to-cycle noise without looking ahead; matches the validated `manifold` pipeline.
    3. **Sensor selection** — sensors with mean |corr(sensor, cycle)| < 0.2 across training units are dropped. These sensors carry no degradation signal and only add noise to the latent space. Selected: 16/17 sensors.

    **Why k=3?**
    The k-sweep trains the autoencoder at each dimension and applies the **PCA elbow method**
    to find the latent dimension that captures the data's intrinsic structure without overfitting.
    The elbow is found using two criteria:
      1. The smallest k where cumulative explained variance ≥ 85%
      2. The k where incremental variance gain drops below 5%

    We take the **more conservative choice** (smaller k). For k=3, the reconstruction
    R² (mean across sensors) = **0.849** (worst single sensor = 0.527).

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
|   1 |     0.5173 |         0.1757 |
|   2 |     0.8745 |         0.5853 |
|   3 |     0.8488 |         0.5271 |
|   4 |     0.9144 |         0.8198 |
|   5 |     0.9172 |         0.8296 |
|   6 |     0.9119 |         0.6762 |

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
| persistence |     0      |     0.1222 |      0      |      0.3098 |      0      |      0.4734 |      0      |      0.949  |           1      | True      |
| cv          |    -2.0523 |     0.2135 |      0.1163 |      0.2913 |      0.2503 |      0.4099 |      0.3007 |      0.7935 |           1.2601 | True      |
| var_sensor  |     0.0678 |     0.118  |      0.3519 |      0.2494 |      0.5543 |      0.316  |      0.5637 |      0.6268 |           5.8823 | False     |
| mlp_noctx   |    -2.0479 |     0.2134 |      0.3205 |      0.2554 |      0.5894 |      0.3033 |      0.6262 |      0.5802 |           3.86   | True      |
| mlp_ctx     |    -2.051  |     0.2135 |      0.3249 |      0.2546 |      0.5905 |      0.3029 |      0.6229 |      0.5827 |           3.8388 | True      |

    **Winner:** `mlp_ctx` (highest mean skill at horizons [10, 25, 50]).

    ---

    ## 5. Observation-space accuracy (best head: `mlp_ctx`)

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
| s10      | 0.3018 | 0.8011 |
| s11      | 0.247  | 0.8674 |
| s12      | 0.2146 | 0.9091 |
| s13      | 0.3411 | 0.7747 |
| s14      | 0.2741 | 0.8812 |
| s15      | 0.1866 | 0.9621 |
| s17      | 0.2687 | 0.8124 |
| s2       | 0.2632 | 0.8481 |
| s20      | 0.2604 | 0.8598 |
| s21      | 0.2439 | 0.8744 |
| s3       | 0.2658 | 0.8003 |
| s4       | 0.2545 | 0.8524 |
| s6       | 0.1521 | 0.9746 |
| s7       | 0.2261 | 0.902  |
| s8       | 0.3347 | 0.7817 |
| s9       | 0.2664 | 0.8699 |

    ---

    ## 6. Figures: Reconstruction and Forecast

    ### Figure (a): Reconstruction — `recon_plot.png`

    ![Reconstruction figure](recon_plot.png)

    This figure shows how faithfully the bounded autoencoder can **rebuild** the
    sensor values from the k=3 latent numbers, using the full observed
    trajectory of the longest test unit.

    | Element | What it means |
    |---|---|
    | **Blue solid line** | Ground truth — actual recorded sensor values |
    | **Green dashed line** | AE reconstruction — what the bounded AE rebuilds from the 3 latent numbers |

    If the green line follows the blue line closely, the k=3 latent space
    captures the data well. Large gaps mean information was lost in compression.
    This is a "no-forecast" check: can the model even represent the data?

    ---

    ### Figure (b): Forecast — `forecast_plot.png`

    ![Forecast figure](forecast_plot.png)

    This figure shows the **predictions** made by the best-head model (`mlp_ctx`)
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

    ## 7. Observation-space accuracy (best head: `mlp_ctx`)

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
| s10      | 0.3018 | 0.8011 |
| s11      | 0.247  | 0.8674 |
| s12      | 0.2146 | 0.9091 |
| s13      | 0.3411 | 0.7747 |
| s14      | 0.2741 | 0.8812 |
| s15      | 0.1866 | 0.9621 |
| s17      | 0.2687 | 0.8124 |
| s2       | 0.2632 | 0.8481 |
| s20      | 0.2604 | 0.8598 |
| s21      | 0.2439 | 0.8744 |
| s3       | 0.2658 | 0.8003 |
| s4       | 0.2545 | 0.8524 |
| s6       | 0.1521 | 0.9746 |
| s7       | 0.2261 | 0.902  |
| s8       | 0.3347 | 0.7817 |
| s9       | 0.2664 | 0.8699 |

    ---

    ## 8. Honest summary

    ### What worked
    - Reconstruction R² at the chosen k = **0.849**.
- Best head (`mlp_ctx`) achieves positive forecast skill at h=10: **+0.325** (beats persistence).
- The best-head free-run forecast stays bounded (growth = 3.84×).

    ### What did not work / limitations
    - **Short-horizon reconstruction tax:** at h=1, skill = -2.051.  Squeezing to k latent numbers discards small-scale detail that matters for very-next-step predictions.  Persistence wins here.

    ---

    *Generated automatically by experiments/acml/exp_all_datasets.py*
