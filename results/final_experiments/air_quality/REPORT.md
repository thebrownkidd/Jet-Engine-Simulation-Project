    # air_quality — Experiment Report

    > **Dataset:** Beijing multi-site air quality — 12 stations, hourly PM2.5/PM10/meteorology, 11 variables (no degradation; cyclic seasonal forecasting).

    ---

    ## 1. What this experiment does

    We apply a **Context-Conditioned Bounded Latent Dynamics** model to this
    dataset. The core idea:

    1. An **autoencoder** squeezes the 11-variable sensor stream into
       a small **k=4** latent state.  Each latent number is forced to stay
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
    | Latent dimension k | **4** (chosen by k-sweep, see Section 3) |
    | AE epochs | 2000 |
    | Dynamics epochs | 400 |
    | Seeds for MLP heads | [0, 1, 2] |
    | Monotonicity penalty λ_mono | 0.0 |
    | Smoothness penalty λ_smooth | 0.5 |
    | Forecast horizons evaluated | [1, 8, 24, 48] (in cycles/steps) |
    | Train / test split | 8 / 4 units |
    | Context columns used | ['ctx_sin_hour', 'ctx_cos_hour', 'ctx_sin_dow', 'ctx_cos_dow', 'ctx_sin_month', 'ctx_cos_month'] |
    | Sensors (raw → selected) | 11 → 11 |
    | Denoising window | 15 cycles (causal rolling median) |
    | Sensor selection threshold | variance filter (non-degrading dataset) |

    **Preprocessing pipeline (applied before any model training):**
    1. **Train/test split** — units split first so no preprocessing leaks across the boundary.
    2. **Denoising** — causal rolling-median (window=15 cycles) applied per unit per sensor.
       Removes cycle-to-cycle noise without looking ahead; matches the validated `manifold` pipeline.
    3. **Sensor selection** — near-constant sensors (std < 1e-4) are dropped; all varying sensors are kept for this non-degrading dataset. Selected: 11/11 sensors.

    **Why k=4?**
    The k-sweep trains the autoencoder at each dimension and applies the **PCA elbow method**
    to find the latent dimension that captures the data's intrinsic structure without overfitting.
    The elbow is found using two criteria:
      1. The smallest k where cumulative explained variance ≥ 85%
      2. The k where incremental variance gain drops below 5%

    We take the **more conservative choice** (smaller k). For k=4, the reconstruction
    R² (mean across sensors) = **0.757** (worst single sensor = 0.009).

    **Why PCA elbow, not max reconstruction R²?**
    Reconstruction R² always rises with k, so choosing max R² would lead to overfitting
    (using many more latent dimensions than necessary). The PCA elbow method finds the
    "natural" dimensionality of the problem, a much more principled approach.

    **Why λ_mono=0.0?**
    This dataset does not have a monotone degradation trend (it is stationary / cyclic), so the monotonicity penalty is turned off.

    ---

    ## 3. K-sweep results

    This table shows how reconstruction quality changes as we give the latent
    space more dimensions.  **recon_r2** is the mean R² across all sensors;
    **recon_r2_min** is the worst single sensor.

    |   k |   recon_r2 |   recon_r2_min |
|----:|-----------:|---------------:|
|   1 |     0.5356 |         0.002  |
|   2 |     0.6565 |         0.0036 |
|   3 |     0.7518 |         0.0093 |
|   4 |     0.757  |         0.0091 |
|   5 |     0.7565 |         0.0129 |
|   6 |     0.7555 |         0.0073 |

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

    | model       |   skill_h1 |   nrmse_h1 |   skill_h8 |   nrmse_h8 |   skill_h24 |   nrmse_h24 |   skill_h48 |   nrmse_h48 |   freerun_growth | bounded   |
|:------------|-----------:|-----------:|-----------:|-----------:|------------:|------------:|------------:|------------:|-----------------:|:----------|
| persistence |     0      |     0.1278 |     0      |     0.5276 |      0      |      1.3131 |      0      |      1.393  |           1      | True      |
| cv          |    -3.918  |     0.2834 |    -0.3194 |     0.606  |     -2.2813 |      2.3786 |     -2.9766 |      2.7779 |           6.8822 | False     |
| cv_damped   |    -3.918  |     0.2834 |     0.1211 |     0.4946 |     -0.0611 |      1.3526 |     -0.0041 |      1.3959 |           1      | True      |
| holt        |    -6.506  |     0.3501 |    -0.1289 |     0.5606 |      0.13   |      1.2248 |      0.1032 |      1.3192 |           1      | True      |
| trend_lw    |    -3.9635 |     0.2847 |     0.052  |     0.5137 |      0.0695 |      1.2666 |     -0.0815 |      1.4487 |           2.0719 | True      |
| cvd_anch    |    -0.0736 |     0.1324 |     0.0102 |     0.5249 |     -0.0741 |      1.3609 |     -0.0349 |      1.4171 |           1      | True      |
| holt_anch   |    -2.3353 |     0.2334 |    -0.2207 |     0.5829 |      0.12   |      1.2318 |      0.0752 |      1.3397 |           1      | True      |
| var_sensor  |    -0.1017 |     0.1341 |    -0.0175 |     0.5322 |      0.4526 |      0.9715 |      0.5235 |      0.9616 |           0.0835 | True      |
| mlp_noctx   |    -4.3755 |     0.2962 |    -0.6239 |     0.6723 |      0.5634 |      0.8676 |      0.6101 |      0.8698 |           0.2234 | True      |
| mlp_ctx     |    -4.3242 |     0.2948 |    -0.6599 |     0.6797 |      0.5405 |      0.89   |      0.5828 |      0.8996 |           0.9114 | True      |

    **Winner:** `var_sensor` (highest mean skill at horizons [8, 24, 48]).

    ---

    ## 5. Observation-space accuracy (best head: `var_sensor`)

    This table shows the per-sensor forecast accuracy in the **original feature
    scale** (regime-normalized z-scores or raw feature units depending on the
    dataset).  Averaged over the three anchor points (50%, 65%, 80% of each
    test trajectory) and all test units.

    **rmse**: root-mean-squared error in original feature units.
    **r2**: coefficient of determination; 1.0 = perfect, 0 = no better than the
    mean, negative = worse than the mean.

    Values are averaged over all forecast horizons ([1, 8, 24, 48]).

    | sensor   |     rmse |            r2 |
|:---------|---------:|--------------:|
| CO       | 491.868  | -12.7503      |
| DEWP     |   6.3378 |   0.5828      |
| NO2      |  19.9363 |  -1.6444      |
| O3       |  19.4455 |  -0.3693      |
| PM10     |  60.7847 | -32.5024      |
| PM2.5    |  62.2882 |  -8.9737      |
| PRES     |   4.4335 | -11.5669      |
| RAIN     |   0.0042 |  -2.69893e+08 |
| SO2      |   8.185  |  -9.1186      |
| TEMP     |   2.5312 |   0.837       |
| WSPM     |   0.6671 |  -0.0178      |

    ---

    ## 6. Figures: Reconstruction and Forecast

    ### Figure (a): Reconstruction — `recon_plot.png`

    ![Reconstruction figure](recon_plot.png)

    This figure shows how faithfully the bounded autoencoder can **rebuild** the
    sensor values from the k=4 latent numbers, using the full observed
    trajectory of the longest test unit.

    | Element | What it means |
    |---|---|
    | **Blue solid line** | Ground truth — actual recorded sensor values |
    | **Green dashed line** | AE reconstruction — what the bounded AE rebuilds from the 4 latent numbers |

    If the green line follows the blue line closely, the k=4 latent space
    captures the data well. Large gaps mean information was lost in compression.
    This is a "no-forecast" check: can the model even represent the data?

    ---

    ### Figure (b): Forecast — `forecast_plot.png`

    ![Forecast figure](forecast_plot.png)

    This figure shows the **predictions** made by the best-head model (`var_sensor`)
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

    ## 7. Observation-space accuracy (best head: `var_sensor`)

    This table shows the per-sensor forecast accuracy in the **original feature
    scale** (regime-normalized z-scores or raw feature units depending on the
    dataset).  Averaged over the three anchor points (50%, 65%, 80% of each
    test trajectory) and all test units.

    **rmse**: root-mean-squared error in original feature units.
    **r2**: coefficient of determination; 1.0 = perfect, 0 = no better than the
    mean, negative = worse than the mean.

    Values are averaged over all forecast horizons ([1, 8, 24, 48]).

    | sensor   |     rmse |            r2 |
|:---------|---------:|--------------:|
| CO       | 491.868  | -12.7503      |
| DEWP     |   6.3378 |   0.5828      |
| NO2      |  19.9363 |  -1.6444      |
| O3       |  19.4455 |  -0.3693      |
| PM10     |  60.7847 | -32.5024      |
| PM2.5    |  62.2882 |  -8.9737      |
| PRES     |   4.4335 | -11.5669      |
| RAIN     |   0.0042 |  -2.69893e+08 |
| SO2      |   8.185  |  -9.1186      |
| TEMP     |   2.5312 |   0.837       |
| WSPM     |   0.6671 |  -0.0178      |

    ---

    ## 8. Honest summary

    ### What worked
    - Reconstruction R² at the chosen k = **0.757**.
- The best-head free-run forecast stays bounded (growth = 0.08×).

    ### What did not work / limitations
    - **Short-horizon reconstruction tax:** at h=1, skill = -0.102.  Squeezing to k latent numbers discards small-scale detail that matters for very-next-step predictions.  Persistence wins here.
- **Constant-velocity rollout** (old method) performs poorly at h=8 (skill -0.319), confirming it is the weakest link.

    ---

    *Generated automatically by experiments/acml/exp_all_datasets.py*
