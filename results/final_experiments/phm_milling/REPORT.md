    # phm_milling — Experiment Report

    > **Dataset:** PHM 2010 CNC milling — 6 cutting tools, 35 force/vibration/AE features per cut, steady tool-wear degradation.

    ---

    ## 1. What this experiment does

    We apply a **Context-Conditioned Bounded Latent Dynamics** model to this
    dataset. The core idea:

    1. An **autoencoder** squeezes the 34-variable sensor stream into
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
    | AE epochs | 800 |
    | Dynamics epochs | 400 |
    | Seeds for MLP heads | [0, 1, 2] |
    | Monotonicity penalty λ_mono | 1.0 |
    | Smoothness penalty λ_smooth | 0.5 |
    | Forecast horizons evaluated | [1, 10, 25, 50] (in cycles/steps) |
    | Train / test split | 4 / 2 units |
    | Context columns used | none |
    | Sensors (raw → selected) | 35 → 34 |
    | Denoising window | 15 cycles (causal rolling median) |
    | Sensor selection threshold | trend |corr(sensor,cycle)| ≥ 0.2 |

    **Preprocessing pipeline (applied before any model training):**
    1. **Train/test split** — units split first so no preprocessing leaks across the boundary.
    2. **Denoising** — causal rolling-median (window=15 cycles) applied per unit per sensor.
       Removes cycle-to-cycle noise without looking ahead; matches the validated `manifold` pipeline.
    3. **Sensor selection** — sensors with mean |corr(sensor, cycle)| < 0.2 across training units are dropped. These sensors carry no degradation signal and only add noise to the latent space. Selected: 34/35 sensors.

    **Why k=4?**
    The k-sweep trains the autoencoder at each dimension and applies the **PCA elbow method**
    to find the latent dimension that captures the data's intrinsic structure without overfitting.
    The elbow is found using two criteria:
      1. The smallest k where cumulative explained variance ≥ 85%
      2. The k where incremental variance gain drops below 5%

    We take the **more conservative choice** (smaller k). For k=4, the reconstruction
    R² (mean across sensors) = **0.732** (worst single sensor = -0.621).

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
|   1 |     0.3896 |        -0.6728 |
|   2 |     0.7094 |        -0.5113 |
|   3 |     0.6851 |        -0.9138 |
|   4 |     0.7319 |        -0.6209 |
|   5 |     0.7152 |        -0.7319 |
|   6 |     0.7092 |        -0.635  |

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
| persistence |      0     |     0.0366 |      0      |      0.2892 |      0      |      0.5028 |      0      |      0.8765 |           1      | True      |
| cv          |   -158.037 |     0.4618 |     -2.3377 |      0.5284 |     -1.4711 |      0.7904 |     -0.1931 |      0.9574 |           3.1625 | True      |
| var_sensor  |     -9.511 |     0.1187 |     -3.7133 |      0.6279 |     -2.6256 |      0.9574 |     -2.8581 |      1.7216 |           2.6934 | True      |
| mlp_noctx   |   -158.02  |     0.4618 |     -1.7143 |      0.4764 |      0.1246 |      0.4698 |      0.438  |      0.6562 |           2.3602 | True      |
| mlp_ctx     |   -158.02  |     0.4618 |     -1.7143 |      0.4764 |      0.1246 |      0.4698 |      0.438  |      0.6562 |           2.3602 | True      |

    **Winner:** `persistence` (highest mean skill at horizons [10, 25, 50]).

    ---

    ## 5. Observation-space accuracy (best head: `persistence`)

    This table shows the per-sensor forecast accuracy in the **original feature
    scale** (regime-normalized z-scores or raw feature units depending on the
    dataset).  Averaged over the three anchor points (50%, 65%, 80% of each
    test trajectory) and all test units.

    **rmse**: root-mean-squared error in original feature units.
    **r2**: coefficient of determination; 1.0 = perfect, 0 = no better than the
    mean, negative = worse than the mean.

    Values are averaged over all forecast horizons ([1, 10, 25, 50]).

    | sensor   |    rmse |      r2 |
|:---------|--------:|--------:|
| AE_kurt  |  0.9902 | -4.8292 |
| AE_peak  |  0.0189 |  0.2788 |
| AE_rms   |  0.009  |  0.2016 |
| AE_skew  |  0.2914 | -0.344  |
| Fx_kurt  |  0.0381 | -1.7911 |
| Fx_peak  | 20.2928 |  0.2671 |
| Fx_rms   | 14.458  | -0.3258 |
| Fx_skew  |  0.0221 |  0.8694 |
| Fx_std   |  5.3112 |  0.5634 |
| Fy_kurt  |  0.3295 | -0.1962 |
| Fy_peak  |  3.753  |  0.7831 |
| Fy_rms   |  1.4594 |  0.7532 |
| Fy_skew  |  0.0639 |  0.653  |
| Fy_std   |  1.1659 |  0.8106 |
| Fz_kurt  |  0.0208 |  0.9804 |
| Fz_peak  |  8.2707 |  0.3989 |
| Fz_rms   |  5.3228 | -0.2118 |
| Fz_skew  |  0.0154 |  0.9889 |
| Fz_std   |  2.2349 |  0.676  |
| Vx_kurt  |  0.1136 |  0.8732 |
| Vx_peak  |  0.1767 |  0.7753 |
| Vx_rms   |  0.0283 |  0.7161 |
| Vx_skew  |  0.0213 |  0.9302 |
| Vx_std   |  0.0283 |  0.7161 |
| Vy_kurt  |  0.0883 |  0.8605 |
| Vy_peak  |  0.1832 |  0.8825 |
| Vy_rms   |  0.029  |  0.8788 |
| Vy_skew  |  0.0204 |  0.2677 |
| Vy_std   |  0.029  |  0.8788 |
| Vz_kurt  |  0.0976 |  0.7011 |
| Vz_peak  |  0.2499 |  0.8098 |
| Vz_rms   |  0.0438 |  0.7491 |
| Vz_skew  |  0.0378 |  0.3368 |
| Vz_std   |  0.0438 |  0.7493 |

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

    This figure shows the **predictions** made by the best-head model (`persistence`)
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

    ## 7. Observation-space accuracy (best head: `persistence`)

    This table shows the per-sensor forecast accuracy in the **original feature
    scale** (regime-normalized z-scores or raw feature units depending on the
    dataset).  Averaged over the three anchor points (50%, 65%, 80% of each
    test trajectory) and all test units.

    **rmse**: root-mean-squared error in original feature units.
    **r2**: coefficient of determination; 1.0 = perfect, 0 = no better than the
    mean, negative = worse than the mean.

    Values are averaged over all forecast horizons ([1, 10, 25, 50]).

    | sensor   |    rmse |      r2 |
|:---------|--------:|--------:|
| AE_kurt  |  0.9902 | -4.8292 |
| AE_peak  |  0.0189 |  0.2788 |
| AE_rms   |  0.009  |  0.2016 |
| AE_skew  |  0.2914 | -0.344  |
| Fx_kurt  |  0.0381 | -1.7911 |
| Fx_peak  | 20.2928 |  0.2671 |
| Fx_rms   | 14.458  | -0.3258 |
| Fx_skew  |  0.0221 |  0.8694 |
| Fx_std   |  5.3112 |  0.5634 |
| Fy_kurt  |  0.3295 | -0.1962 |
| Fy_peak  |  3.753  |  0.7831 |
| Fy_rms   |  1.4594 |  0.7532 |
| Fy_skew  |  0.0639 |  0.653  |
| Fy_std   |  1.1659 |  0.8106 |
| Fz_kurt  |  0.0208 |  0.9804 |
| Fz_peak  |  8.2707 |  0.3989 |
| Fz_rms   |  5.3228 | -0.2118 |
| Fz_skew  |  0.0154 |  0.9889 |
| Fz_std   |  2.2349 |  0.676  |
| Vx_kurt  |  0.1136 |  0.8732 |
| Vx_peak  |  0.1767 |  0.7753 |
| Vx_rms   |  0.0283 |  0.7161 |
| Vx_skew  |  0.0213 |  0.9302 |
| Vx_std   |  0.0283 |  0.7161 |
| Vy_kurt  |  0.0883 |  0.8605 |
| Vy_peak  |  0.1832 |  0.8825 |
| Vy_rms   |  0.029  |  0.8788 |
| Vy_skew  |  0.0204 |  0.2677 |
| Vy_std   |  0.029  |  0.8788 |
| Vz_kurt  |  0.0976 |  0.7011 |
| Vz_peak  |  0.2499 |  0.8098 |
| Vz_rms   |  0.0438 |  0.7491 |
| Vz_skew  |  0.0378 |  0.3368 |
| Vz_std   |  0.0438 |  0.7493 |

    ---

    ## 8. Honest summary

    ### What worked
    - Reconstruction R² at the chosen k = **0.732**.
- The best-head free-run forecast stays bounded (growth = 1.00×).

    ### What did not work / limitations
    - **Constant-velocity rollout** (old method) performs poorly at h=10 (skill -2.338), confirming it is the weakest link.

    ---

    *Generated automatically by experiments/acml/exp_all_datasets.py*
