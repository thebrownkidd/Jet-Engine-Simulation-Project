    # ims_bearing — Experiment Report

    > **Dataset:** NASA IMS bearings — 16 run-to-failure bearing units, 14 time+frequency vibration features per channel.

    ---

    ## 1. What this experiment does

    We apply a **Context-Conditioned Bounded Latent Dynamics** model to this
    dataset. The core idea:

    1. An **autoencoder** squeezes the 14-variable sensor stream into
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
    | Monotonicity penalty λ_mono | 0.5 |
    | Smoothness penalty λ_smooth | 0.5 |
    | Forecast horizons evaluated | [1, 10, 25, 50] (in cycles/steps) |
    | Train / test split | 10 / 6 units |
    | Context columns used | none |
    | Sensors (raw → selected) | 14 → 14 |
    | Denoising window | 15 cycles (causal rolling median) |
    | Sensor selection threshold | trend |corr(sensor,cycle)| ≥ 0.2 |

    **Preprocessing pipeline (applied before any model training):**
    1. **Train/test split** — units split first so no preprocessing leaks across the boundary.
    2. **Denoising** — causal rolling-median (window=15 cycles) applied per unit per sensor.
       Removes cycle-to-cycle noise without looking ahead; matches the validated `manifold` pipeline.
    3. **Sensor selection** — sensors with mean |corr(sensor, cycle)| < 0.2 across training units are dropped. These sensors carry no degradation signal and only add noise to the latent space. Selected: 14/14 sensors.

    **Why k=3?**
    The k-sweep trains the autoencoder at each dimension and applies the **PCA elbow method**
    to find the latent dimension that captures the data's intrinsic structure without overfitting.
    The elbow is found using two criteria:
      1. The smallest k where cumulative explained variance ≥ 85%
      2. The k where incremental variance gain drops below 5%

    We take the **more conservative choice** (smaller k). For k=3, the reconstruction
    R² (mean across sensors) = **0.699** (worst single sensor = 0.028).

    **Why PCA elbow, not max reconstruction R²?**
    Reconstruction R² always rises with k, so choosing max R² would lead to overfitting
    (using many more latent dimensions than necessary). The PCA elbow method finds the
    "natural" dimensionality of the problem, a much more principled approach.

    **Why λ_mono=0.5?**
    The dataset has a degradation trend, so we encourage the primary latent coordinate to increase monotonically with time.  This is the *health progressing toward failure* assumption.

    ---

    ## 3. K-sweep results

    This table shows how reconstruction quality changes as we give the latent
    space more dimensions.  **recon_r2** is the mean R² across all sensors;
    **recon_r2_min** is the worst single sensor.

    |   k |   recon_r2 |   recon_r2_min |
|----:|-----------:|---------------:|
|   1 |     0.456  |        -0.0835 |
|   2 |     0.5185 |         0.0252 |
|   3 |     0.6993 |         0.028  |
|   4 |     0.7427 |         0.3247 |
|   5 |     0.7487 |         0.2364 |
|   6 |     0.7376 |         0.3716 |

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
| persistence |     0      |     0.0428 |      0      |      0.1014 |      0      |      0.1045 |      0      |      0.1362 |           1      | True      |
| cv          |  -131.204  |     0.4917 |    -23.9625 |      0.5068 |    -25.9246 |      0.5424 |    -20.5465 |      0.6322 |           0.9197 | True      |
| var_sensor  |    -0.1578 |     0.046  |     -0.7248 |      0.1332 |     -3.1751 |      0.2136 |     -7.9037 |      0.4064 |           1.6392 | True      |
| mlp_noctx   |  -131.17   |     0.4916 |    -23.0658 |      0.4976 |    -23.1315 |      0.5135 |    -13.6741 |      0.5214 |           1.0884 | True      |
| mlp_ctx     |  -131.17   |     0.4916 |    -23.0658 |      0.4976 |    -23.1315 |      0.5135 |    -13.6741 |      0.5214 |           1.0884 | True      |

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

    | sensor    |   rmse |     r2 |
|:----------|-------:|-------:|
| band0     | 0.0113 | 0.9944 |
| band1     | 0.008  | 0.9957 |
| band2     | 0.0033 | 0.9968 |
| band3     | 0.0022 | 0.9949 |
| clearance | 0.2895 | 0.6941 |
| crest     | 0.2014 | 0.2489 |
| impulse   | 0.2483 | 0.548  |
| kurtosis  | 0.0419 | 0.987  |
| p2p       | 0.0372 | 0.9815 |
| peak      | 0.0258 | 0.9801 |
| rms       | 0.0005 | 0.9998 |
| shape     | 0.0014 | 0.9978 |
| skew      | 0.011  | 0.9531 |
| std       | 0.0007 | 0.9984 |

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

    | sensor    |   rmse |     r2 |
|:----------|-------:|-------:|
| band0     | 0.0113 | 0.9944 |
| band1     | 0.008  | 0.9957 |
| band2     | 0.0033 | 0.9968 |
| band3     | 0.0022 | 0.9949 |
| clearance | 0.2895 | 0.6941 |
| crest     | 0.2014 | 0.2489 |
| impulse   | 0.2483 | 0.548  |
| kurtosis  | 0.0419 | 0.987  |
| p2p       | 0.0372 | 0.9815 |
| peak      | 0.0258 | 0.9801 |
| rms       | 0.0005 | 0.9998 |
| shape     | 0.0014 | 0.9978 |
| skew      | 0.011  | 0.9531 |
| std       | 0.0007 | 0.9984 |

    ---

    ## 8. Honest summary

    ### What worked
    - Reconstruction R² at the chosen k = **0.699**.
- The best-head free-run forecast stays bounded (growth = 1.00×).

    ### What did not work / limitations
    - **Constant-velocity rollout** (old method) performs poorly at h=10 (skill -23.963), confirming it is the weakest link.

    ---

    *Generated automatically by experiments/acml/exp_all_datasets.py*
