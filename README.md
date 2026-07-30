# Bounded Latent-Dynamics for Machine Health and Forecasting

This project is about teaching a computer to watch the sensors on a machine (like a jet engine or a drill) and answer two questions:

1. **How healthy is this machine right now?**
2. **What will its sensors look like in the future?**

You do **not** need any math to read this file. Everything below is written in plain language. If you are in your first year of data science, you should be able to follow the whole thing.

---

## 1. The big idea in plain words

Real machines have a lot of sensors. A jet engine can have 20+ readings (temperatures, pressures, speeds...). That is a lot of numbers changing every second, and most of them are noisy and tangled together.

Our algorithm does something simple and powerful: it **squeezes all those sensors down into just 2 small numbers** that summarize the "hidden state" of the machine. Think of it like turning a messy dashboard of 20 dials into a single "health bar" in a video game.

We call these 2 squeezed numbers the **latent state**.

There are two parts to the machine that does the squeezing:

- An **encoder**: takes the many sensors → gives back the 2 small numbers.
- A **decoder**: takes the 2 small numbers → tries to rebuild the original sensors.

If the decoder can rebuild the sensors well, it means the 2 small numbers really did capture what mattered. This "squeeze and rebuild" machine is called an **autoencoder**.

### The special trick: keeping the numbers "bounded"

Here is the key contribution of the project.

We **force the 2 small numbers to always stay between 0 and 1**. They can never run off to infinity. This is done by design, not by luck.

Why does that matter? Because we don't just want to describe the present, we want to **predict the future**. To predict the future we "roll forward": we guess the next state, then the next, then the next. Normally, when you keep guessing forward like this, small errors pile up and the prediction **explodes** into nonsense (numbers blow up to huge values).

Because our 2 numbers are locked between 0 and 1, **the prediction can never explode**. It is mathematically impossible for it to blow up. This is the "bounded" in the project name. It is like predicting a temperature and knowing for certain your prediction will never claim the room is a million degrees.

So the core contribution is:

> **A safe, small (2-number) summary of a machine's state that never blows up, so you can safely predict far into the future.**

---

## 2. What changed recently (the newest and most important part)

The original version predicted the future using a very simple rule called **constant velocity**:

> "Whatever direction the health was moving lately, just keep moving that same way forever."

That is fine for a machine that is steadily wearing out. But it is a **bad** rule when things are more complicated — for example when the weather changes, the workload changes, or the time of day matters. Constant velocity can't react to any of that. It just keeps going straight.

So we **replaced the constant-velocity rule with a small learned model** that can react to context. The new rule is:

> "Look at the current state **and** the current context (like the hour of the day or the season), then take a small, careful step."

We call this a **context-conditioned bounded dynamics model**. It is still bounded (still can't explode), but now it is smart enough to change its behavior depending on the situation.

Two extra design choices keep it safe and honest:

- **Small steps only.** The model is only allowed to nudge the state a little bit at each step, never make a wild jump.
- **Trained for the long run.** We don't just teach it to predict one step ahead; we teach it to predict many steps ahead, so its long-range forecasts stay sensible.

---

## 3. The datasets (what data we tested on)

We tested the idea on **five different real-world datasets**. Four are about machines wearing out ("degradation"), and one is about air pollution ("forecasting"). Using very different data on purpose is how we check whether the idea is genuinely useful or just got lucky once.

### 3.1 NASA C-MAPSS turbofan engines (the original dataset)
Simulated jet engines run until they fail. Each engine has many sensors recorded every cycle until it breaks. This is the dataset the whole method was designed around.

### 3.2 IMS bearings
Real bearings (the round parts that let shafts spin) were run non-stop until they physically wore out, while vibration sensors listened at very high speed. We turned the raw vibration into simpler features (how strong, how rough, etc.).

### 3.3 NASA batteries
Rechargeable batteries charged and discharged over and over until they aged. The main signal is that each battery slowly holds less charge over time.

### 3.4 PHM 2010 milling (cutting tools)
Cutting tools on a CNC milling machine were used for hundreds of cuts until they got dull. Force and vibration sensors recorded each cut. This is the dataset most similar to the jet engines: a slow, steady wearing-out.

### 3.5 Beijing air quality
This one is **not** a machine. It is 12 air-monitoring stations around Beijing recording pollution (like PM2.5) and weather every hour for years. There is no "failure" here — the goal is just to **forecast** future pollution. We used it to stress-test the method on a totally different kind of problem.

---

## 4. The experiments (what we actually did)

For each dataset we ran some or all of these tests:

1. **Reconstruction test** — Can the 2-number summary rebuild the original sensors? (Is the squeeze lossy or faithful?)
2. **Stability test** — If we roll the prediction far into the future, does our bounded method stay calm while a normal method blows up?
3. **Health-tracking test** (machines only) — Does our health number move steadily in one direction as the machine wears out?
4. **Remaining-life test** (machines only) — Can we predict how much life the machine has left, better than a lazy "just guess the average" baseline?
5. **Forecast test** — Can we predict future sensor values better than a "lazy" baseline that just repeats the last value?
6. **The new head-to-head ablation** (air quality and milling) — We compared the old constant-velocity rule against the new context-aware model and several other options, to see exactly which ingredient helps.

---

## 5. What each metric means (in simple language)

These are the "scores" we report. Here is what each one is really telling you:

- **Reconstruction R² (recon)** — "How faithfully did the 2-number summary rebuild the sensors?" 1.0 is perfect; 0 means no better than guessing the average; negative means worse than guessing.

- **Spectral radius (rho)** — "Is the simple competing predictor stable or does it explode?" If this number is **above 1**, that predictor blows up over time. If it is **below or near 1**, it stays calm on its own. This tells us whether our stability trick is even needed.

- **Free-run growth** — "When we predict far into the future, how much bigger did the numbers get?" A value near 1 means the forecast stayed calm. A huge value means it exploded. We tag a forecast as **"bounded"** (good) if it stayed calm.

- **Monotonicity violation** — "How often did the health number go the *wrong* way?" (machines only). Low is good: it means health moves steadily toward failure instead of jumping around.

- **Remaining-life error (RMSE) vs baseline** — "How far off were our life predictions, compared to a lazy average guess?" Lower than the baseline is good.

- **Forecast skill vs persistence** — "Did we beat the laziest possible forecast (just repeat the last value)?" **Positive = we beat it. Zero = we tied it. Negative = we were worse than doing nothing.** This is the honest, hard test for forecasting.

- **Saturation** — "Did the prediction get stuck jammed in a corner (all 0s and 1s)?" High saturation is a warning sign that the forecast collapsed to a boring, wrong extreme.

- **Params / train time** — Just how big and how slow the model is. Smaller and faster is nicer, all else equal.

---

## 6. What was good and what was bad (per dataset)

### Jet engines (C-MAPSS) — **worked great**
- Good: faithful rebuild, health moved steadily, remaining-life predictions were strong, and the stability trick clearly helped.
- Bad: nothing major — this is the home-turf dataset.

### Milling tools (PHM) — **worked great, the strongest outside result**
- Good: the competing simple predictor was unstable (it blew up ~800×) while ours stayed calm. Health tracked tool wear nicely. Remaining-life error was about **2.4× better** than the lazy baseline. Everything reproduced even with just 2 numbers.
- In the new head-to-head: predicting **far ahead**, our smart bounded model clearly beat the sensor-space model and the classic linear method, and it beat the old constant-velocity rule too.
- Bad: predicting **one step ahead**, our method was worse than just repeating the last value — because squeezing to 2 numbers throws away small details that matter for very short predictions. We call this the "reconstruction tax."

### Bearings (IMS) — **mostly worked**
- Good: the competing predictor blew up (~180×) while ours stayed calm. Rebuild quality and remaining-life were fine once we used **3 numbers instead of 2**.
- Bad: the "steady health" idea **failed** here. Bearings stay healthy for a long time and then fail suddenly, so the health number jumps instead of sliding. The steady-progress assumption is not universal.

### Batteries (NASA) — **the stability trick wasn't needed**
- Good: the method still rebuilt the signals (with 4+ numbers).
- Bad: the competing simple predictor was **already stable on its own** here, so our anti-explosion trick had nothing to fix. Also the batteries were run under very different temperatures and protocols, and a tiny 2-number summary couldn't absorb that variety. Remaining-life predictions were weak.
- Lesson: our stability advantage only matters when the competing method would otherwise blow up.

### Air quality (Beijing) — **the honest stress test**
- Good: even here, the 2-number summary rebuilt the pollution + weather signals well.
- Bad (old method): the **constant-velocity forecast was a disaster** — worse than just repeating the last value at every horizon, and it drifted off into a stuck corner. Pollution is cyclic (day/night, seasons), and "keep going straight" is exactly the wrong instinct.
- Good (new method): the **new context-aware bounded model completely fixed this.** By feeding it the time of day and season, it went from **worse-than-nothing to clearly-better-than-nothing** at the useful horizons, and it stayed calm and bounded the whole time. This is the headline win for the new work.

---

## 7. The head-to-head results (the newest experiment)

We lined up nine different forecasting rules on the same data and scored them. The important takeaways:

- **The old constant-velocity rule is genuinely broken** on complicated data. On air quality it blew up and scored far worse than doing nothing.
- **The new context-aware bounded model fixed it** and stayed calm.
- **Context helps.** Telling the model the hour and season made its longer forecasts better.
- **Training for the long run helps.** Models taught to predict many steps ahead stayed steadier than those taught only one step ahead.
- **The bounded space earns its keep exactly when the sensors are unstable.** On the milling tools (where the raw sensors want to explode), doing the learning inside our safe 2-number space beat doing it directly on the raw sensors. On air quality (where the sensors were already calm), our space tied the raw-sensor approach — no free lunch, but also no harm, and always safe.

Two honest limitations we found and report openly:

1. **Short-horizon tax:** squeezing to a few numbers hurts very-short-term predictions.
2. **It's not always needed:** the anti-explosion benefit only shows up when the alternative would actually explode.

---

## 8. One-paragraph summary

We turn a machine's many messy sensors into a tiny 2-number "health summary" that is locked between 0 and 1 so it can **never** blow up, even when predicting far into the future. That safety is the core contribution. We used to predict the future with a naive "keep going straight" rule; we replaced it with a small, safe model that also looks at context (like time of day or season) and takes careful steps. Across five very different datasets, the method works best when a machine wears out steadily and the raw sensors would otherwise become unstable (jet engines, milling tools), is only partly useful when failures are sudden (bearings) or the sensors are already calm (batteries), and — with the new context-aware upgrade — finally works on a completely different problem (forecasting air pollution) where the old rule had failed badly.

---

## 9. Where things live in this repo

- `experiments/acml/` — all the experiment scripts:
  - `*_prep.py` — turn each raw dataset into a clean table (bearings, battery, milling, air quality).
  - `exp_ims_bearing.py` — the general machine-health runner (rebuild, stability, remaining-life).
  - `exp_air_quality.py` — the forecasting runner.
  - `latent_dynamics.py` — the **new** context-aware bounded prediction model and its baselines.
  - `exp_latent_dynamics_compare.py` — the **new** head-to-head experiment that produced Section 7.
- `data/processed/` — the cleaned tables the experiments read.
- `results/acml/tables/` and `results/acml/figures/` — the scores and plots each experiment writes out.

## 10. How to run it

Use the project's Python environment, then run any experiment. Two examples:

```powershell
# Rebuild the air-quality data with time-of-day / season context
.\.venv\Scripts\python.exe experiments\acml\air_quality_prep.py --context --stride 3 --out data\processed\air_quality_features_ctx.csv

# Run the new head-to-head forecasting comparison
.\.venv\Scripts\python.exe experiments\acml\exp_latent_dynamics_compare.py --epochs-ae 800 --epochs-dyn 400 --seeds 0 1 2 --horizons 1 8 24 48
```

Each run prints a plain-language summary at the end and saves a table and a figure into `results/acml/`.
