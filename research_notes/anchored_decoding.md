# Research Note: Anchored Decoding for Bounded-Latent Forecasting

**Date:** 2026-08-26
**Status:** working note (result committed; extension to N-CMAPSS pending)

## 1. Motivation

The bounded-latent rollout forecasts by decoding a rolled-out latent state,
`ŷ_{c+τ} = D(ĥ_{c+τ})`. On steadily-degrading data (C-MAPSS, PHM milling) this
wins at long horizons. On near-stationary external data it does not: on the NASA
Li-ion battery set the sensor-space VAR beat every plain-decoded latent head at
every horizon, and all latent heads were catastrophically worse than persistence
at short horizon.

Two structural causes were identified:

1. **Unit-root latent dynamics.** The sensor-space companion spectral radius is
   `ρ ≈ 1` (a near-driftless random walk). For a martingale, persistence is the
   minimum-variance predictor, so no latent *rollout* can beat it at short/mid
   horizon.
2. **Reconstruction tax.** Even a perfect latent forecast decodes with the fixed
   origin error `y_c − D(h_c)`. This constant offset dominates the tiny per-step
   change on a near-stationary signal, which is exactly what handed the win to
   the sensor-space VAR (it forecasts in observation space and pays no decode
   tax).

## 2. Method: anchored decoding

Anchor the decoded **increment** to the last observation:

```
ŷ_{c+τ} = y_c + [ D(ĥ_{c+τ}) − D(h_c) ].
```

At τ = 0 this is exactly `y_c` (persistence-consistent); for τ > 0 it adds only
the model's decoded *change*. The constant reconstruction offset `D(h_c) − y_c`
cancels by construction. `y_c` is the observed value at the forecast origin, so
there is no look-ahead.

## 3. Theory (same family as Theorem 3)

Let `r_t = y_t − D(h_t)` be the reconstruction residual.

**Identity.** With `y_t = D(h_t) + r_t`, the `D(h_c)` terms cancel:

```
y_{c+τ} − ŷ_{c+τ} = [ D(h_{c+τ}) − D(ĥ_{c+τ}) ] + ( r_{c+τ} − r_c ).
```

**Theorem 4 (anchored decoding forecast error).** If `D` is `L`-Lipschitz on the
compact latent domain and each latent coordinate has `|Δ²h^{(j)}_t| ≤ κ`, then

```
‖ y_{c+τ} − ŷ_{c+τ} ‖ ≤ L·√K·(κ/2)·τ(τ+1) + ‖ r_{c+τ} − r_c ‖,
```

and if `‖r_t‖ ≤ ρ` then the last term is `≤ 2ρ`.

*Proof.* Lipschitz continuity bounds the first term by `L‖h_{c+τ} − ĥ_{c+τ}‖`;
Theorem 3 applied coordinatewise bounds `‖h_{c+τ} − ĥ_{c+τ}‖ ≤ √K·(κ/2)τ(τ+1)`.
The triangle inequality bounds the residual difference by `2ρ`. ∎

**Corollary (anchoring cancels the origin offset).** Plain decoding obeys
`‖y_{c+τ} − D(ĥ_{c+τ})‖ ≤ L√K(κ/2)τ(τ+1) + ‖r_{c+τ}‖`. Anchoring replaces
`‖r_{c+τ}‖` by `‖r_{c+τ} − r_c‖`, which is `0` at `τ = 0` and small whenever the
residual is temporally persistent. The leading curvature term is identical, so
the bounded-rollout (Theorem 2) and smoothness (Theorem 3) guarantees carry over.

**Honesty note.** The earlier informal claim `|err| ≤ L(κ/2)τ(τ+1)` was
incomplete: it dropped the residual term `‖r_{c+τ} − r_c‖`. The bound above is
the correct statement. The decoder is genuinely Lipschitz for the HealthAE
architecture (Linear–Tanh–Linear–Tanh–Linear, then `×diag(sd)`), so `L` is finite
and computable.

## 4. Empirical result (NASA battery, condition-normalized)

Horizons `{1,10,30,60}`, 7 operating conditions, skill vs persistence:

| head              | h1      | h10    | h30   | h60   | growth |
|-------------------|---------|--------|-------|-------|--------|
| persistence       | 0       | 0      | 0     | 0     | 1.00   |
| sensor VAR        | −29.84  | −7.44  | −6.72 | −3.65 | 0.92   |
| damped CV (plain) | −610.8  | −15.19 | −7.89 | −2.81 | 1.00   |
| **anchored (cvd)**| **+0.05** | **+0.01** | **+0.05** | **+0.04** | **1.00** |

Anchored decoding is the only latent head with positive skill at every horizon,
and it beats the sensor-space VAR everywhere. Free-run growth is 1.00 (bounded).

## 5. Caveats

- Margins are small (`+0.006 … +0.048`): near a martingale this is close to the
  achievable ceiling. The result is a sign flip, not a large-margin win.
- The winning configuration is anchored + heavy damping (`γ*=0`): a small,
  bounded decoded drift added to persistence. Anchored Holt slightly
  over-corrects.

## 6. Next step

Run the same head set on N-CMAPSS, where degradation drift is real (`γ` and Holt
fits should move toward 1). There anchored decoding should win by a wider margin
than on the near-stationary battery.

## 7. Reproduce

```powershell
.\.venv\Scripts\python.exe experiments\acml\battery_prep.py
.\.venv\Scripts\python.exe experiments\acml\exp_all_datasets.py --datasets battery `
    --epochs-sweep 200 --epochs-ae 2000 --epochs-dyn 400 --lambda-smooth 2.0 --seeds 0 1 2
```
