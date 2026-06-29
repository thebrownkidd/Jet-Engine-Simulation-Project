# Theory / Implementation Consistency Check (ACML TASK 1)

**Claim under audit.** Theorem 2 (bounded latent rollout) states that if the
*entire* latent vector is confined to a compact box $\mathcal{H}=[0,1]^k$ and the
decoder is continuous, then the decoded rollout is uniformly bounded for all
horizons. This note checks whether the released implementation actually projects
the whole latent vector, and whether enforcing the theorem verbatim changes the
reported metrics.

## 1. What the implementation did before

The production rollout lives in
[experiments/exp_rollout_stability.py](../experiments/exp_rollout_stability.py),
function `rollout_manifold`:

```python
future_h = np.array([h0 + v * (s + 1) for s in range(steps)])
future_h[:, 0] = np.clip(future_h[:, 0], 0.0, 1.5)   # only coordinate 0
return man.decode(future_h)
```

Only the **primary coordinate** $h_0$ is clipped, and to $[0, 1.5]$ rather than
$[0,1]$. The remaining coordinates $h_1,\dots,h_{k-1}$ are extrapolated by
constant velocity with **no projection**. Consequently:

- For $k=2$ the decoder could in principle receive an unbounded $h_1$, so the
  formal guarantee of Theorem 2 was not matched verbatim by the code.
- The clip range $[0,1.5]$ on $h_0$ also exceeds the unit box used in the proof
  (a deliberate allowance so the wear coordinate can slightly exceed 1 near
  end-of-life).

In practice the unconstrained coordinates stayed bounded because the fitted
velocities are small, which is why the original experiments still reported
bounded free-runs. But "empirically bounded" is weaker than the theorem.

## 2. What was added (no silent replacement)

A new, clearly named rollout is provided in
[experiments/acml/acml_common.py](../experiments/acml/acml_common.py)
(`rollout_latent`) with an explicit `projection` argument and three modes:

| Mode | Behaviour | Role |
|---|---|---|
| `h0_clip` | clip coordinate 0 to $[0,1.5]$ (legacy) | reproduces production |
| `full_box` | clip **all** coordinates to $[0,1]$ | **theory-matched** (Theorem 2 holds verbatim) |
| `none` | no projection | unbounded-AE contrast (Tasks 2/5) |

The production code path is **not modified**; the legacy behaviour remains
available as `h0_clip`. All ACML experiments that claim Theorem-2 boundedness use
`full_box`.

## 3. Which results use theory-matched full projection

Every ACML task that evaluates the *proposed bounded model's* stability uses
`projection="full_box"`: the extended ablation (Task 2), seed robustness
(Task 3), K-aware sweep (Task 4), boundedness mechanism (Task 5), and the
baseline comparison (Task 6). The unbounded-AE variants use `projection="none"`
so their (lack of) boundedness is reported honestly.

## 4. Does the fix change metrics materially?

Trained a bounded $k=2$ model per dataset (seed 42, 1500 epochs) and evaluated
both rollout modes. Source:
[results/acml/tables/theory_impl_rollout_compare.csv](../results/acml/tables/theory_impl_rollout_compare.csv).

| Dataset | Mode | Free-run growth | Bounded | NRMSE@1 | NRMSE@10 | NRMSE@25 | NRMSE@50 |
|---|---|---:|:--:|---:|---:|---:|---:|
| FD001 | h0_clip | 1.301 | yes | 0.173 | 0.217 | 0.266 | 0.465 |
| FD001 | full_box | 1.301 | yes | 0.173 | 0.217 | 0.266 | 0.465 |
| FD002 | h0_clip | 2.380 | yes | 0.306 | 0.415 | 0.523 | 0.891 |
| FD002 | full_box | 2.380 | yes | 0.306 | 0.415 | 0.523 | 0.886 |
| FD003 | h0_clip | 3.022 | yes | 0.127 | 0.158 | 0.179 | 0.295 |
| FD003 | full_box | 2.620 | yes | 0.127 | 0.158 | 0.179 | 0.295 |
| FD004 | h0_clip | 1.050 | yes | 0.190 | 0.233 | 0.316 | 0.514 |
| FD004 | full_box | 1.050 | yes | 0.190 | 0.233 | 0.316 | 0.514 |

**Conclusion.** The change is immaterial to the headline metrics. The bounded
flag is unchanged (all bounded). The only visible effect is on FD003, where
projecting the auxiliary coordinate reduces the free-run norm growth from
$3.02\times$ to $2.62\times$ — i.e. the auxiliary coordinate *was* drifting
slightly and the theory-matched projection removes that drift. Rollout NRMSE is
unchanged to three decimals everywhere except a negligible FD002 improvement at
horizon 50 ($0.891 \to 0.886$).

**Recommendation for the paper.** Report Theorem 2 with the `full_box`
projection (it is now satisfied verbatim) and state in an implementation note
that the legacy single-coordinate clip yields statistically indistinguishable
results, with the auxiliary-coordinate drift on FD003 being the only measurable
difference. This removes the prior "implementation is weaker than the theorem"
caveat without changing any conclusion.
