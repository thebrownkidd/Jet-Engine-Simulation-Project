"""
ACML TASK 1 — Theory/implementation consistency check.

Question: does the latent rollout project/clip the ENTIRE latent vector (as
Theorem 2 requires) or only the primary coordinate?

Finding (static code audit): the production rollout in
`experiments/exp_rollout_stability.py::rollout_manifold` clips ONLY coordinate 0
(`future_h[:, 0] = np.clip(future_h[:, 0], 0.0, 1.5)`). The remaining
coordinates are extrapolated by constant velocity WITHOUT projection. Theorem 2
("decoded latent rollout is uniformly bounded") is stated for the full bounded
box [0,1]^k, so the released code satisfies the theorem verbatim only when k=1
or when the unconstrained coordinates happen to stay bounded.

This script quantifies whether switching to the theory-matched FULL-box
projection changes the reported stability/accuracy metrics materially. It trains
a bounded k=2 model per dataset and evaluates both rollout variants:

  h0_clip   legacy production behaviour (clip coordinate 0 only)
  full_box  theory-matched projection of ALL coordinates to [0,1]

Outputs
  results/acml/tables/theory_impl_rollout_compare.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import acml_common as ac  # noqa: E402

DATASETS = [1, 2, 3, 4]
EPOCHS = ac.ACML_EPOCHS


def main(fds=DATASETS):
    rows = []
    for fd in fds:
        print("=" * 70)
        print(f"TASK 1  FD00{fd}  (theory-matched full-box vs legacy h0-clip)")
        print("=" * 70)
        tr_den, te_den = ac.setup_dataset(fd, k=2)
        man = ac.train_flex_ae(tr_den, k=2, bounded=True, lambda_mono=5.0,
                               lambda_smooth=2.0, seed=42, epochs=EPOCHS)
        for proj in ("h0_clip", "full_box"):
            fn, growth, bounded = ac.freerun_growth(man, tr_den, te_den, proj)
            nrmse = ac.rollout_nrmse_by_horizon(man, tr_den, te_den, proj)
            row = dict(dataset=f"FD00{fd}", projection=proj,
                       freerun_norm=fn, freerun_growth=growth, bounded=bounded,
                       nrmse_h1=nrmse[1], nrmse_h10=nrmse[10],
                       nrmse_h25=nrmse[25], nrmse_h50=nrmse[50])
            rows.append(row)
            print(f"  {proj:<9} growth=x{growth:7.3f} bounded={bounded}  "
                  f"nrmse_h50={nrmse[50]:.3f}")
    df = pd.DataFrame(rows)
    out = os.path.join(ac.ACML_TAB, "theory_impl_rollout_compare.csv")
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    fds = [int(x) for x in sys.argv[1:]] or DATASETS
    main(fds)
