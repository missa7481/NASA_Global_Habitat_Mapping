#!/usr/bin/env python3
"""RSI iteration 3b (corrects I3-03): is AlphaEarth's value a re-encoding of named drivers, or residual structure?

I3-03 compared Drivers+AE with Drivers+AE_perp, which are linear reparameterisations of one another, so the
test could not separate them. Here each component is used ALONE:
  AE          the full embedding
  AE_hat      the part of AE linearly explained by Climate+Landscape+Human (5-fold ridge, label-free)
  AE_perp     the residual, re-standardised
  Drivers+Human   reference
If AE's skill comes from re-encoding drivers, AE_hat ~ AE and AE_perp ~ chance.
If AE carries unnamed surface structure (U in the DAG), AE_perp keeps substantial skill, especially in-distribution;
if that structure is region-specific, AE_perp's skill should collapse under leave-region-out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.common import ART, Data, designs, metrics  # noqa: E402
from rsi.it3_causal import fit_pred  # noqa: E402


def main():
    D = Data()
    DS = designs(D)
    drv = D.X[:, D.cols["Climate"] + D.cols["Landscape"] + D.cols["Human"]]
    AE = D.X[:, D.cols["AlphaEarth"]]
    hat = cross_val_predict(Ridge(1.0), drv, AE, cv=5)
    perp = AE - hat
    z = lambda a: ((a - a.mean(0)) / (a.std(0) + 1e-6)).astype(np.float32)  # noqa: E731
    total_r2 = 1 - ((AE - hat) ** 2).sum() / ((AE - AE.mean(0)) ** 2).sum()
    sets = {"AE": z(AE), "AE_hat (driver-explained)": z(hat), "AE_perp (residual)": z(perp), "Drivers+Human": drv}
    rows = []
    for dname in ("block10", "loro", "zone"):
        for si, (tr, Ytr, te, Yte, label) in enumerate(DS[dname]):
            for nm, X in sets.items():
                rows.append({"design": dname, "split": label, "variant": nm, **metrics(fit_pred(D, X, tr, Ytr, te, sid=700 + si), Yte)})
        print(dname, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ART / "it3/proxy_decomposition.csv", index=False)
    print("total variance of AE explained by drivers (linear): %.3f" % total_r2)
    print(df.groupby(["variant", "design"])["AUC_mean"].mean().unstack().round(3))
    (ART / "it3/proxy_r2.txt").write_text(f"{total_r2:.4f}")


if __name__ == "__main__":
    main()
