#!/usr/bin/env python3
"""PCam: does the redundancy omega_k = t * rho_k (study-only, on the gain's metric) explain the sample gain?

For each model x study size x loss: G_K on the benchmark-aware deltas q_k - b_k (fold score minus the same model's
benchmark score), with the single-split variance taken from the first split (paper default) or averaged over the
splits (same estimand, less noisy); omega_k = t * mean_seed rho_cum(k), rho the OOF-intersection per-sample loss
correlation after k folds on the matched per-sample loss (0-1 error for accuracy, NLL, squared error for Brier);
the curve prediction K / (1 + (K - 1) omega) and the implied coupling kappa = rho_delta - omega,
rho_delta = (K / G - 1) / (K - 1).

Run set: the redundancy runs of Table D.2 and Figure 6 of the paper (configs/redundancy_round3_50seeds_3sizes.txt).
Writes <out-dir>/redundancy_vs_gain.csv; analysis/out_round3_redundancy/redundancy_vs_gain.csv, tracked in this
repository, is its output on those runs and an input of Figure 6 (paper_style_scatter.py in benchmark_regression).

Usage (module pytorch-gpu/py3/2.8.0, or the packages of requirements.txt), from the repository root:
    python analysis/pcam_redundancy_vs_gain.py \
        --glob "results_output_pcam/pcam_results_CV_shuffle_split_unstrat_*_nsplits_20_seeds_1[0-4][0-9]-*_sizes_*.parquet" \
        --out-dir analysis/out_round3_redundancy
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

T = 0.2
LOSS = {"accuracy": ("test_accuracy", "benchmark_accuracy", "study_error01_rho_cum_oof_intersection"),
        "nll": ("test_nll", "benchmark_nll", "study_nll_rho_cum_oof_intersection"),
        "brier": ("test_brier", "benchmark_brier", "study_squared_error_rho_cum_oof_intersection")}


def load(pattern):
    files = sorted(glob.glob(pattern))
    key = lambda f: re.sub(r"_\d{8}_\d{6}\.parquet$", "", os.path.basename(f))
    latest = {}
    for f in files:  # file names end with a timestamp: keep the latest run of each (model, seeds, size)
        latest[key(f)] = max(latest.get(key(f), f), f)
    cols = ["model", "study_size", "seed", "fold"] + [c for v in LOSS.values() for c in v]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in latest.values()], ignore_index=True)
    return d, len(files), len(latest)


def gains(E):
    cm = E.mean(axis=1)
    vk = np.var(cm, ddof=1)
    return np.var(E[:, 0], ddof=1) / vk, np.var(E, axis=0, ddof=1).mean() / vk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--K", type=int, default=20)
    a = ap.parse_args()
    d, n_files, n_kept = load(a.glob)
    print(f"{n_files} files, {n_kept} kept (latest per model/seed/size); seeds {d.seed.min()}-{d.seed.max()}")
    K = a.K
    rows = []
    for loss, (q, b, rc) in LOSS.items():
        for (m, n), g in d.groupby(["model", "study_size"]):
            E = (g.assign(x=g[q].astype(float) - g[b].astype(float))
                 .pivot_table(index="seed", columns="fold", values="x").dropna().values[:, :K])
            if E.shape[1] < K:
                continue
            g_first, g_mean = gains(E)
            r = dict(loss=loss, model=m, size=n, seeds=len(E), G20_first=g_first, G20_mean=g_mean)
            for k in (3, 5):
                rho = g.loc[g.fold == k, rc].astype(float)
                r[f"omega{k}"] = T * rho.mean()
                r[f"cov{k}"] = int(rho.notna().sum())
            r["pred5"] = K / (1 + (K - 1) * r["omega5"])
            r["kappa5"] = (K / g_mean - 1) / (K - 1) - r["omega5"]
            rows.append(r)
    r = pd.DataFrame(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    r.to_csv(os.path.join(a.out_dir, "redundancy_vs_gain.csv"), index=False)
    pd.set_option("display.width", 200)
    print(r.round(3).to_string(index=False))
    for loss, g in list(r.groupby("loss")) + [("pooled", r)]:
        s = {f"{w}_vs_{G}": spearmanr(g[w], g[G]).correlation for w in ("omega3", "omega5") for G in ("G20_mean", "G20_first")}
        err = (g.pred5 / g.G20_mean).describe()[["min", "50%", "max"]]
        print(loss, {k: round(v, 2) for k, v in s.items()}, "| pred5/G20_mean min %.2f median %.2f max %.2f" % tuple(err))


if __name__ == "__main__":
    main()
