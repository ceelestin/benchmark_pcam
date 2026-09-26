#!/usr/bin/env python3
"""PCam: exact variance-equivalent test sample gains and study-only redundancy candidates
on the redundancy-calibration set (job 9433 + resubmit 18278: 4 models x 4 train sizes x
25 seeds, K = 20 ShuffleSplit folds, test fraction 0.2).

Usage (Jean-Zay login node, benchcv venv):
    python analysis/pcam_gain_analysis.py \
        --glob "$WORK/benchmark_pcam/results_output_pcam/pcam_results_CV_shuffle_split_unstrat_*_nsplits_20_seeds_*_sizes_*_2026092[12]_*.parquet" \
        --out-dir analysis/out_calibration --loss accuracy

Columns used (pcam_deep_training_adapted.py): per fold ``test_<loss>`` (fold score q_k) and
``benchmark_<loss>`` (the same model on the benchmarking set, b_k); on fold 1 the list
``hidden_test_<loss>s`` = scores of that model on up to 200 hidden chunks of the test-fold
size (the outer chunks); the cumulative study-only statistics
``study_{squared_error,nll,error01}_rho_cum_oof_intersection`` at every fold k.

Definitions follow analysis/derivation_levers_analysis.py in benchmark_regression:
G_K^paper = smallest test multiplier alpha whose pooled single-split error variance
Var_seeds(delta_HO(alpha)) drops below Var_seeds(Delta_K), Delta_K = mean_k q_k - mean_k b_k;
V1(alpha) is averaged exactly over the order of the hidden chunks (v1_alpha_curve, same as
benchmark_regression's derivation_levers_analysis); pooling them in one fixed order, as before
2026-09-26, adds noise shared by every configuration using the same chunks (kept as G_paper_fixed).
G_K^err = Var(q_1-b_1)/Var(qbar_K-bbar_K). Losses: accuracy (higher better; error01 rho is the
matched candidate), nll, brier (both lower-better: negated).

PRE-REGISTERED RULE (see benchmark_regression/analysis/PREREGISTRATION_f_rule.md): read at
fold k=3, f_hat = 1 - rho_cum(k) of the matched loss, predicted G_K = 1/[(1-f)(t+(1-t)/K)+f/K];
'stop' iff predicted G_20 < 10 (<=> rho_cum(3) > 0.2593). Checks: P(G_20 >= 10 | stop) <= 0.05
and the bound P(predicted G_20 >= observed G_20).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

T = 0.2
LOSS = {  # loss -> (test col, bench col, hidden list col, sign (+1 higher-better), matched rho cum col)
    "accuracy": ("test_accuracy", "benchmark_accuracy", "hidden_test_accuracies", +1, "study_error01_rho_cum_oof_intersection"),
    "nll": ("test_nll", "benchmark_nll", "hidden_test_nlls", -1, "study_nll_rho_cum_oof_intersection"),
    "brier": ("test_brier", "benchmark_brier", "hidden_test_briers", -1, "study_squared_error_rho_cum_oof_intersection"),
}
# DEVIATION FROM THE PRE-REGISTRATION (2026-09-23): the 391 job-9433 parquets predate the
# cumulative columns and only carry the K=20 broadcast value `study_<loss>_rho_all_oof_intersection`
# (the 9 resubmit parquets have both). The fold-k read of the rule is therefore only possible
# at k = 20 on this set; the threshold (rho > 0.2631 <=> stop) is unchanged. `--rho cum` uses
# the cumulative columns where they exist (9/400 cells) for reference.
RHO_COLS_ALL = {"rho_err01": "study_error01_rho_all_oof_intersection", "rho_nll": "study_nll_rho_all_oof_intersection",
                "rho_sq": "study_squared_error_rho_all_oof_intersection"}
RHO_COLS_CUM = {"rho_err01": "study_error01_rho_cum_oof_intersection", "rho_nll": "study_nll_rho_cum_oof_intersection",
                "rho_sq": "study_squared_error_rho_cum_oof_intersection"}
RHO_COLS = RHO_COLS_ALL
K_GRID = [2, 3, 5, 10, 20]


def G_closed(f, K, t=T):
    return 1.0 / ((1 - f) * (t + (1 - t) / K) + f / K)


def v1_alpha_curve(x0, X):
    """V1(alpha), alpha = 1..n+1, averaged over the order of the n chunks: the pooled error
    (x0 + sum_{j in S} x_j) / alpha, |S| = alpha - 1, has a variance across seeds that is a quadratic
    form, so its expectation over a uniformly random S is [C00 + 2 m c0 + m d + m (m-1) o] / alpha^2
    (m = alpha - 1; C = cov of [x0, X] with ddof=1; c0 mean C0j, d mean Cjj, o mean Cjl, j != l)."""
    Z = np.column_stack([x0, X])
    C = np.cov(Z, rowvar=False, ddof=1)
    n = X.shape[1]
    Cc = C[1:, 1:]
    d = np.trace(Cc) / n
    o = (Cc.sum() - np.trace(Cc)) / (n * (n - 1)) if n > 1 else 0.0
    m = np.arange(0, n + 1, dtype=float)
    return (C[0, 0] + 2 * m * C[0, 1:].mean() + m * d + m * (m - 1) * o) / (m + 1) ** 2


def paper_gain(var_k, alpha_var):
    v = np.asarray(alpha_var, dtype=float)
    below = np.where(v <= var_k)[0]
    if len(below) == 0:
        return float(len(v)), True
    j = int(below[0])
    if j == 0:
        return 1.0, False
    lv0, lv1 = np.log(v[j - 1]), np.log(v[j])
    frac = (lv0 - np.log(var_k)) / (lv0 - lv1) if lv0 > lv1 else 1.0
    return float(np.exp(np.log(j) + frac * (np.log(j + 1) - np.log(j)))), False


def load(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet matches {pattern}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[(df["n_splits"] == 20) & (df["cv_method"] == "shuffle_split")]
    return df


def analyse(df, loss):
    qc, bc, hc, sign, _ = LOSS[loss]
    rows, cands = [], []
    rng = np.random.default_rng(0)
    for (model, size), g in df.groupby(["model", "study_size"]):
        piv_q = g.pivot_table(index="seed", columns="fold", values=qc).dropna(axis=0)
        piv_b = g.pivot_table(index="seed", columns="fold", values=bc).loc[piv_q.index]
        n_seeds, K_avail = piv_q.shape
        if n_seeds < 5:
            continue
        Q, B = sign * piv_q.values, sign * piv_b.values
        E = Q - B
        first = g[g["fold"] == 1].set_index("seed").loc[piv_q.index]
        outer = sign * np.vstack([np.asarray(o, dtype=float) for o in first[hc].values])
        n_alpha = outer.shape[1] + 1
        cum = np.cumsum(outer, axis=1)
        pooled = np.concatenate([Q[:, :1], (Q[:, :1] + cum) / (np.arange(1, n_alpha)[None, :] + 1)], axis=1)
        delta_alpha_var_fixed = np.var(pooled - B[:, :1], axis=0, ddof=1)   # one fixed chunk order
        Xc = outer - B[:, :1]
        delta_alpha_var = v1_alpha_curve(E[:, 0], Xc)                      # order-averaged
        v1_err = np.var(E[:, 0], ddof=1)
        rec = {"model": model, "train_size": int(size), "n_seeds": n_seeds, "n_alpha": n_alpha, "m_test": int(round(size * T / (1 - T)))}
        for K in [k for k in K_GRID if k <= K_avail]:
            me = E[:, :K].mean(axis=1)
            g_err = v1_err / np.var(me, ddof=1)
            g_paper, capped = paper_gain(np.var(me, ddof=1), delta_alpha_var)
            g_paper_fixed, _ = paper_gain(np.var(me, ddof=1), delta_alpha_var_fixed)
            bs = []
            for _ in range(200):
                idx = rng.integers(0, n_seeds, n_seeds)
                bs.append(paper_gain(np.var(me[idx], ddof=1), v1_alpha_curve(E[idx, 0], Xc[idx]))[0])
            rows.append({**rec, "K": K, "G_paper": g_paper, "G_paper_capped": capped, "G_paper_fixed": g_paper_fixed,
                         "G_paper_lo": np.percentile(bs, 5), "G_paper_hi": np.percentile(bs, 95), "G_err": g_err})
            rec[f"G_paper_{K}"], rec[f"G_paper_fixed_{K}"], rec[f"G_err_{K}"] = g_paper, g_paper_fixed, g_err
        for k in (3, 5, 10, 20):
            sub = g[g["fold"] == k]
            for name, col in RHO_COLS.items():
                vals = sub[col].astype(float)
                rec[f"{name}_{k}"] = float(vals.median()) if vals.notna().any() else np.nan
                rec[f"{name}_{k}_nseeds"] = int(vals.notna().sum())
        cands.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(cands)


def rule_report(cand, loss):
    rho = {"accuracy": "rho_err01", "nll": "rho_nll", "brier": "rho_sq"}[loss]
    f_star = next(f for f in np.linspace(0, 1, 100001) if G_closed(f, 20) >= 10)
    print(f"\n=== pre-registered rule, loss={loss}, candidate {rho}: continue iff f_hat >= {f_star:.4f} (rho_cum(3) <= {1-f_star:.4f}) ===")
    for k in (3, 5, 20):
        if cand[f"{rho}_{k}"].isna().any():
            print(f"  k={k}: candidate missing for {int(cand[f'{rho}_{k}'].isna().sum())}/{len(cand)} configurations (seeds with the column: {cand[f'{rho}_{k}_nseeds'].min()}-{cand[f'{rho}_{k}_nseeds'].max()}) -> skipped")
            continue
        fh = 1 - cand[f"{rho}_{k}"]
        stop = fh < f_star
        large = cand["G_paper_20"] >= 10
        pred = G_closed(fh, 20)
        print(f"  k={k}: configurations={len(cand)} stop rate={stop.mean():.2f} | P(G_20>=10 | stop)={large[stop].mean() if stop.any() else float('nan'):.3f} "
              f"| P(G_20>=10 | continue)={large[~stop].mean() if (~stop).any() else float('nan'):.2f} | large-gain configs stopped={int((stop & large).sum())}/{int(large.sum())} "
              f"| bound P(pred G_20 >= obs)={(pred >= cand['G_paper_20']).mean():.2f}, median log(pred/obs)={np.log(pred / cand['G_paper_20']).median():+.2f}")
    from scipy.stats import spearmanr
    for name in RHO_COLS:
        ok = cand[[f"{name}_20", "G_paper_20"]].dropna()
        print(f"  Spearman({name}_20, G_paper_20) = {spearmanr(ok.iloc[:, 0], ok.iloc[:, 1]).statistic:+.2f} (n={len(ok)})", end="")
    print()
    pd.set_option("display.width", 250)
    print(cand.pivot_table(index="model", columns="train_size", values=["G_paper_20", "G_paper_5", f"{rho}_20"], aggfunc="median").round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--loss", default="accuracy", choices=list(LOSS))
    ap.add_argument("--rho", default="all", choices=["all", "cum"], help="all: K=20 broadcast column (available on all 400 cells); cum: cumulative column (9 cells)")
    args = ap.parse_args()
    global RHO_COLS
    RHO_COLS = RHO_COLS_ALL if args.rho == "all" else RHO_COLS_CUM
    os.makedirs(args.out_dir, exist_ok=True)
    df = load(args.glob)
    print(f"rows={len(df):,} models={df.model.nunique()} sizes={sorted(df.study_size.unique())} seeds={df.seed.nunique()} K={df.n_splits.iloc[0]}")
    gains, cand = analyse(df, args.loss)
    gains.to_csv(os.path.join(args.out_dir, f"gains_{args.loss}.csv"), index=False)
    cand.to_csv(os.path.join(args.out_dir, f"candidates_{args.loss}.csv"), index=False)
    print(f"capped G_paper_20: {int(gains[gains.K == 20].G_paper_capped.sum())} of {int((gains.K == 20).sum())}")
    rule_report(cand, args.loss)
    print(f"\nwrote {args.out_dir}/gains_{args.loss}.csv, candidates_{args.loss}.csv")


if __name__ == "__main__":
    main()
