"""Set sizes of PCam runs, read off their parquets (CE, 2026-11-02).

An accuracy measured on n images is a multiple of 1/n, so the smallest n that makes every recorded value of a column
an integer count is the size of the set behind it: the test fold (`test_accuracy`), the validation subset
(`val_accuracy`) and the outer chunks (`hidden_test_accuracies`). Set against the size label (`study_size`), it shows
which convention a run followed:
  * runs before commit e194b00 (2026-08-01): the label is the study set; the test fold is a fifth of it (a sixth
    before 2026-03-29) and n_tr, the training and validation subsets together, is the label minus the test fold;
  * later runs: the label is n_tr, the study set (column `study_pool_size`) has 1.25 x label observations and the
    test fold label / 4.

Usage (Jean-Zay, module load pytorch-gpu/py3/2.8.0):
    python analysis/pcam_set_sizes.py DIR_OR_GLOB [DIR_OR_GLOB ...]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def set_size(values, n_max=3000, tol=1e-6):
    """Smallest n such that every value is a multiple of 1/n (accuracies in [0, 1] or in percent)."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size == 0:
        return None
    if v.max() > 1.0 + 1e-9:
        v = v / 100.0
    v = np.unique(np.round(v, 12))
    for n in range(1, n_max + 1):
        if np.all(np.abs(v * n - np.round(v * n)) < tol):
            return n
    return None


def files_of(arg):
    if os.path.isdir(arg):
        return sorted(glob.glob(os.path.join(arg, "**", "*.parquet"), recursive=True))
    return sorted(glob.glob(arg))


def load(files):
    cols = ["model", "study_size", "study_pool_size", "cv_method", "seed", "test_accuracy", "val_accuracy",
            "hidden_test_accuracies"]
    frames = []
    for f in files:
        names = pq.read_schema(f).names
        d = pd.read_parquet(f, columns=[c for c in cols if c in names])
        stamp = re.search(r"_(\d{8})_\d{6}\.parquet$", f)
        d["month"] = stamp.group(1)[:6] if stamp else "?"
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def summarise(d):
    if "cv_method" not in d.columns:
        d["cv_method"] = "single split"
    if "study_pool_size" not in d.columns:
        d["study_pool_size"] = np.nan
    rows = []
    for (month, model, cv, label), g in d.groupby(["month", "model", "cv_method", "study_size"]):
        hidden = [np.asarray(h, float) for h in g["hidden_test_accuracies"]
                  if isinstance(h, (list, np.ndarray)) and len(h)] if "hidden_test_accuracies" in g else []
        n_te = set_size(g["test_accuracy"].values)
        pool = g["study_pool_size"].dropna().unique()
        label_is_ntr = len(pool) > 0
        study = int(pool[0]) if label_is_ntr and len(pool) == 1 else (None if label_is_ntr else int(label))
        rows.append({"month": month, "model": model, "cv": cv, "label": int(label),
                     "seeds": f"{g.seed.min()}-{g.seed.max()} ({g.seed.nunique()})",
                     "test fold": n_te, "validation": set_size(g["val_accuracy"].values),
                     "chunk": set_size(np.concatenate(hidden)) if hidden else None,
                     "label is": "n_tr" if label_is_ntr else "study set",
                     "study set": study, "n_tr": int(label) if label_is_ntr else (int(label) - n_te if n_te else None)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="directories (searched recursively) or globs of run parquets")
    a = ap.parse_args(argv)
    for src in a.sources:
        files = files_of(src)
        print(f"== {src}: {len(files)} files")
        if files:
            print(summarise(load(files)).to_string(index=False))


if __name__ == "__main__":
    main()
