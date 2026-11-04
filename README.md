# PatchCamelyon runs of "Crossing the Validation Crisis"

Training loop and run configurations of the PatchCamelyon (PCam) experiments of

> Célestin Eve, Gaël Varoquaux and Thomas Moreau (2026). *Crossing the Validation Crisis: Cross-Validation
> Reduces Benchmarking Variance Surprisingly Well.*

The entry point to the code of the paper is
[ceelestin/benchmark_regression](https://github.com/ceelestin/benchmark_regression): it holds the simulated
benchmark and the scripts that draw every figure and table of the paper from the run outputs, the PCam and Yelp
ones included. This repository trains the PCam networks and writes those outputs; its companion
[ceelestin/benchmark_nlp](https://github.com/ceelestin/benchmark_nlp) does the same for Yelp Review Full.

The training script is adapted from a training script shared by the
[TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox) authors (Pocock et al., 2022); it does not
call TIAToolbox.

| Path | Contents |
|---|---|
| `pcam_deep_training_adapted.py` | the training loop: one call trains and scores every (seed, network, size) given on its command line |
| `configs/*.txt` | the run sets of the paper: one line = the arguments of one job (table below) |
| `configs/legacy/` | earlier and exploratory run sets, not used in the paper |
| `launch.sh` | SLURM array launcher: task *i* runs line *i* of a configuration file |
| `analysis/pcam_gain_analysis.py` | variance-equivalent sample gains with bootstrap intervals, and the redundancy after *k* splits (Table D.2) |
| `analysis/pcam_redundancy_vs_gain.py` | redundancy against sample gain per network, size and metric (Table D.2, Figure 6) |
| `analysis/out_round3_redundancy/redundancy_vs_gain.csv` | output of the previous script on the runs of Table D.2; an input of Figure 6 |
| `analysis/pcam_set_sizes.py` | reads the test-fold, validation and outer-chunk sizes off run outputs (checks which size convention a run used) |

## What the script does

Appendix F.3 of the paper describes the protocol; in short:

* The official training and validation splits of PCam (294,912 images of 96 x 96 pixels) are merged. A fixed
  stratified split keeps 244,912 images as the **benchmarking set**, which stands for the population, and leaves a
  **pool** of 50,000.
* For each seed, a stratified **study set** of 1.25 n_tr images is drawn from the pool (`--train-sizes` gives n_tr,
  the training and validation subsets together); the rest of the pool is the **outer set**.
* The study set is split K times (`--n-splits`), by ShuffleSplit (Monte Carlo CV) or repeated 5-fold CV
  (`--cv-method`), with a test fold of a fifth, optionally stratified (`--stratify`). In each split, an
  ImageNet-pretrained `torchvision` network (DenseNet121, MobileNetV2, WideResNet101_2 or ViT-B/16; the head is
  replaced, nothing is frozen) is trained with Adam on three fifths of the study set for `--epochs` epochs, and the
  epoch with the best accuracy on the remaining fifth (validation) is kept.
* The kept model is scored (accuracy, negative log-likelihood, Brier score) on the validation subset, the test fold
  and the benchmarking set, or a fixed stratified subset of it (`--bench-max-samples`). The model of the first split
  also scores the outer set, in chunks of the test-fold size (at most 200).
* The study-only redundancy statistics of Appendix D.2 (per-sample loss correlation on the observations that two
  splits both test) are computed from the per-sample test losses, after each split and over all K.
* Each run writes one Parquet file, one row per split, to `results_output_pcam/`.

`--no-cross-validation` gives the single-split mode of the January 2026 runs (every split scores the outer set);
see the last row of the table below for how those runs were made.

## Install

Python 3.12 and one GPU per job (the paper's runs used one NVIDIA V100 each):

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions of the Jean-Zay module `pytorch-gpu/py3/2.8.0` (torch 2.8.0, torchvision
0.23.0, datasets 4.0.0, scikit-learn 1.7.1), which every PCam run of the paper since 2026-03-29 loaded; install
the torch wheel that matches your CUDA. On Jean-Zay, `module load pytorch-gpu/py3/2.8.0` is enough.

## Data

The script reads `./pcam_data/`, a Hugging Face cache of the dataset
[`1aurent/PatchCamelyon`](https://huggingface.co/datasets/1aurent/PatchCamelyon) (about 7 GB of Arrow files).
If that folder holds no Arrow file, the script downloads the dataset into it on its first call, at the revision
of the paper's runs. As cluster compute nodes are often offline, prepare it once from the repository root on a
node with internet access:

```bash
python -c "from datasets import load_dataset; load_dataset('1aurent/PatchCamelyon', revision='e4bd149e7a868a9d811fdd9f9a9fb78c05c104ab', cache_dir='pcam_data')"
```

The script then reads `./pcam_data/` as a local folder, which writes a second Arrow copy (about 7 GB) to the
`datasets` cache (`$HF_DATASETS_CACHE`, set in `launch.sh`), and caches the labels in `pcam_data/all_labels.npz`
on its first run. `torchvision` downloads the ImageNet weights into `$TORCH_HOME`; for offline nodes, fetch them
beforehand with the same `TORCH_HOME`:

```bash
python -c "import torchvision.models as m; [getattr(m, a)(weights='IMAGENET1K_V1') for a in ('densenet121', 'mobilenet_v2', 'wide_resnet101_2', 'vit_b_16')]"
```

**Terms of use.** PatchCamelyon is released under CC0 1.0, following the license of the Camelyon16 data it is
built from. Its authors ask users to cite

* B. S. Veeling, J. Linmans, J. Winkens, T. Cohen and M. Welling. Rotation Equivariant CNNs for Digital
  Pathology. *MICCAI*, 2018 ([arXiv:1806.03962](https://arxiv.org/abs/1806.03962));
* B. Ehteshami Bejnordi et al. Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node
  Metastases in Women With Breast Cancer. *JAMA*, 318(22):2199-2210, 2017.

The pretrained weights come from `torchvision` (BSD-3-Clause), trained on ImageNet, whose terms of access restrict
its use to non-commercial research and education. The data are not redistributed here.

## Run one job

From the repository root:

```bash
python pcam_deep_training_adapted.py --n-splits 20 --cv-method shuffle_split --seeds 0 1 \
    --train-sizes 400 --model-choices mobilenet_v2 --epochs 30 --batch-size 64
```

This is one line of `configs/mccv_cnns.txt`: MobileNetV2, seed 0, n_tr = 400 (a study set of 500
images, test folds of 100), K = 20 splits. It writes
`results_output_pcam/pcam_results_CV_shuffle_split_unstrat_mobilenet_v2_nsplits_20_seeds_0-0_sizes_400_<date>_<time>.parquet`.
For a quick test, use `--n-splits 4 --epochs 1 --train-sizes 80 --bench-max-samples 60000`.

| Argument | Meaning |
|---|---|
| `--train-sizes N [N ...]` | n_tr, the training and validation subsets together; the study set has `round(N / 0.8)` images |
| `--n-splits K` | number of splits; for `repeated_kfold`, K / 5 repeats of 5-fold CV |
| `--cv-method` | `shuffle_split` (Monte Carlo CV, default) or `repeated_kfold` |
| `--stratify` | stratify the splits (and the outer chunks) by label |
| `--seeds START END` | seeds START to END - 1; a seed fixes the study set, the splits and the initialisation |
| `--model-choices` | `torchvision` networks (`densenet121`, `mobilenet_v2`, `wide_resnet101_2`, `vit_b_16`, ...) |
| `--epochs`, `--batch-size` | 30 and 64 in every run of the paper |
| `--bench-max-samples M` | score a fixed stratified subset of M benchmarking images (0, the default: all 244,912) |
| `--no-cross-validation` | single-split mode: every split scores the outer set |

Main output columns: `model`, `seed`, `fold`, `study_size` (n_tr), `study_pool_size` (study set);
`{val,test,benchmark}_{accuracy,nll,brier}` and `_runtime`; `hidden_test_{accuracies,nlls,briers}` (the outer
chunks, on the first split); `study_{error01,nll,squared_error}_rho_{cum,all}_oof_intersection` (the redundancy
after k splits and after all K; squared error is the per-sample Brier score).

## From a configuration file to a SLURM array

Each line of a file of `configs/` holds the arguments of one job, usually one seed, one network and one size.
`launch.sh` runs line `$SLURM_ARRAY_TASK_ID` of the file given in `$CONFIG`:

```bash
mkdir -p slurm_output                       # SLURM writes the logs there and does not create it
sbatch --array=1-750 --export=ALL,CONFIG=configs/mccv_cnns.txt launch.sh
```

The array size is the number of lines (`grep -c . FILE`). Before submitting, set `#SBATCH --account=<project>@v100`
in `launch.sh`, and adapt the partition, the time limit (6 h; ViT-B/16 on the full benchmarking set is the slowest)
and the module line to your cluster. The launcher sets `HF_HOME`, `HF_DATASETS_CACHE` and `TORCH_HOME` under `$WORK`,
the work space of Jean-Zay.

## Run sets behind the paper

Figure and table numbers are those of the paper. n_tr counts the training and validation subsets together; the
study set, in parentheses, has 1.25 n_tr images. Networks: D DenseNet121, M MobileNetV2, W WideResNet101_2,
V ViT-B/16. Every run uses 30 epochs and batches of 64.

| Paper | Configuration file (lines) | Networks | n_tr (study set) | Seeds | Splits | Script that ran it |
|---|---|---|---|---|---|---|
| Figures 2 (M, n_tr = 400), 5 (middle) and D.1; Tables E.2 and N.1; the K = 20 column of Table M.1; Table F.2 | `mccv_cnns.txt` (750), `mccv_vit.txt` (250) | D, M, W; V | 24, 80, 240, 400, 640 (30-800) | 0-49 | 20, ShuffleSplit | `0d97538` and `ccdbba8`, 2026-03-29 (`ccdbba8` adds ViT-B/16 and leaves the CNN code unchanged) |
| Figure I.1 and Table E.5 (n_tr = 240), against the ShuffleSplit runs above | `repeated_kfold_cnns.txt` (750) | D, M, W | 24-640 (30-800) | 0-49 | 20 = 4 x 5-fold | `0d97538`, 2026-03-29 |
| Table J.1, Figures J.1 and K.1, Tables E.6 and E.7 | unstratified: the lines without `--stratify` of `stratification_and_newmetrics_full.txt` and `stratification_and_newmetrics_full_newseeds.txt` (200 + 200); stratified: `stratificationONLY_and_newmetrics_full.txt` (400) | D, M, W, V | 400 (500) | 0-99 | 20, ShuffleSplit | `92f131c` (unstratified, 2026-07-25), `f50dff6` (stratified, 2026-07-26) |
| Appendix L (PCam at n_tr = 24); Table F.2 | `redundancy_smalldata.txt`: its 200 lines of n_tr = 24 (the 200 lines of n_tr = 40 feed no result) | D, M, W, V | 24 (30) | 0-49 | 20, ShuffleSplit | `6f7f568`, 2026-07-31 |
| Table D.2; Figure 6 (PCam points); Table F.2 | `redundancy_round3_50seeds_3sizes.txt` (600), oracle on a fixed subset of 60,000 benchmarking images | D, M, W, V | 100, 300, 800 (125, 375, 1,000) | 100-149 | 20, ShuffleSplit | `11148cb`, 2026-09-24 |
| Table M.1 (t-test and single-split columns); Table F.2 (single-split rows) | `configs_adapted_100seeds_5study.txt` (1,500) | D, M, W | study sets 30, 100, 300, 500, 800; test and validation subsets of a sixth each, so n_tr = 25, 83, 250, 416, 666 | 0-99 | 1 (`--no-cross-validation`) | `a2a9fd3`, 2026-01-15 |

**Reproducing them with the current script.** The first five rows run as they are: the current script rebuilds
the same study sets and splits, and later commits only added output columns, the optional `--bench-max-samples`
(off by default) and the n_tr label of `study_size` (GPU training is not bitwise deterministic, so scores match
in distribution, not to the last digit). Before commit `e194b00` (2026-08-01), `--train-sizes` was
called `--study-sizes` and gave the study-set size; that commit renamed the flag without changing the numbers, so
the files of the first four rows now give n_tr = 0.8 x the study set of their runs (24, 40, 80, 240, 400, 640).
In the outputs of those runs, `study_size` holds the study set and `study_pool_size` is absent
(`analysis/pcam_set_sizes.py` tells the two conventions apart). Re-running the stratified lines of the two
`stratification_and_newmetrics_full*.txt` files gives the stratified arm again (with the stratified outer chunks
of `f50dff6`), so these files can also be run whole.

The single-split runs of the last row held out a sixth of each study set for testing and a sixth for validation,
before the protocol change of `34df534` (2026-03-29); the current script cannot reproduce them and rejects their
file (`--study-sizes` is unknown to it). To re-run them, check out `a2a9fd3`, where the script and the file
(then at the repository root, with `launch.sh` pointing at it) are as they ran; the version of the unpinned
`pytorch-gpu` module they loaded is not recorded:

```bash
git checkout a2a9fd3
python pcam_deep_training_adapted.py --no-cross-validation --seeds 0 1 --study-sizes 30 --model-choices densenet121 --epochs 30 --batch-size 64
```

The run outputs are not part of this repository.

## License

BSD-3-Clause (see `LICENSE`). The datasets and pretrained weights keep their own terms (see Data).
