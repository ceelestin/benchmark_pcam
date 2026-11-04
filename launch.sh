#!/bin/bash
# SLURM array launcher: array task i runs pcam_deep_training_adapted.py with the arguments on line i of $CONFIG.
# Submit from the repository root (every path is relative to it), for example
#   mkdir -p slurm_output
#   sbatch --array=1-750 --export=ALL,CONFIG=configs/rebuttal_shufflesplit.txt launch.sh
# The array size is the number of lines of the file (wc -l counts one less when the last line has no newline).
# Written for the V100 partition of Jean-Zay (IDRIS): set your account below, and adapt the #SBATCH lines and the
# module line to your cluster.
#SBATCH --time=06:00:00
#SBATCH -C v100
#SBATCH --gres=gpu:1
#SBATCH --account=<project>@v100
#SBATCH --cpus-per-gpu 8
#SBATCH --partition=gpu_p13
#SBATCH --job-name=pcam
#SBATCH --output=slurm_output/pcam_%A_%a.out
#SBATCH --error=slurm_output/pcam_%A_%a.out

if [ -z "${CONFIG}" ]; then
    echo "Set CONFIG to a configuration file, e.g. sbatch --export=ALL,CONFIG=configs/rebuttal_shufflesplit.txt ..." >&2
    exit 1
fi

module purge
module load pytorch-gpu/py3/2.8.0   # the module of every PCam run since 2026-03-29 (see requirements.txt)

# Caches on a persistent file system ($WORK on Jean-Zay; the default ~/.cache sits on the small $HOME quota).
# The compute nodes have no internet access: ./pcam_data and the ImageNet weights in $TORCH_HOME must be
# downloaded beforehand from a node that has it (see README.md).
export HF_HOME=$WORK/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$WORK/torch_cache

params=$(awk -v idx_param="${SLURM_ARRAY_TASK_ID}" 'NR==idx_param' "${CONFIG}")

python pcam_deep_training_adapted.py $params
