#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH -C v100
#SBATCH --gres=gpu:1
#SBATCH --account=sth@v100
#SBATCH --cpus-per-gpu 8
#SBATCH --partition=gpu_p13
#SBATCH --job-name=ceve_pcam
#SBATCH --output=slurm_output/pcam_%a.out
#SBATCH --error=slurm_output/pcam_%a.out

module purge
module load pytorch-gpu

# The Hugging Face `datasets` cache defaults to ~/.cache/huggingface, i.e. $HOME,
# which has a tiny quota on Jean-Zay and overflows when building the PCam Arrow
# cache (-> "OSError: [Errno 122] Disk quota exceeded"). Point it at the prepared
# cache already sitting in $WORK/hf_cache/datasets/pcam_data so load_dataset finds
# it and skips regeneration entirely.
export HF_HOME=$WORK/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets

params=$(awk -v  idx_param="${SLURM_ARRAY_TASK_ID}" 'NR==idx_param' configs/stratification_and_newmetrics.txt)

python pcam_deep_training_adapted.py $params