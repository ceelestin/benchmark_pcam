#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH -C v100
#SBATCH --gres=gpu:1
#SBATCH --account=lsd@v100
#SBATCH --cpus-per-gpu 8
#SBATCH --partition=gpu_p13
#SBATCH --job-name=ceve_pcam
#SBATCH --output=slurm_output/pcam_%a.out
#SBATCH --error=slurm_output/pcam_%a.out

module purge
module load pytorch-gpu

params=$(awk -v  idx_param="${SLURM_ARRAY_TASK_ID}" 'NR==idx_param' configs2.txt)

python pcam_deep_training.py $params