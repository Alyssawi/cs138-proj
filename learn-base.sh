#!/bin/bash
#SBATCH -J c-learn-base
#SBATCH --time=00-02:00:00
#SBATCH -p preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --gres=gpu:h100:1
#SBATCH --output=base/learn.%j.out
#SBATCH --error=base/learn.%j.err

hostname
date

module purge
module load anaconda/2024.10 cuda/12.2 cudnn/8.9.7-12.x
module list

source activate sb3

export CHECKPOINTS_DIR=base/checkpoints-$SLURM_JOB_ID
export LOGS_DIR=base/logs
~/condaenv/sb3/bin/python baseline_agent.py train

conda deactivate
