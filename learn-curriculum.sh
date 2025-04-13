#!/bin/bash
#SBATCH -J c-learn-curr
#SBATCH --time=00-02:00:00
#SBATCH -p preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --gres=gpu:h100:1
#SBATCH --output=curr-softmax-noise-2/learn.%j.out
#SBATCH --error=curr-softmax-noise-2/learn.%j.err

hostname
date

module purge
module load anaconda/2024.10 cuda/12.2 cudnn/8.9.7-12.x
module list

source activate sb3

export LOGS_DIR=curr-softmax-noise-2/logs
~/condaenv/sb3/bin/python baseline_agent.py train curriculum --softmax --noise

conda deactivate
