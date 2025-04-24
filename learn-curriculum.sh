#!/bin/bash
#SBATCH -J c-learn-curr
#SBATCH --time=00-02:00:00
#SBATCH -p preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --gres=gpu:l40:1
#SBATCH --output=curr/learn.%j.out
#SBATCH --error=curr/learn.%j.err

hostname
date

module purge
module load anaconda/2024.10 cuda/12.2 cudnn/8.9.7-12.x
module list

source activate sb3

export LOGS_DIR=curr/logs
~/condaenv/sb3/bin/python agent.py train curriculum

conda deactivate
