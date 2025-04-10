#!/bin/bash
#SBATCH -J c-learn-curr
#SBATCH --time=00-02:00:00
#SBATCH -p preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --gres=gpu:h100:1
#SBATCH --output=c-curr-learn.%j.out
#SBATCH --error=c-curr-learn.%j.err

hostname
date

module purge
module load anaconda/2024.10 cuda/12.2 cudnn/9.5.0.50-12.x
module list

source activate sb3

~/condaenv/sb3/bin/python baseline_agent.py train curriculum

conda deactivate
