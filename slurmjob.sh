#!/bin/bash
#SBATCH --job-name=QEC-QELM
#SBATCH --time=30-00:00:00
#SBATCH --output=%job.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=64G

python code_experiment_cluster.py