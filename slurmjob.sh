#!/bin/bash
#SBATCH --job-name=QEC-QELM
#SBATCH --time=30-00:00:00
#SBATCH --output=%job.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=64G

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python code_experiment_cluster.py