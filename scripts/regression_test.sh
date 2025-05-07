#!/bin/bash

#SBATCH --job-name=regression_test
#SBATCH --output=run_logs/%x-%j.out
#SBATCH --error=run_logs/%x-%j.err
#SBATCH --time=20-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/stea/llm_resilient_bibliometrics/tfsc2025

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate triplet_extraction

export REGRESSION_TEST=1

# Run the regression test
pytest src/regression_test/test_pipeline.py

conda deactivate
