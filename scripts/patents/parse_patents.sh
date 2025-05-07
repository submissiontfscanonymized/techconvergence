#!/bin/bash

#SBATCH --job-name=parse_patents
#SBATCH --output=run_logs/%x-%j.out
#SBATCH --error=run_logs/%x-%j.err
#SBATCH --time=20-00:00:00
#SBATCH --cpus-per-task=25
#SBATCH --mem=100G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:0
#SBATCH --chdir=/cluster/raid/home/stea/llm_resilient_bibliometrics

echo "Running with config file: $CONFIG_PATH"
export NUM_CPUS_PATENT_PARSING=25

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate triplet_extraction

python src/parse_patent.py

conda deactivate
