#!/bin/bash

#SBATCH --job-name=preprocess
#SBATCH --output=run_logs/%x-%j.out
#SBATCH --error=run_logs/%x-%j.err
#SBATCH --time=20-00:00:00
#SBATCH --mem=125G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:0
#SBATCH --chdir=/cluster/raid/home/stea/llm_resilient_bibliometrics

# Initialize the shell to use local conda
echo "Preparing shell for conda"
eval "$(conda shell.bash hook)"
echo "Conda is initialized, now activating environment"

# Activate (local) env
conda activate triplet_extraction

echo "Conda is activated and ready, running code now"
# python -m spacy download en_core_web_lg
python src/preprocessing.py

conda deactivate
