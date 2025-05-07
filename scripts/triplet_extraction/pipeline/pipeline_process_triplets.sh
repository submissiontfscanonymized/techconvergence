#!/bin/bash

#SBATCH --job-name=process_triplets
#SBATCH --output=run_logs/%x-%j.out
#SBATCH --error=run_logs/%x-%j.err
#SBATCH --time=20-00:00:00
#SBATCH --mem=200G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:0
#SBATCH --chdir=/cluster/raid/home/stea/llm_resilient_bibliometrics

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate triplet_extraction

if ! python -c "import spacy; spacy.load('en_core_web_lg')" 2>/dev/null; then
    echo "Downloading en_core_web_lg..."
    python -m spacy download en_core_web_lg
fi

python src/process_triplets.py
python src/filter_triplets.py


conda deactivate
