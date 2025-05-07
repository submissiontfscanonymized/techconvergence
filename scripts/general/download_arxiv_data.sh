#!/bin/bash

#SBATCH --job-name=download_arxiv_data
#SBATCH --output=run_logs/%x-%j.out
#SBATCH --error=run_logs/%x-%j.err
#SBATCH --time=20-00:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:0
#SBATCH --chdir=/cluster/raid/home/stea/resilient_forecasting

export GOOGLE_APPLICATION_CREDENTIALS=/cluster/raid/home/stea/llm_resilient_bibliometrics/data/google_cloud_key.json

DATA_PATH=/cluster/raid/data/stea/arxiv/

#./nix-portable nix-shell -p google-cloud-sdk --command "gcloud auth login"

for i in {2403..2412}
do
    ./nix-portable nix-shell -p google-cloud-sdk --command "gsutil -m cp -r gs://arxiv-dataset/arxiv/arxiv/pdf/$i/ $DATA_PATH"
done
