#!/bin/bash

# Set config file path
# In your terminal, first run "export CONFIG_PATH="your_config_path"

echo "Running with config file: $CONFIG_PATH"
export NUM_CPUS_DATA_LOADING=50
export NUM_CPUS_PREPROCESSING=50
export NUM_CPUS_TRIPLET_EXTRACTION=10
export NUM_CPUS_TRIPLET_PROCESSING=50

# Submit the first job (load_data.sh) and capture the job ID
JOB1_ID=$(sbatch --cpus-per-task=$NUM_CPUS_DATA_LOADING scripts/armasuisse_cluster/triplet_extraction/pipeline/pipeline_load_data.sh | awk '{print $4}')

# Submit the first job (load_data.sh) and capture the job ID
JOB2_ID=$(sbatch --dependency=afterok:$JOB1_ID --cpus-per-task=$NUM_CPUS_PREPROCESSING scripts/armasuisse_cluster/triplet_extraction/pipeline/pipeline_preprocessing.sh | awk '{print $4}')

# Submit the second job (extract_triplets.sh) with a dependency on the first job
JOB3_ID=$(sbatch --dependency=afterok:$JOB2_ID --cpus-per-task=$NUM_CPUS_TRIPLET_EXTRACTION scripts/armasuisse_cluster/triplet_extraction/pipeline/pipeline_extract_triplets.sh | awk '{print $4}')

# Submit the third job (post_processing.sh) with a dependency on the first job
JOB4_ID=$(sbatch --dependency=afterok:$JOB3_ID --cpus-per-task=$NUM_CPUS_TRIPLET_PROCESSING scripts/armasuisse_cluster/triplet_extraction/pipeline/pipeline_process_triplets.sh | awk '{print $4}')
