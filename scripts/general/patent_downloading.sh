#!/bin/bash

#SBATCH --job-name=download_patents
#SBATCH --output=run_logs/%x-%j.out
#SBATCH --error=run_logs/%x-%j.err
#SBATCH --time=20-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:0
#SBATCH --chdir=/cluster/raid/home/stea/llm_resilient_bibliometrics
# eval "$(conda shell.bash hook)"


years=("2018" "2019" "2020" "2021" "2022" "2023" "2024")  # Add more years as needed

# Base URL
base_url="https://bulkdata.uspto.gov/data/patent/application/redbook/fulltext/"

# Loop through each year and run wget command
for year in "${years[@]}"
do
    echo "Downloading ZIP files for year $year..."

    # Construct the URL for the current year
    url="${base_url}${year}/"

    # check if output directory exists, if not create it
    mkdir -p /cluster/raid/data/stea/full_text_uspo_patents_applications/"$year"

    # same command now also specifying the output directory, last folder should be the year
    wget -r -np -l1 -nd -A zip -P /cluster/raid/data/stea/full_text_uspo_patents_applications/"$year" "$url"

    echo "Download complete for year $year."
done

echo "All ZIP files downloaded successfully, unzipping..."

# Unzip all downloaded ZIP files, remove the ZIP files after unzipping
for year in "${years[@]}"
do
    echo "Unzipping files for year $year..."

    # Unzip all ZIP files in the current year directory, avoid the error caution filename not matched
    for file in /cluster/raid/data/stea/full_text_uspo_patents_applications/"$year"/*.zip
    do
        unzip -o "$file" -d /cluster/raid/data/stea/full_text_uspo_patents_applications/"$year"
    done

    # Remove all ZIP files in the current year directory
    rm /cluster/raid/data/stea/full_text_uspo_patents_applications/"$year"/*.zip

    echo "Unzipping complete for year $year."
done

