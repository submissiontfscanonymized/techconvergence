
import os
os.environ["REGRESSION_TEST"] = "1"

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from extraction.helpers import logger, remove_folder
from extraction.contexts import GeneralContext, DataLoadingContext, TripletExtractionContext, TripletProcessingContext, TripletFilteringContext
import pytest
import subprocess
import json

GENERAL_CONTEXT = GeneralContext()
DATA_LOADING_CONTEXT = DataLoadingContext()
TRIPLET_EXTRACTION_CONTEXT = TripletExtractionContext(GENERAL_CONTEXT)
TRIPLET_PROCESSING_CONTEXT = TripletProcessingContext(GENERAL_CONTEXT)
TRIPLET_FILTERING_CONTEXT = TripletFilteringContext(GENERAL_CONTEXT)

@pytest.fixture
def number_of_files():
    return len(list(GENERAL_CONTEXT.path_raw_pdf.glob('*.pdf')))

# clean up some folders after the test
@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests(request):
    # This code will run after all tests have completed
    def remove_folders():
        remove_folder(GENERAL_CONTEXT.path_results_folder)
        remove_folder(GENERAL_CONTEXT.path_save_text_folder)

    request.addfinalizer(remove_folders)  # Ensures the cleanup happens at the end

def test_load_data(number_of_files):
    assert os.getenv("REGRESSION_TEST") == "1", "The environment variable REGRESSION_TEST is not set to 1"
    result = subprocess.run(["python", "src/extraction_pipeline/load_data.py"])
    assert result.returncode == 0, f"Loading the data failed, the error is: {result.stderr}"
    # open path_regression_test_folder and check if the data is there
    with open(GENERAL_CONTEXT.path_regression_test_folder.joinpath('load_data_test_output.json'), 'r') as f:
        data = json.load(f)

    assert data['time_taken'] < 1000, f"Loading the data took too long, it took {data['time_taken']} seconds"
    assert data['num_files_in_raw_texts'] == number_of_files, f"Number of files in raw texts is incorrect, expected {number_of_files}, got {data['num_files_in_raw_texts']}"

def test_preprocessing(number_of_files):
    result = subprocess.run(["python", "src/extraction_pipeline/preprocessing.py"])
    assert result.returncode == 0, f"preprocessing failed, the error is: {result.stderr}"
    # open path_regression_test_folder and check if the data is there
    with open(GENERAL_CONTEXT.path_regression_test_folder.joinpath('preprocessing_test_output.json'), 'r') as f:
        data = json.load(f)

    assert data['time_taken'] < 1000, f"preprocessing took too long, it took {data['time_taken']} seconds"
    assert data['num_files_in_processed_texts'] == number_of_files, f"Number of files in processed_texts is incorrect, expected {number_of_files}, got {data['num_files_in_processed_texts']}"

def test_triplet_extraction(number_of_files):
    result = subprocess.run(["python", "src/extraction_pipeline/extract_triplets_llm.py"])
    assert result.returncode == 0, f"Extracting triplets failed, the error is: {result.stderr}"
    # open path_regression_test_folder and check if the data is there
    with open(GENERAL_CONTEXT.path_regression_test_folder.joinpath('extract_triplets_test_output.json'), 'r') as f:
        data = json.load(f)

    assert data['time_taken'] < 1000, f"triplet extraction took too long, it took {data['time_taken']} seconds"
    assert data['num_files'] == number_of_files, f"Number of files in triplets csv is incorrect, expected {number_of_files}, got {data['num_files']}"

def test_triplet_processing(number_of_files):
    result = subprocess.run(["python", "src/extraction_pipeline/process_triplets.py"])
    assert result.returncode == 0, f"Processing the triplets failed, the error is: {result.stderr}"
    # open path_regression_test_folder and check if the data is there
    with open(GENERAL_CONTEXT.path_regression_test_folder.joinpath('process_triplets_test_output.json'), 'r') as f:
        data = json.load(f)

    assert data['time_taken'] < 1000, f"Triplet processing took too long, it took {data['time_taken']} seconds"
    assert data['num_files'] == number_of_files, f"Number of files in processed triplets pkl file is incorrect, expected {number_of_files}, got {data['num_files']}"

def test_triplet_filtering(number_of_files):
    result = subprocess.run(["python", "src/extraction_pipeline/filter_triplets.py"])
    assert result.returncode == 0, f"Filtering the triplets failed, the error is: {result.stderr}"
    # open path_regression_test_folder and check if the data is there
    with open(GENERAL_CONTEXT.path_regression_test_folder.joinpath('filter_triplets_test_output.json'), 'r') as f:
        data = json.load(f)

    assert data['time_taken'] < 1000, f"Triplet filtering took too long, it took {data['time_taken']} seconds"
    assert data['num_files'] == number_of_files, f"Number of files in final triplets pkl file is incorrect, expected {number_of_files}, got {data['num_files']}"


