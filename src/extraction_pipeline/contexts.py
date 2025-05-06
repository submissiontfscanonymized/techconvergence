""" Here all the context classes for the different steps of the pipeline are made """
from pathlib import Path
import pandas as pd
from helpers import logger, remove_folder, get_paths_armasuisse_cluster
import json
import pickle
import os
import sys
import yaml

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

# CONFIGS path is an environment variable called CONFIG_FILE
# print the environment variable
# load the config file, it is yaml
with open(os.getenv("CONFIG_PATH"), 'r') as stream:
    try:
        configs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

""" General settings for the pipeline """
class GeneralContext:
    def __init__(self):
        self.seed = configs["seed"]
        self.verbose = configs["verbose"]

        self.run_name = configs["run_name"]

        # Logging
        self.logging_level_cmd = configs["logging_level_cmd"]
        self.logging_level_file = configs["logging_level_file"]

        self.use_lemmatization = configs["use_lemmatization"]
        self.spacy_model = configs["spacy_model"]

        # paths
        self.root_path = Path.cwd()
        
        self.path_metadata = self.root_path.joinpath(configs["path_metadata"])
        if isinstance(configs["path_raw_pdf"], list):
            # check if the elements are also lists
            if isinstance(configs["path_raw_pdf"][0], list):
                self.path_raw_pdf = get_paths_armasuisse_cluster(configs["path_raw_pdf"][0], configs["path_raw_pdf"][1])
                if isinstance(self.path_raw_pdf, list):
                    self.path_raw_pdf = [self.root_path.joinpath(path) for path in self.path_raw_pdf]
            else:
                self.path_raw_pdf = [self.root_path.joinpath(path) for path in configs["path_raw_pdf"]]
        else:
            self.path_raw_pdf = self.root_path.joinpath(configs["path_raw_pdf"])
        self.path_data_folder = self.root_path.joinpath(configs["path_data_folder"])
        self.path_save_text_folder = self.path_data_folder.joinpath(self.run_name) # path to the directory to save the raw text
        self.path_save_raw_texts = self.path_save_text_folder.joinpath('raw_papers_' + self.run_name) # path to the directory to save the raw text
        self.path_save_processed_texts = self.path_save_text_folder.joinpath('processed_papers_' + self.run_name) # path to the directory to save the processed text
        self.path_results_folder = self.root_path.joinpath('results_' + self.run_name)
        if os.getenv("REGRESSION_TEST") == "1":
            self.path_regression_test_folder = self.root_path.joinpath('regression_test_results')
            self.path_regression_test_folder.mkdir(parents=True, exist_ok=True)
        self.path_log = self.path_results_folder.joinpath('logs_' + self.run_name) # path to the directory to save the logs

        self.path_save_triplets_folder = self.path_results_folder.joinpath('triplets') # path to the directory to save the triplets
        self.path_save_triplets_file = self.path_save_triplets_folder.joinpath('triplets.csv')

        self.path_save_processed_triplets_folder = self.path_results_folder.joinpath('processed_triplets') # path to the directory to save the processed triplets
        self.path_save_processed_triplets_file = self.path_save_processed_triplets_folder.joinpath('processed_triplets.pkl')

        self.path_save_triplet_filtering_folder = self.path_results_folder.joinpath('triplet_filtering') # path to where to save the triplet filtering results

        self.path_save_final_results_folder = self.path_results_folder.joinpath('final_results') # path to where to save the final results
        self.path_save_final_triplets_file = self.path_save_final_results_folder.joinpath('final_triplets.pkl') # path to where to save the final triplets

        self.path_save_final_results_folder.mkdir(parents=True, exist_ok=True)
        self.path_neo4j = self.path_results_folder.joinpath('neo4j') # path to where to save the Neo4J results

        # if path_metadata ends in .json, load it and save it also as a pickle file
        self.path_metadata_pickle = self.path_metadata.with_suffix('.pickle')
        self.save_metadata_as_pickle()

        self.clear_before_run = configs["clear_before_run"]

        self.path_log.mkdir(parents=True, exist_ok=True)
        self.path_save_raw_texts.mkdir(parents=True, exist_ok=True)
        self.path_save_processed_texts.mkdir(parents=True, exist_ok=True)
        self.path_save_triplets_folder.mkdir(parents=True, exist_ok=True)
        self.path_save_processed_triplets_folder.mkdir(parents=True, exist_ok=True)
        self.path_save_triplet_filtering_folder.mkdir(parents=True, exist_ok=True)
        self.path_save_final_results_folder.mkdir(parents=True, exist_ok=True)
        self.path_neo4j.mkdir(parents=True, exist_ok=True)

    def save_metadata_as_pickle(self):
        if not self.path_metadata_pickle.exists():
            metadata = pd.read_json(self.path_metadata, lines=True)
            with open(self.path_metadata_pickle, 'wb') as f:
                pickle.dump(metadata, f)

""" Settings for the data loading """
class DataLoadingContext:
    def __init__(self):

        self.inverse = configs["inverse"]
        self.subset = configs["subset"]
        self.filter_categories = configs["filter_categories"]
        self.filter_by_version = configs["filter_by_version"]
        self.categories = configs["categories"]

        self.filter_for_terms_in_abstract = configs["filter_for_terms_in_abstract"]
        self.terms_to_filter_for = configs["terms_to_filter_for"]
        self.average_abstract_length = configs["average_abstract_length"]
        self.max_num_pages = configs["max_num_pages"]

""" Settings for the triplet extraction """
class TripletExtractionContext:
    def __init__(self, general_context):
        self.model_type = configs["model_type"]
        self.max_input_length = configs["max_input_length"]
        self.max_new_tokens = configs["max_new_tokens"]
        self.use_fixed_input_length = configs["fixed_input_length"]
        self.num_lines_per_model_call = configs["num_lines_per_model_call"]
        self.fixed_input_length = configs["fixed_input_length"]
        self.use_fewshot_prompting = configs["use_fewshot_prompting"]
        self.quantize = configs["quantize"]
        self.path_fewshot_examples = configs["path_fewshot_examples"]
        self.device_map = configs["device_map"]

        self.use_sample_generation_strategy = configs["use_sample_generation_strategy"]
        self.repetition_penalty = configs["repetition_penalty"]
        self.temperature = configs["temperature"]
        self.top_p = configs["top_p"]

        self.path_memory_estimate = configs["path_memory_estimate"]

        if self.use_sample_generation_strategy:
            assert self.repetition_penalty is not None
            assert self.temperature is not None
            assert self.top_p is not None

""" Settings for triplet processing """
class TripletProcessingContext:
    def __init__(self, general_context):
        self.max_length = configs["max_length"]
        self.min_num_characters = configs["min_num_characters"]
        self.filler_verbs = configs["filler_verbs"]
        

""" Settings for triplet filtering """
class TripletFilteringContext:
    def __init__(self, general_context):
        self.refilter_triplets = configs["refilter_triplets"]
        
        #################### Frequency filtering #####################
        self.min_num_appearances_for_frequency = configs["min_num_appearances_for_frequency"]
        self.path_word_frequency = general_context.path_save_triplet_filtering_folder.joinpath('word_frequency.pickle')

        #################### Book corpus filtering #####################
        self.path_bookcorpus_gutenberg_folder = general_context.path_data_folder.joinpath('bookcorpus_gutenberg')
        self.path_book_corpus_gutenberg = self.path_bookcorpus_gutenberg_folder.joinpath('book_corpus_gutenberg.pkl')
        self.subset_book_corpus = 0.1
        self.max_length_book = configs["max_length_book"]
        self.min_num_counts_term_in_single_book = configs["min_num_counts_term_in_single_book"]
        self.path_term_frequency_bookcorpus = self.path_bookcorpus_gutenberg_folder.joinpath('term_frequency_bookcorpus_gutenberg.pickle')
        self.path_save_term_scores_bookcorpus = general_context.path_save_triplet_filtering_folder.joinpath('term_scores_bookcorpus.pickle')
        self.cutoff_bookcorpus = configs["cutoff_bookcorpus"]
        self.min_num_paper_appearances_term_bookcorpus_scores = configs["min_num_paper_appearances_term_bookcorpus_scores"]

        #################### Entropy filtering #####################
        self.use_entropy = configs["use_entropy"]
        self.entropy_percentile_threshold = configs["entropy_percentile_threshold"]

        self.path_removed_triplets = general_context.path_save_triplet_filtering_folder.joinpath('removed_triplets.txt')

        general_context.path_save_triplet_filtering_folder.mkdir(parents=True, exist_ok=True)
        self.path_bookcorpus_gutenberg_folder.mkdir(parents=True, exist_ok=True)
        

""" Settings for triplet analysis """
class TripletAnalysisContext:
    def __init__(self, general_context):
        self.path_triplet_analysis_results = general_context.path_results_folder.joinpath('triplet_analysis')
        self.n_max = configs["n_max"]
        self.threshold_num_papers_ngrams = configs["threshold_num_papers_ngrams"]
        self.distance_matrix = configs["distance_matrix"]

        self.path_triplet_analysis_results.mkdir(parents=True, exist_ok=True)

        

class EntropyContext:
    def __init__(self, general_context):
        self.min_num_counts_term_in_single_paper = configs["min_num_counts_term_in_single_paper"]
        self.min_num_paper_appearances_term = configs["min_num_paper_appearances_term"]
        self.path_texts_entropy_calculation = configs["path_texts_entropy_calculation"]

        self.path_entropy_results = general_context.path_data_folder.joinpath('entropy')

        self.path_num_files = self.path_entropy_results.joinpath('num_files.pickle')
        self.path_paper_id_to_category_dict = self.path_entropy_results.joinpath('paper_id_to_category_dict.json')
        self.path_save_entropy_file = self.path_entropy_results.joinpath('entropy.pickle')
        self.path_save_category_counts = self.path_entropy_results.joinpath('category_counts.pickle')
        self.path_num_files_per_category = self.path_entropy_results.joinpath('num_files_per_cat.pickle')

        self.path_entropy_results.mkdir(parents=True, exist_ok=True)

class ParsePatentsContexts:
    def __init__(self, general_context):
        self.run_name = configs["run_name_patent_parsing"]
        self.root_folder = Path.cwd()

        # paths
        self.path_xml_patents = Path(configs["path_patent_xml_files"])
        self.path_results = self.root_folder.joinpath(self.run_name)
        self.path_save_patent_attributes = self.path_results.joinpath('patent_attributes')
        self.save_json_patents = self.path_results.joinpath('parsed_patents')
        self.save_raw_patent_text = general_context.path_data_folder.joinpath('raw_text_' + self.run_name)

        self.path_log = self.path_results.joinpath('logs')
        self.keywords_for_patent_filtering = configs["keywords_for_patent_filtering"]
        self.keys_attribute_dict = configs["keys_attribute_dict"]

        #self.num_cpus = general_context.num_cpus
        self.logging_level_cmd = general_context.logging_level_cmd
        self.logging_level_file = general_context.logging_level_file

        self.save_raw_patent_text.mkdir(parents=True, exist_ok=True)
        self.path_save_patent_attributes.mkdir(parents=True, exist_ok=True)
        self.save_json_patents.mkdir(parents=True, exist_ok=True)
        self.path_log.mkdir(parents=True, exist_ok=True)