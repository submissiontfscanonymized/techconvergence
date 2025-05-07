from helpers import logger, get_device, listener_configurer, listener_process, remove_folder

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import fitz

import pandas as pd

import logging
import random
import nltk
import multiprocessing as mp
import time
import os
import json
from extraction_pipeline.contexts import GeneralContext, DataLoadingContext

try: 
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

mp.set_start_method('spawn', force=True)

class LanguageDetector:

    """Detect the language of a text with a LLM"""

    language_model_name = "papluca/xlm-roberta-base-language-detection"

    def __init__(self):
        device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.language_model_name
        )
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )

    def get_language(self, text: str) -> list[dict]:
        """Get the language of a text

        Parameters:
            text (int): text to analyze

        Return:
            language_score list[{'label': str, 'score': int}]: list of dict with the language and the score
        """

        # If the input is too long can return an error
        try:
            return self.classifier(text[:1300])
        except Exception:
            try:
                return self.classifier(text[:514])
            except Exception:
                return [{"label": "err"}]

def keep_newest_versions_one_folder(folder_name, present_files, logger_general_output):
    """Keep only the newest version of each paper, remove the older versions

    Parameters:
        folder_name (Path): path to the directory with the pdf files

    Return:
        None
    """
    files_to_keep = set()
    num_new_files = 0
    version_dict = {}
    all_files = list(folder_name.rglob('*.pdf'))

    files_to_not_consider = set()
    num_files = len(all_files)

    # iterate over the files and keep the newest version
    for file in all_files:
        filename = file.name
        filename_reduced = filename.split('v')[0] # 38718v1.pdf -> 38718

        if filename_reduced in present_files:
            continue

        num_new_files += 1
        # try to get the version, if it cannot be converted to an integer, then we should just skip this file
        try:
            version = int(filename.split('v')[-1].split('.')[0])
        except: 
            files_to_not_consider.add(filename)
            continue

        # if the filename is in the dictionary, check whether the version is higher
        if filename_reduced in version_dict:
            if version > version_dict[filename_reduced]:
                version_dict[filename_reduced] = version
        else:
            version_dict[filename_reduced] = version

    for filename_reduced, version in version_dict.items():
        files_to_keep.add(filename_reduced + 'v' + str(version) + '.pdf')

    num_files_to_keep = len(files_to_keep)

    if num_new_files == 0:
        logger_general_output.write_log('No new files in folder: {}'.format(folder_name))
        return None

    else:
        logger_general_output.write_log('Percentage of files removed due to being older versions: {:.2f}%'.format(100*(num_files - num_files_to_keep)/num_files))
        # remove all files that are not in files_to_keep
        for file in folder_name.rglob('*.pdf'):
            filename = file.name
            if filename not in files_to_keep:
                files_to_not_consider.add(filename)

        return files_to_not_consider

def keep_newest_versions(present_files, general_logger):
    """Keep only the newest version of each paper, remove the older versions

    Parameters:
        folder_name (Path): path to the directory with the pdf files

    Return:
        None
    """
    files_to_not_consider = set()
    if isinstance(GENERAL_CONTEXT.path_raw_pdf, list):
        for path in GENERAL_CONTEXT.path_raw_pdf:
            not_consider = keep_newest_versions_one_folder(path, present_files, general_logger)
            if not_consider is not None:
                files_to_not_consider.update(not_consider)
    else:
        not_consider = keep_newest_versions_one_folder(GENERAL_CONTEXT.path_raw_pdf, present_files, general_logger)
    if not_consider is not None:
        files_to_not_consider.update(not_consider)
    return files_to_not_consider

def keep_categories(general_logger):
    """Keep only the files with the allowed categories

    Parameters:
        PATH_RAW_FILES (str): path to the directory with the raw files
        PATH_METADATA (str): path to the directory with the metadata
        categories (list[str]): list of allowed categories
        remove_files (bool): whether to remove the files that are not in the allowed categories
        inverse (bool): if True, we remove the files that are in the parameter categories

    Return:
        kept_files (list[str]): list of the kept files
    """
    # read metadata, it is a json file, it ends in .json
    general_logger.write_log('Loading metadata...')
    df = pd.read_pickle(GENERAL_CONTEXT.path_metadata_pickle)
    general_logger.write_log('Now starting to filter...')
    if DATA_LOADING_CONTEXT.inverse:
        df_filtered = df[df['categories'].apply(lambda x: all([c not in x.split() for c in DATA_LOADING_CONTEXT.categories]))]
    else:
        df_filtered = df[df['categories'].apply(lambda x: any([c in x.split() for c in DATA_LOADING_CONTEXT.categories]))]
    general_logger.write_log('Percentage of articles from metadata with required categories: {:.2f}%'.format(100*len(df_filtered)/len(df)))
    kept_files = df_filtered["id"].to_list()
    return kept_files

def get_all_pdf_from_batch(files, parameters, present_files, kept_files, files_to_not_consider, queue):
    """Get all the pdf files from a directory

    Parameters:
        path (str): path to the directory

    Return:
        pdf_list (list[Path]): list of Path to the pdf files
    """
    process_logger = logger(queue=queue, logging_level_file=parameters['logging_level_file'], logging_level_cmd=parameters['logging_level_cmd'], verbose=parameters['verbose'])
    # also search subdirectories
    pdf_list = []
    total_num_files = len(files)
    # get all the pdf files
    for idx, file in enumerate(files):
        if idx % 500 == 0:
            process_logger.write_log('Checking file {} out of {} for the right version and category'.format(idx, total_num_files))
        if file.name.split('v')[0] in present_files:
            continue
        if kept_files is None:
            pdf_list.append(file)
        else:
            if file.name[:-6] in kept_files:
                pdf_list.append(file)

    # remove the files that are not the latest versions
    if files_to_not_consider is not None:
        pdf_list = [file for file in pdf_list if file.name not in files_to_not_consider]
    
    return pdf_list


def get_all_pdf_from_dir(present_files, kept_files, files_to_not_consider, general_logger, path_log):
    """Get all the pdf files from a directory

    Parameters:
        path (str): path to the directory

    Return:
        pdf_list (list[Path]): list of Path to the pdf files
    """
    num_processes = num_cpus
    path_raw_pdf = GENERAL_CONTEXT.path_raw_pdf
    # also search subdirectories
    pdf_list = []

    # if it is a list, then we iterate over the list to get all the pdfs
    if isinstance(path_raw_pdf, list):  
        all_pdfs = []
        for path in path_raw_pdf:
            all_pdfs += list(path.rglob('*.pdf'))
    else:
        all_pdfs = list(path_raw_pdf.rglob('*.pdf'))

    num_files = len(all_pdfs)
    num_files_per_process = num_files // num_processes
    batches = []
    for i in range(num_processes):
        if i == num_processes - 1:
            batches.append(all_pdfs[i*num_files_per_process:])
        else:
            batches.append(all_pdfs[i*num_files_per_process:(i+1)*num_files_per_process])

    parameters = {
        'logging_level_file': GENERAL_CONTEXT.logging_level_file,
        'logging_level_cmd': GENERAL_CONTEXT.logging_level_cmd,
        'verbose': GENERAL_CONTEXT.verbose,
    }

    with mp.Manager() as manager:
        queue = manager.Queue()
        listener = mp.Process(target=listener_process, args=(queue, listener_configurer, path_log, parameters['logging_level_file'], parameters['logging_level_cmd']))
        listener.start()

        pool = mp.Pool(num_processes)
        res = []
        for i in range(num_processes):
            res.append(pool.apply_async(get_all_pdf_from_batch, args=(batches[i], parameters, present_files, kept_files,  files_to_not_consider, queue)))
        pool.close()
        pool.join()
        for r in res:
            pdf_list_folder = r.get()
            pdf_list += pdf_list_folder
        
        queue.put(None)
        listener.join()

    subset = DATA_LOADING_CONTEXT.subset
    if subset is not None:
        if subset < 1:
            pdf_list = random.sample(pdf_list, int(subset * len(pdf_list)))
        else:
            # if the size of the list is smaller than the subset, then just return the list. Also log the name of the folder
            if len(pdf_list) < subset:
                general_logger.write_log('The folder has less pdfs than the subset of size {}')
            else:
                pdf_list = random.sample(pdf_list, subset)
          
    return pdf_list

def get_abstract_from_pdf(pdf_file, average_abstract_length, local_logger, path):
    # Now we get the text from the first 2 pages, make sure the code does not crash if there are less than 2 pages
    page_counter = 0
    txt_first_two = ''
    try: 
        while page_counter < len(pdf_file) and page_counter < 2:
            page = pdf_file[page_counter]
            txt_first_two += page.get_text(flags=0)
            page_counter += 1
    except Exception as e:
        local_logger.write_log(f'Error reading the first two pages of the pdf {path}: {e}', level=logging.ERROR)
        return ''
        
    txt_first_two_lower = txt_first_two.lower()
    abstract_pos = txt_first_two_lower.find("abstract")
    introduction_pos = txt_first_two_lower.find("introduction")
    if introduction_pos != -1 and abstract_pos != -1:
        # "normal scenario", if a position is -1 it would mean the word is not found
        if abstract_pos < introduction_pos: # if the abstract is before the introduction, we take the abstract until the introduction starts
            starting_pos = abstract_pos
            abstract_length = max(introduction_pos - abstract_pos, average_abstract_length)
        else:
            # odd scenario, likely the word abstract does not identify the start of the abstract
            starting_pos = introduction_pos
            abstract_length = average_abstract_length

    elif introduction_pos == -1 and abstract_pos == -1:
        # Skip the preface of ~100 characters
        starting_pos = 100
        abstract_length = average_abstract_length
    elif introduction_pos == -1:
        starting_pos = abstract_pos
        abstract_length = average_abstract_length
    else:
        starting_pos = introduction_pos
        abstract_length = average_abstract_length

    abstract = txt_first_two[starting_pos:starting_pos + abstract_length]
    return abstract

def check_if_term_in_abstract(abstract, terms_to_filter_for):
    for term in terms_to_filter_for:
        term_lower = term.lower()
        abstract_lower = abstract.lower()
        if term_lower in abstract_lower:
            return True
    return False

    
def get_text_from_pdf(path, parameters, local_logger) -> str:
    """Extract text from a pdf file

    Parameters:
        path (str|Path): path to the pdf file

    Return:
        text (str): text extracted from the pdf file
    """
    try:
        pdf_file = fitz.open(path)
        assert len(pdf_file) > 0 
        if parameters['filter_for_terms_in_abstract']:
            abstract = get_abstract_from_pdf(pdf_file, parameters['average_abstract_length'], local_logger, path)
            if not check_if_term_in_abstract(abstract, parameters['terms_to_filter_for']):
                pdf_file.close()
                return ''   

        page_counter = 0
        txt = ''
        while page_counter < len(pdf_file) and page_counter < parameters['max_num_pages']:
            page = pdf_file[page_counter]
            txt += page.get_text(flags=0)
            page_counter += 1
        pdf_file.close()
    except Exception as e:
        if local_logger is not None:
            local_logger.error(f'Error opening {path}: {e}')
        return ''
    return txt

def process_pdf_subprocess(pdf_list_sub, queue, parameters):
    #language_detector = LanguageDetector()
    process_logger = logger(queue=queue, logging_level_file=parameters['logging_level_file'], logging_level_cmd=parameters['logging_level_cmd'], verbose=parameters['verbose'])
    texts = []
    num_filtered_away = 0
    num_pdfs = len(pdf_list_sub)
    for i, pdf in enumerate(pdf_list_sub):
        pdf_name = pdf.name[:-4]
        if i % 500 == 0:
            process_logger.write_log(text='Converted {}/{} pdfs to text'.format(i, num_pdfs))
        # check if it is already in the save folder
        if parameters['path_save_raw_texts'].joinpath(pdf_name + '.txt').exists():
            continue
        txt = get_text_from_pdf(pdf, parameters, process_logger)
        if txt == '':
            num_filtered_away += 1
            continue
        # process only english papers or papers with no language detected 
        #if language_detector.get_language(txt)[0]["label"] not in ["en", "err"]:
        #    continue
        txt = remove_header_references(txt)
        # split text into sentences
        txt = txt.replace('\n', ' ')
        sentences = nltk.sent_tokenize(txt)
        
        # if a sentence ends with 'et al.', then merge it with the next sentence (this is a common failure mode of the sentence tokenizer)
        for i in range(len(sentences) - 1):
            if i + 1 < len(sentences):
                num_replacements = 0
                while sentences[i].endswith("et al.") and (i + 1 + num_replacements) < len(sentences):
                    sentences[i] = sentences[i] + " " + sentences[i + 1 + num_replacements]
                    num_replacements += 1
                # remove the merged sentences
                sentences = sentences[:i + 1] + sentences[i + 1 + num_replacements:]

        # Now simply write it to where it needs to be saved
        with open(parameters['path_save_raw_texts'].joinpath(pdf_name + '.txt'), 'w', encoding='utf-8') as f:
            f.write("\n".join(sentences))
            
        texts.append(pdf_name)
    process_logger.close()
    return texts, num_filtered_away
        

def process_pdfs(pdf_list, path_log, general_logger):
    """Process the pdfs

    Parameters:
        save_folder (str): folder to save the processed texts
        pdf_list (list[Path]): list of Path to the pdf files
        
    Return:
        texts (list[str]): list of texts
    """
    num_processes = num_cpus
    num_folders = len(pdf_list) // num_processes
    num_filtered_away = 0
    num_texts_saved = 0
    with mp.Manager() as manager:
        res = []
        queue = manager.Queue()
        listener = mp.Process(target=listener_process, args=(queue, listener_configurer, path_log, GENERAL_CONTEXT.logging_level_file, GENERAL_CONTEXT.logging_level_cmd))
        listener.start()

        pool = mp.Pool(num_processes)
        parameters = {
            'path_save_raw_texts': GENERAL_CONTEXT.path_save_raw_texts,
            'filter_for_terms_in_abstract': DATA_LOADING_CONTEXT.filter_for_terms_in_abstract,
            'terms_to_filter_for': DATA_LOADING_CONTEXT.terms_to_filter_for,
            'average_abstract_length': DATA_LOADING_CONTEXT.average_abstract_length,
            'max_num_pages': DATA_LOADING_CONTEXT.max_num_pages,
            'logging_level_file': GENERAL_CONTEXT.logging_level_file,
            'logging_level_cmd': GENERAL_CONTEXT.logging_level_cmd,
            'verbose': GENERAL_CONTEXT.verbose,
        }
        for i in range(num_processes):
            if i == num_processes - 1:
                pdf_list_sub = pdf_list[i*num_folders:]
            else:
                pdf_list_sub = pdf_list[i*num_folders:(i+1)*num_folders]
            res.append(pool.apply_async(process_pdf_subprocess, args=(pdf_list_sub, queue, parameters)))
        pool.close()
        pool.join()
        texts = []
        for r in res:
            texts += r.get()[0]
            num_filtered_away += r.get()[1]

        queue.put(None)
        listener.join()

        for pdf_name in texts:
            # assert that the file exists
            num_texts_saved += 1
            assert GENERAL_CONTEXT.path_save_raw_texts.joinpath(pdf_name + '.txt').exists()
    
    general_logger.write_log('Done processing pdfs to text, percentage of pdfs that is filtered away due to the selection by term or that has a corrupt page: {:.2f}%'.format(100*num_filtered_away/len(pdf_list)))
    general_logger.write_log('Number of texts saved: {}'.format(num_texts_saved))

    return texts, num_filtered_away

def remove_header_references(text: str) -> str:
    """Remove the header and the references from a text

    Parameters:
        text (str): text to clean

    Return:
        text (str): cleaned text
    """
    txt_lower = text.lower()
    abstract_pos = txt_lower.find("abstract")
    introduction_pos = txt_lower.find("introduction")

    if introduction_pos != -1 and abstract_pos != -1:
        abstract_pos = min(abstract_pos, introduction_pos)
    else:
        abstract_pos = max(abstract_pos, introduction_pos)

    if abstract_pos == -1:
        # If not foud remove fixed number of characters to remove part of the header
        abstract_pos = 100

    references_pos = txt_lower.rfind("reference")
    acknowledgements_pos = txt_lower.rfind("acknowledgement")
    if (
        acknowledgements_pos != -1
        and acknowledgements_pos < references_pos
        and acknowledgements_pos > len(text) / 2
    ):
        references_pos = acknowledgements_pos
    if references_pos == -1:
        references_pos = len(text)

    return text[abstract_pos:references_pos]

def find_processed_files(path_save_raw_texts):
    present_files = set()
    for file in path_save_raw_texts.rglob('*.txt'):
        name = file.name.split('v')[0]
        present_files.add(name)
    return present_files

def regression_test(time_taken):
    results_regression_test = {}
    path_regression_test_folder = GENERAL_CONTEXT.path_regression_test_folder
    results_regression_test['time_taken'] = time_taken
    results_regression_test['num_files_in_raw_texts'] = len(list(GENERAL_CONTEXT.path_save_raw_texts.rglob('*.txt')))
    
    # save as json
    with open(path_regression_test_folder.joinpath('load_data_test_output.json'), 'w') as f:
        json.dump(results_regression_test, f)


def main():
    """Convert pdfs to text and process the text"""
    run_regression_test = True if os.getenv("REGRESSION_TEST") == "1" else False
    start_time = time.time()
    global GENERAL_CONTEXT, DATA_LOADING_CONTEXT, num_cpus
    GENERAL_CONTEXT = GeneralContext()
    DATA_LOADING_CONTEXT = DataLoadingContext()
    num_cpus = int(os.getenv("NUM_CPUS_DATA_LOADING"))

    if GENERAL_CONTEXT.clear_before_run:
        remove_folder(GENERAL_CONTEXT.path_save_text_folder)
        remove_folder(GENERAL_CONTEXT.path_results_folder)
        if run_regression_test:
            remove_folder(GENERAL_CONTEXT.path_regression_test_folder)
        GENERAL_CONTEXT.path_regression_test_folder.mkdir(parents=True, exist_ok=True)
        GENERAL_CONTEXT.path_log.mkdir(parents=True, exist_ok=True)
        GENERAL_CONTEXT.path_save_raw_texts.mkdir(parents=True, exist_ok=True)
        GENERAL_CONTEXT.path_save_processed_texts.mkdir(parents=True, exist_ok=True)
        GENERAL_CONTEXT.path_save_triplets_folder.mkdir(parents=True, exist_ok=True)
        GENERAL_CONTEXT.path_save_processed_triplets_folder.mkdir(parents=True, exist_ok=True)

    general_logger = logger('load_data', GENERAL_CONTEXT.path_log.joinpath('load_data.txt'), logging_level_file=GENERAL_CONTEXT.logging_level_file, logging_level_cmd=GENERAL_CONTEXT.logging_level_cmd, verbose=GENERAL_CONTEXT.verbose)

    general_logger.write_log(f'Starting the load data script, run name: {GENERAL_CONTEXT.run_name}')

    if run_regression_test:
        general_logger.write_log('This is a regression test run')

    general_logger.write_log('Number of CPUs according to multiprocessing (mp.cpu_count): ' + str(mp.cpu_count()))
    general_logger.write_log('Number of CPUs according to the context (user input): ' + str(num_cpus))

    general_logger.write_log('Path to the raw pdfs: {}'.format(str(GENERAL_CONTEXT.path_raw_pdf)))
    general_logger.write_log('Subset: {}'.format(DATA_LOADING_CONTEXT.subset))

    if isinstance(GENERAL_CONTEXT.path_raw_pdf, list):
        total_num_files = 0
        for path in GENERAL_CONTEXT.path_raw_pdf:
            total_num_files += len(list(path.rglob('*.pdf')))
        general_logger.write_log('Total number of raw pdfs: {}'.format(total_num_files))
    else:
        general_logger.write_log('Total number of raw pdfs: {}'.format(len(list(GENERAL_CONTEXT.path_raw_pdf.rglob('*.pdf')))))

    
    present_files = find_processed_files(GENERAL_CONTEXT.path_save_raw_texts)
    general_logger.write_log('Total number of raw pdfs that are already processed: {}'.format(len(present_files)))

    # ------------------- KEEP NEWEST PAPER VERSION -------------------
    if DATA_LOADING_CONTEXT.filter_by_version:
        general_logger.write_log('Keeping newest paper version...')
        files_to_not_consider = keep_newest_versions(present_files, general_logger)
        if files_to_not_consider == None:
            general_logger.write_log('There are no new files, we can continue to the preprocessing.')
            return

    else:
        files_to_not_consider = None

    # ------------------- KEEP ALLOWED CATEGORIES AND GET PDFS-------------------
    if DATA_LOADING_CONTEXT.filter_categories:
        general_logger.write_log('Keeping only the allowed categories...')
        general_logger.write_log('Removing files not in the allowed categories...')
        kept_categories = keep_categories(general_logger)

    else:
        kept_categories = None

    general_logger.write_log('Getting pdfs...')
    pdf_list = get_all_pdf_from_dir(present_files, kept_categories, files_to_not_consider, general_logger, GENERAL_CONTEXT.path_log.joinpath('load_data.txt'))
    general_logger.write_log('Number of pdfs after filtering for categories and versions: {}'.format(len(pdf_list)))

    # ------------------- CONVERT PDF TO TEXT -------------------
    general_logger.write_log('Converting pdfs to text...')

    if len(pdf_list) == 0:
        general_logger.write_log('No pdfs to process, exiting...')
        return
    # process the pdfs in parallel
    process_pdfs(pdf_list, GENERAL_CONTEXT.path_log.joinpath('load_data.txt'), general_logger)

    end_time = time.time()

    if run_regression_test:
        regression_test(end_time - start_time)
        # Remove the results folder, make sure the logger is closed such that the folder can be removed
 
if __name__ == "__main__":
    main()