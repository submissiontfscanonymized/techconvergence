import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from extraction_pipeline.helpers import logger, listener_process, listener_configurer
from pathlib import Path

from extraction_pipeline.contexts import GeneralContext, DataLoadingContext

from abbreviations import schwartz_hearst

import logging


import re
import time
import json
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

def get_text_files_in_dir(path):
    """ Get all the .txt files in a directory
    
    Parameters:
        path (Path): path to the directory
        
        Return:
        texts (list[str]): list of texts
        text_file_names (list[str]): list of names of the text files  
    """
    texts = []
    text_file_names = []

    #iterate over txt files with Path.glob()
    for file in path.rglob('*.txt'):
        text_file_names.append(file.name)
        with open(file, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts, text_file_names

def remove_citations(texts, local_logger):
    """Remove citations through a rule based heuristic

    Parameters:
        texts (list[str]): list of texts
        log_file (str): path to the log file
    Return:
        new_texts (list[str]): list of texts with the citations removed

    """
    new_texts = []
    changes = []
    for i, text in enumerate(texts):
        if i % 1000 == 0:
            local_logger.write_log('Removing citations: ' + str(i) + '/' + str(len(texts)))
        new_text = ''
        line_count = 0
        for line in text.split('\n'):
            # Brackets with only a number inside are removed
            # Brackets with a year inside are removed
            # Brackets with a number inside and other text, e.g. [llm2], are not removed
            re_expression = '\[\d{4}[a-zA-Z0-9 .,!/\-"\']*\]|\[\d+\]|\[[a-zA-Z0-9 .,!/\-"\']*\d{4}\]|\([a-zA-Z0-9 .,!/\-"\']*\d{4}\)|\(\d{4}[a-zA-Z0-9 .,!/\-"\']*\)|\(\d+\)'
            if re.search(re_expression, line):
                # get starting and ending position of citation. If there are multiple citations in one line, store starting and ending position of each in a list
                new_line = re.sub(re_expression, '', line)
                start_pos, end_pos = [], []
                for match in re.finditer(re_expression, line):
                    start_pos.append(match.start())
                    end_pos.append(match.end())
                
                changes.append(local_logger.get_log_changing_sentence(line, new_line, line_count, start_pos, end_pos, 'Removing citations'))
            else:
                new_line = line
            line_count += 1
            new_text += new_line + '\n'
        new_texts.append(new_text)
    return new_texts, changes



def expand_abbreviations(texts, local_logger):
    """Expand the abbreviations using the Schwartz-Hearst algorithm

    Parameters:
        texts (list[str]): list of texts
        log_file (str): path to the log file

    Return:
        new_texts (list[str]): list of texts with the abbreviations expanded
        pairs (dict): dictionary with the abbreviations as keys and the definitions as values
    """
    changes = []
    new_texts = []
    for i, text in enumerate(texts):
        if i % 1000 == 0:
            local_logger.write_log('Expanding abbreviations: ' + str(i) + '/' + str(len(texts)))
        pairs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=text)
        # Add the fully lowercased versions of the abbreviations as keys
        pairs_copy = pairs.copy()
        for abbrev, definition in pairs_copy.items():
            if abbrev.lower() != abbrev:
                pairs[abbrev.lower()] = definition
        # iterate over the lines in the text file and replace the abbreviations
        # split by \n to get the lines
        sentences = text.split('\n')
        new_sentences = []
        for i, sentence in enumerate(sentences):
            old_sentence = sentence
            start_pos, end_pos = [], []
            replacements = []
            for abbrev, definition in pairs.items():
                # check whether the abbreviation is in the sentence
                if abbrev in sentence:
                    # we have to make sure that the abbreviation is not inside a word, e.g. "in" in "within". It is allowed to have punctuation before and after the abbreviation, e.g. AI, or AI.
                    # We add a "try" since the abbreviation might contain a backslash, which would cause an error. If there is an error, we skip the abbreviation
                    try:
                        for m in re.finditer(abbrev, old_sentence):
                            # check whether there is a letter before and after the abbreviation
                            if m.start() > 0 and sentence[m.start()-1].isalpha():
                                    continue
                            if m.end() < len(sentence) and sentence[m.end()].isalpha():
                                    continue
                            replacements.append(((m.start(), m.end()), definition))
                    except:
                        continue
            # Now we want to make sure that the replacements do not overlap. We do this by sorting the replacements by their start index and then iterating over them and only keeping the first replacement that does not overlap with the previous replacements
            replacements = sorted(replacements, key=lambda x: x[0][0])
            replacements_to_keep = []
            for replacement in replacements:
                if len(replacements_to_keep) == 0:
                    replacements_to_keep.append(replacement)
                else:
                    # check whether the replacement overlaps with the previous replacements
                    overlap = False
                    for replacement_to_keep in replacements_to_keep:
                        if replacement[0][0] <= replacement_to_keep[0][1]:
                            overlap = True
                            break
                    if not overlap:
                        replacements_to_keep.append(replacement)
            # Now we can replace the abbreviations with their definitions
            sorted_replacements_to_keep = sorted(replacements_to_keep, key=lambda x: x[0][0], reverse=True)
            for replacement in sorted_replacements_to_keep:
                sentence = sentence[:replacement[0][0]] + replacement[1] + sentence[replacement[0][1]:]
                start_pos.append(replacement[0][0])
                end_pos.append(replacement[0][1])
            new_sentences.append(sentence)
            if (len(replacements_to_keep) > 0):
                changes.append(local_logger.get_log_changing_sentence(old_sentence, sentence, i, start_pos, end_pos, 'Abbreviation replacement'))
        # Get new_text by joining the sentences
        new_text = '\n'.join(new_sentences)
        new_texts.append(new_text)
    return new_texts, changes


def fix_line_breaks(texts, local_logger):
    """Fix line breaks in the texts

    Parameters:
        texts (list[str]): list of texts
        PATH_LOG (str): path to the log file

    Return:
        new_texts (list[str]): list of texts with the line breaks fixed
    """
    changes = []
    new_texts = []
    for idx, text in enumerate(texts):
        if idx % 1000 == 0:
            local_logger.write_log('Fixing line breaks in text: ' + str(idx) + '/' + str(len(texts)))
        new_text = ''
        for idx, line in enumerate(text.split('\n')):
            # We start by fixing structures such as "beau- tiful" and "beau- tifully" to "beautiful" and "beautifully"
            # We also want to fix structures such as "beau-  tiful" to "beautiful", or "beau-   tiful" to "beautiful". 
            regex_expression = r'(\w)-\s+(\w)'
            start_positions = [m.start() for m in re.finditer(regex_expression, line)]
            end_positions = [m.end() for m in re.finditer(regex_expression, line)]
            new_line = re.sub(regex_expression, r'\1\2', line)
            # write log
            if len(start_positions) > 0:
                changes.append(local_logger.get_log_changing_sentence(line, new_line, idx, start_positions, end_positions, 'Fixing line breaks'))
            new_text += new_line + '\n'
        new_texts.append(new_text)
    return new_texts, changes


def process_batch(text_files, text_names, parameters, queue):
    local_logger = logger(queue=queue, logging_level_file=parameters['logging_level_file'], logging_level_cmd=parameters['logging_level_cmd'], verbose=parameters['verbose'])

    # ------------------- FIX LINE BREAKS -------------------
    texts, changes_line_breaks = fix_line_breaks(text_files, local_logger)

    # ------------------- REMOVE CITATIONS -------------------
    texts, changes_citations = remove_citations(texts, local_logger)

    # ------------------- EXPAND ABBREVIATIONS -------------------
    texts, changes_abbreviations = expand_abbreviations(texts, local_logger)

    return (texts, text_names, changes_line_breaks, changes_citations, changes_abbreviations)

def regression_test(time_taken):
    results_regression_test = {}
    path_regression_test_folder = GENERAL_CONTEXT.path_regression_test_folder
    results_regression_test['time_taken'] = time_taken
    results_regression_test['num_files_in_processed_texts'] = len(list(GENERAL_CONTEXT.path_save_processed_texts.rglob('*.txt')))
    
    # save as json
    with open(path_regression_test_folder.joinpath('preprocessing_test_output.json'), 'w') as f:
        json.dump(results_regression_test, f)

def main():
    """Convert pdfs to text and process the text"""
    ###################################   SETTINGS  ###################################################
    start_time = time.time()
    run_regression_test = True if os.getenv("REGRESSION_TEST") == "1" else False
    global GENERAL_CONTEXT, DATA_LOADING_CONTEXT
    GENERAL_CONTEXT = GeneralContext()
    DATA_LOADING_CONTEXT = DataLoadingContext()
    num_cpus = int(os.getenv("NUM_CPUS_PREPROCESSING"))
    path_save_raw_texts = GENERAL_CONTEXT.path_save_raw_texts

    parameters = {'logging_level_file': GENERAL_CONTEXT.logging_level_file,
                    'logging_level_cmd': GENERAL_CONTEXT.logging_level_cmd,
                    'verbose': GENERAL_CONTEXT.verbose}

    general_logger = logger('preprocessing_output', GENERAL_CONTEXT.path_log.joinpath('general_output_preprocessing.txt'), logging_level_file=parameters['logging_level_file'], logging_level_cmd=parameters['logging_level_cmd'], verbose=GENERAL_CONTEXT.verbose)
    general_logger.write_log(f'Starting preprocessing, run name: {GENERAL_CONTEXT.run_name}')

    logger_abbr = logger('preprocessing_abbreviations', GENERAL_CONTEXT.path_log.joinpath('abbreviations_preprocessing.txt'), logging_level_file=parameters['logging_level_file'], logging_level_cmd=parameters['logging_level_cmd'], verbose=GENERAL_CONTEXT.verbose)
    logger_citations = logger('preprocessing_citations', GENERAL_CONTEXT.path_log.joinpath('citations_preprocessing.txt'), logging_level_file=parameters['logging_level_file'], logging_level_cmd=parameters['logging_level_cmd'], verbose=GENERAL_CONTEXT.verbose)
    logger_line_breaks = logger('preprocessing_line_breaks', GENERAL_CONTEXT.path_log.joinpath('line_breaks_preprocessing.txt'), logging_level_file=parameters['logging_level_file'], logging_level_cmd=parameters['logging_level_cmd'], verbose=GENERAL_CONTEXT.verbose)

    texts, text_file_names = get_text_files_in_dir(path_save_raw_texts)

    if run_regression_test:
        general_logger.write_log('This is a regression test run')
    general_logger.write_log('Number of raw text files: ' + str(len(list(path_save_raw_texts.rglob('*.txt')))))

    filtered_text_file_names = []
    filtered_text_files = []    
    num_already_processed = 0               
    for i, file in enumerate(text_file_names):
        if Path.joinpath(GENERAL_CONTEXT.path_save_processed_texts, file).exists():
            num_already_processed += 1
        else:
            filtered_text_file_names.append(file)
            filtered_text_files.append(texts[i])

    general_logger.write_log('Number of texts already processed: ' + str(num_already_processed))
    general_logger.write_log('Number of texts to preprocess: ' + str(len(filtered_text_files)))

    #-------------------- SPLIT FOR MULTIPROCESSING -------------------
    # Set up queue
    with mp.Manager() as manager:
        queue = manager.Queue()
        listener = mp.Process(target=listener_process, args=(queue, listener_configurer, GENERAL_CONTEXT.path_log.joinpath('general_output_preprocessing.txt'), GENERAL_CONTEXT.logging_level_file, GENERAL_CONTEXT.logging_level_cmd))
        listener.start()


        # Split the texts in chunks for multiprocessing
        num_files = len(filtered_text_files)
        num_processes = num_cpus
        chunk_size = num_files // num_processes
        res = []
        pool = mp.Pool(num_processes)

        for i in range(num_processes):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            if i == num_processes - 1:
                end = num_files
            res.append(pool.apply_async(process_batch, args=(filtered_text_files[start:end], filtered_text_file_names[start:end], parameters, queue)))

        pool.close()
        pool.join()

        # Get the results
        texts = []
        text_file_names = []
        for r in res:
            text, text_name, changes_line_breaks, changes_citations, changes_abbreviations = r.get()
            # Write the changes to the log files
            for change in changes_line_breaks:
                logger_line_breaks.write_log(change, level=logging.DEBUG)
            for change in changes_citations:
                logger_citations.write_log(change, level=logging.DEBUG)
            for change in changes_abbreviations:
                logger_abbr.write_log(change, level=logging.DEBUG)
            texts.extend(text)
            text_file_names.extend(text_name)

        queue.put(None)
        listener.join()

    general_logger.write_log('Number of processed texts to save: ' + str(len(texts)))
    # ------------------- SAVE TEXT -------------------
    for i, text in enumerate(texts):
        new_path = Path.joinpath(GENERAL_CONTEXT.path_save_processed_texts, text_file_names[i])
        with open(new_path,
                    'w', encoding='utf-8') as f:
                f.write(text)
    general_logger.write_log('Done!')
    end_time = time.time()
    
    if run_regression_test:
        regression_test(end_time-start_time)



if __name__ == "__main__":
    main()