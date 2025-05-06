import re

import nltk
import pickle
import string
import sys
import logging
import time
import json
import pandas as pd
import os
import multiprocessing as mp
from nltk.corpus import stopwords
from collections import defaultdict

from helpers import logger, listener_configurer, listener_process, load_spacy, check_spacy_installed
from contexts import GeneralContext, TripletProcessingContext
mp.set_start_method('spawn', force=True)
nltk.download('stopwords')

LLM_FORMAT_REGEX = "\(.+?;.+?;.+?\)"
LLM_FORMAT_REGEX_EXTRACTOR= "\((.+);(.+);(.+)\)"

def parse_triplet(raw_triplets_string):
    parsed_triplets = []
    for triplets in raw_triplets_string:
        lines = triplets.splitlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            # remove the first and last character
            matches = re.findall(LLM_FORMAT_REGEX, line)
            for match in matches:
                result = re.search(LLM_FORMAT_REGEX_EXTRACTOR, match)
                parsed_triplets.append(result.groups())
    return parsed_triplets



def process_csv_file(path):
    """ Process the csv file containing the triplets

    Parameters:
        path (str): The path to the csv file
        verbose (bool): Whether to print the number of incorrect triplets

    Returns:
        dict: A dictionary containing the triplets, key is the paper id and value is the triplets (list of tuples)
    """

    df = pd.read_csv(path, sep=',')

    triplets = defaultdict(list)
    texts = defaultdict(list)

    for _, row in df.iterrows():
        paper_id = row['paper_id']
        triplet = row['triplets']
        text = row['text']
        if isinstance(triplet, str):
            triplets[paper_id].append(triplet)
            texts[paper_id].append(text)

    triplets_parsed = {}
  
    for paper_id, raw_triplets_string in triplets.items():
        triplets_parsed[paper_id] = {}
        triplets_parsed[paper_id]['triplets'] = parse_triplet(raw_triplets_string)
        
    return triplets_parsed

def lower_case(triplets):
    """ Lower case the triplets

    Parameters:
        triplets (dict): A dictionary containing the triplets and the text

    Returns:
        pd.DataFrame: The dataframe containing the lower cased triplets
    """
    # iterate over the dictionary
    for key, values in triplets.items():
        # lower case the triplets
        values['triplets'] = [[subject.lower(), verb.lower(), obj.lower()] for subject, verb, obj in values['triplets']]
    return triplets

def filter_length(triplets, logger, max_length):
    """ Filter the triplets based on length

    Parameters:
        triplets (dict): A dictionary containing the triplets and the text
        logger (logging.Logger): The logger
        max_length (int): The cutoff length

    Returns:
        pd.DataFrame: The dataframe containing the filtered triplets
    """

    # iterate over the dictionary
    num_removed = 0
    for key, values in triplets.items():
        new_triplets = []
        old_triplets = values['triplets']
        for triplet in old_triplets:
            subject, verb, obj = triplet
            if len(subject.split()) > max_length or len(verb.split()) > max_length or len(obj.split()) > max_length:
                num_removed += 1
            else:
                new_triplets.append(triplet)
        values['triplets'] = new_triplets

    return triplets, num_removed

def keep_only_text(triplets):
    """ Keep only the text in the triplets

    Parameters:
        triplets (dict): A dictionary containing the triplets and the text
        logger (logging.Logger): The logger

    Returns:
        pd.DataFrame: The dataframe containing the triplets with only text
    """
    # we want to keep letters, numbers and hyphens, but remove any other character
    to_keep = string.ascii_letters + string.digits + "-" + " "
    for key, values in triplets.items():
        new_triplet_list = []
        for triplet in values['triplets']:
            subject, verb, obj = triplet
            # keep only the characters that are in to_keep
            subject_new = ''.join([c for c in subject if c in to_keep])
            object_new = ''.join([c for c in obj if c in to_keep])
            verb_new = ''.join([c for c in verb if c in to_keep])
            # if the subject, verb or object changed, log it
            new_triplet_list.append((subject_new, verb_new, object_new))
        # append the new triplet list
        values['triplets'] = new_triplet_list
    # update the dataframe
    return triplets

def remove_stopwords(triplets, redundant_verbs):
    """ Remove the stopwords from the triplets

    Parameters:
        triplets (dict): A dictionary containing the triplets and the text
        logger (logging.Logger): The logger
        redundant_verbs (list[str]): A list of redundant verbs

    Returns:
        pd.DataFrame: The dataframe containing the triplets with the stopwords removed
    """

    for key, values in triplets.items():
        new_triplets = []
        for triplet in values['triplets']:
            subject, verb, obj = triplet
            # if verb has multiple words, remove the redundant ones
            verbs = verb.split()
            if len(verbs) > 1:
                verbs = [v for v in verbs if v not in redundant_verbs]
                # remove words that are 1 character long
                verbs = [v for v in verbs if len(v) > 1]
                verb = ' '.join(verbs)
            # Remove stopwords from subject, verb and object, if any of them is empty, do not append the triplet
            new_subject = ' '.join([word for word in subject.split() if word not in stopwords.words('english')])
            new_object = ' '.join([word for word in obj.split() if word not in stopwords.words('english')])
            if len(new_subject) > 0 and len(new_object) > 0:
                new_triplets.append((new_subject, verb, new_object))
        values['triplets'] = new_triplets
    return triplets

def lemmatize(triplets, spacy_model, local_logger):
    """ Lemmatize the triplets using spacy

    Parameters:
        triplets (dict): A dictionary containing the triplets and the text
        logger (logging.Logger): The logger

    Returns:
        pd.DataFrame: The dataframe containing the lemmatized triplets
    """
    nlp = load_spacy(spacy_model)

    for _, values in triplets.items():
        triplet_list = values['triplets']
        new_triplets = []
        for triplet in triplet_list:
            subject, verb, obj = triplet
            subject_doc = nlp(subject)
            verb_doc = nlp(verb)
            obj_doc = nlp(obj)
            new_subject = ' '.join([token.lemma_ for token in subject_doc])
            new_verb = ' '.join([token.lemma_ for token in verb_doc])
            new_obj = ' '.join([token.lemma_ for token in obj_doc])
            new_triplets.append((new_subject, new_verb, new_obj))
        values['triplets'] = new_triplets
    return triplets

def remove_duplicate_words(triplets, logger):
    # Remove any key that has an empty list of triplets
    triplets = {key: values for key, values in triplets.items() if len(values['triplets']) > 0}
    for key, values in triplets.items():
        triplets_list = values['triplets']
        new_triplets = []
        for triplet in triplets_list:
            subject, verb, obj = triplet
            subject_words = subject.split()
            object_words = obj.split()
            subject_words = [word for idx, word in enumerate(subject_words) if idx == 0 or word != subject_words[idx-1]]
            subject = ' '.join(subject_words)
            object_words = [word for idx, word in enumerate(object_words) if idx == 0 or word != object_words[idx-1]]
            obj = ' '.join(object_words)
            new_triplets.append((subject, verb, obj))
        values['triplets'] = new_triplets
    return triplets


def filter_triplets_for_term_length(triplets, min_num_characters, logger):
    """ Clean up the triplets

    Parameters:
        triplets (dict): A dictionary containing the triplets and the text
        logger (logging.Logger): The logger

    Returns:
        pd.DataFrame: The dataframe containing the cleaned up triplets
    """
    # Remove any key that has an empty list of triplets
    triplets = {key: values for key, values in triplets.items() if len(values['triplets']) > 0}
    # remove triplets that are empty
    num_removed = 0
    for key, values in triplets.items():
        triplets_list = values['triplets']
        new_triplets = []
        for triplet in triplets_list:
            subject, verb, obj = triplet
            subject_words = [word for word in subject.split() if len(word) > min_num_characters]
            subject = ' '.join(subject_words)
            object_words = [word for word in obj.split() if len(word) > min_num_characters]
            obj = ' '.join(object_words)
            if len(subject) == 0 or len(obj) == 0:
                num_removed += 1
            else:
                new_triplets.append((subject, verb, obj))
        values['triplets'] = new_triplets
    logger.write_log('Removed ' + str(num_removed) + ' triplets in total when filtering for the number of characters.')
    return triplets

def save_triplets(triplets_dict, path_save_triplets):
    with open(path_save_triplets, 'wb') as file:
        pickle.dump(triplets_dict, file)
    # Now also save the triplets as txt files, first edit the path, change the extension from pkl to txt
    path_save_triplets = path_save_triplets.with_suffix('.txt')
    with open(path_save_triplets, 'w', encoding='utf-8') as file:
        for key, values in triplets_dict.items():
            file.write(key + '\n')
            for triplet in values['triplets']:
                file.write(str(triplet) + '---')
            file.write('\n')
            file.write('---'*20 + '\n')


def process_batch(partial_triplet_dict, queue, process_id, parameters):
    # get the process id
    local_logger = logger(queue=queue, logging_level_file=parameters['logging_level_file'], logging_level_cmd=parameters['logging_level_cmd'], verbose=parameters['verbose'])

    current_time = time.time()
    local_logger.write_log(f'Current time: {time.time()}, Starting lower casing...', prepend='Process ' + str(process_id) + ': ')   
    partial_triplet_dict = lower_case(partial_triplet_dict)
    local_logger.write_log(f'Current time: {time.time()}, Lower casing took ' + str((time.time() - current_time) / 60) + ' minutes, starting to remove undesirable characters...', prepend='Process ' + str(process_id) + ': ')
    current_time = time.time()
    partial_triplet_dict = keep_only_text(partial_triplet_dict)
    local_logger.write_log(f'Current time: {time.time()}, Removing undesirable characters took ' + str((time.time() - current_time) / 60) + ' minutes, starting to remove stopwords...', prepend='Process ' + str(process_id) + ': ')
    current_time = time.time()
    partial_triplet_dict = remove_stopwords(partial_triplet_dict, parameters['filler_verbs'])
    if parameters['use_lemmatization']:
        local_logger.write_log(f'Current time: {time.time()}, Removing stopwords took ' + str((time.time() - current_time) / 60) + ' minutes, starting to lemmatize...', prepend='Process ' + str(process_id) + ': ')
        partial_triplet_dict = lemmatize(partial_triplet_dict, parameters['spacy_model'], local_logger)
    local_logger.write_log(f'Current time: {time.time()}, Lemmatization took ' + str((time.time() - current_time) / 60) + ' minutes, starting to filter for term length...', prepend='Process ' + str(process_id) + ': ')
    current_time = time.time()
    partial_triplet_dict = filter_triplets_for_term_length(partial_triplet_dict, parameters['min_num_characters'], local_logger)
    local_logger.write_log(f'Current time: {time.time()}, Filtering for term length took ' + str((time.time() - current_time) / 60) + ' minutes, starting to remove duplicate words...', prepend='Process ' + str(process_id) + ': ')
    current_time = time.time()
    partial_triplet_dict = remove_duplicate_words(partial_triplet_dict, local_logger)
    local_logger.write_log(f'Current time: {time.time()}, Removing duplicate words took {str((time.time() - current_time) / 60)} minutes, starting to filter for length...', prepend='Process ' + str(process_id) + ': ') 
    current_time = time.time()
    partial_triplet_dict, _ = filter_length(partial_triplet_dict, local_logger, parameters['max_length'])
    local_logger.write_log(f'Current time: {time.time()}, Filtering for length took ' + str((time.time() - current_time) / 60) + ' minutes, done processing the batch.', prepend='Process ' + str(process_id) + ': ')
    return partial_triplet_dict


def process_all_triplets(triplet_dict, num_processes, parameters):
    check_spacy_installed(parameters['spacy_model'])
    with mp.Manager() as manager:
        queue = manager.Queue()
        listener = mp.Process(target=listener_process, args=(queue, listener_configurer, GENERAL_CONTEXT.path_log.joinpath('process_triplets.txt'), GENERAL_CONTEXT.logging_level_file, GENERAL_CONTEXT.logging_level_cmd))
        listener.start()

        pool = mp.Pool(num_processes)
        res = []
        paper_ids = list(triplet_dict.keys())
        chunk_size = len(triplet_dict) // num_processes
        for i in range(num_processes):
            if i == num_processes - 1:
                batch_keys = paper_ids[i*chunk_size:]
            else:
                batch_keys = paper_ids[i*chunk_size:(i+1)*chunk_size]
            batch_dict = {k: triplet_dict[k] for k in batch_keys}
            res.append(pool.apply_async(process_batch, args=(batch_dict, queue, i, parameters)))
        pool.close()
        pool.join()
        triplet_dict = {}
        for r in res:
            triplet_dict.update(r.get())

        queue.put(None)
        listener.join()

        return triplet_dict

def regression_test(time_taken):
    results_regression_test = {}
    path_regression_test_folder = GENERAL_CONTEXT.path_regression_test_folder
    results_regression_test['time_taken'] = time_taken
    
    #load triplets file
    with open(GENERAL_CONTEXT.path_save_processed_triplets_file, 'rb') as file:
        triplets = pickle.load(file)

    # get the number of unique papers
    num_papers = len(triplets)

    results_regression_test['num_files'] = num_papers
    results_regression_test['time_taken'] = time_taken
    # save as json
    with open(path_regression_test_folder.joinpath('process_triplets_test_output.json'), 'w') as f:
        json.dump(results_regression_test, f)

def main():
    begin_time = time.time()
    run_regression_test = True if os.getenv("REGRESSION_TEST") == "1" else False
    global GENERAL_CONTEXT, TRIPLET_PROCESSING_CONTEXT, num_cpus
    GENERAL_CONTEXT = GeneralContext()
    TRIPLET_PROCESSING_CONTEXT = TripletProcessingContext(GENERAL_CONTEXT)
    num_cpus = int(os.getenv("NUM_CPUS_TRIPLET_PROCESSING"))
    
    path_save_triplets_file = GENERAL_CONTEXT.path_save_triplets_file
    path_save_processed_triplets_file = GENERAL_CONTEXT.path_save_processed_triplets_file

    general_logger = logger('general', GENERAL_CONTEXT.path_log.joinpath('process_triplets.txt'), logging_level_file=GENERAL_CONTEXT.logging_level_file, logging_level_cmd=GENERAL_CONTEXT.logging_level_cmd, verbose=GENERAL_CONTEXT.verbose)

    if path_save_processed_triplets_file.exists():
        general_logger.write_log('Processed triplets are already present, proceeding to filtering')
        return

    triplet_dict = process_csv_file(path_save_triplets_file)
    general_logger.write_log('Number of papers: ' + str(len(triplet_dict)))
    general_logger.write_log('Average number of triplets per paper: ' + str(sum([len(values['triplets']) for values in triplet_dict.values()]) / len(triplet_dict)))

    parameters = {
        'filler_verbs': TRIPLET_PROCESSING_CONTEXT.filler_verbs,
        'use_lemmatization': GENERAL_CONTEXT.use_lemmatization,
        'spacy_model': GENERAL_CONTEXT.spacy_model,
        'min_num_characters': TRIPLET_PROCESSING_CONTEXT.min_num_characters,
        'max_length': TRIPLET_PROCESSING_CONTEXT.max_length,
        'logging_level_file': GENERAL_CONTEXT.logging_level_file,
        'logging_level_cmd': GENERAL_CONTEXT.logging_level_cmd,
        'verbose': GENERAL_CONTEXT.verbose
    }

    general_logger.write_log('Number of CPUs according to multiprocessing (mp.cpu_count): ' + str(mp.cpu_count()))
    general_logger.write_log('Number of CPUs according to the context (user input): ' + str(num_cpus))

    triplet_dict = process_all_triplets(triplet_dict, num_cpus, parameters)

    general_logger.write_log('Done processing the triplets, saving the processed triplets...')
    save_triplets(triplet_dict, path_save_processed_triplets_file)

    general_logger.write_log('Done processing the triplets, saved the processed triplets at ' + str(path_save_processed_triplets_file))
    end_time = time.time()
    if run_regression_test:
        regression_test(end_time-begin_time)

if __name__ == '__main__':
    main()
