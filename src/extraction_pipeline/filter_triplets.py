import pickle
import string
import multiprocessing as mp
from collections import Counter
import numpy as np
from datasets import load_dataset

import time
import os
import json

from extraction_pipeline.helpers import logger, listener_configurer, listener_process, load_spacy, lemmatize_text
from extraction_pipeline.contexts import GeneralContext, TripletFilteringContext, EntropyContext
mp.set_start_method('spawn', force=True)

def get_words_from_subject_object(triplets):
    """ Get the words from the triplets

    Parameters:
        triplets (dict): The triplets

    Returns:
        list[str]: A list of words
    """
    all_words = set()
    for key, value in triplets.items():
        for triplet in value['triplets']:
            subject, _, obj = triplet
            subject_words = subject.split()
            obj_words = obj.split()
            all_words.update(subject_words)
            all_words.update(obj_words)
    # make it a list
    all_words = list(all_words)
    return all_words

def compute_frequency(triplets, path_word_freq, gen_logger):
    freq_counter = Counter()
    for idx, (key, value) in enumerate(triplets.items()):
        if idx % 25000 == 0:
            gen_logger.write_log(f'Updating frequency dictionary with triplet number {idx} out of {len(triplets)}')
        for triplet in value['triplets']:
            subject, _, obj = triplet
            subject_words = subject.split()
            obj_words = obj.split()
            for word in subject_words:
                freq_counter[word] += 1
            for word in obj_words:
                freq_counter[word] += 1
    with open(path_word_freq, 'wb') as f:
        pickle.dump(freq_counter, f)
    
    return freq_counter



def filter_by_frequency(triplets, path_word_freq, gen_logger, min_num_appearances=5):
    if not path_word_freq.exists():
        gen_logger.write_log("Computing word frequency")
        word_freq_papers = compute_frequency(triplets, path_word_freq, gen_logger)
    else:
        gen_logger.write_log("Loading paper frequency")
        with open(path_word_freq, 'rb') as f:
            word_freq_papers = pickle.load(f)

    all_words = get_words_from_subject_object(triplets)
    words_to_remove = [word for word in all_words if word_freq_papers[word] < min_num_appearances]
    gen_logger.write_log(f"Number of words to remove by frequency: {len(words_to_remove)}")
    gen_logger.write_log(f"Percentage of words to remove by frequency: {(len(words_to_remove) / len(all_words)) if len(all_words) != 0 else 'NaN'}")
    return words_to_remove

def truncate_book_corpus(book_corpus, max_length):
    """ Filter the book corpus based on length

    Parameters:
        book_corpus (list[str]): The book corpus
        max_length (int): The maximum length

    Returns:
        list[str]: The filtered book corpus
    """
    # For every book, truncate all characters after max_length
    book_corpus = [book[:max_length] for book in book_corpus]
    return book_corpus

def get_bookcorpus_frequency_subprocess(batch_corpus, parameters, queue):
    if parameters['use_lemmatization']:
        nlp = load_spacy(parameters['spacy_model'])
        batch_corpus = [' '.join(lemmatize_text(text, nlp)) for text in batch_corpus]
    # get fraction subset of the corpus
    document_counts = Counter()
    for i in range(len(batch_corpus)):
        # at every 10% of the corpus, log the progress
        if i % 250 == 0:
            queue.put(f"Processing {i} out of {len(batch_corpus)}")
        # lower case everything
        words_book = batch_corpus[i].split()
        # lower case the words
        words_book = [word.lower() for word in words_book]
        unique_words = set(words_book)
        for word in unique_words:
            if words_book.count(word) >= parameters['min_num_counts_term_in_single_book']:
                document_counts[word] += 1
    return document_counts

def get_bookcorpus_frequency(corpus, parameters):
    """ Get the frequency of the words in the corpus

    Parameters:
        corpus (list[str]): The corpus
        subset (float): The fraction of the corpus to use
        min_terms (int): The minimum number of terms

    Returns:
        Counter: A counter containing the frequency of the words
    """

    num_parts = parameters['num_cpus']
    part_size = len(corpus) // num_parts
    corpus_parts = [corpus[i*part_size:(i+1)*part_size] for i in range(num_parts)]
    corpus_parts.append(corpus[num_parts*part_size:])

    with mp.Manager() as manager:
        queue = manager.Queue()
        listener = mp.Process(target=listener_process, args=(queue, listener_configurer, parameters['path_log'],  parameters['log_level_file'], parameters['log_level_cmd']))
        listener.start()
        pool = mp.Pool(num_parts)
        results = [pool.apply_async(get_bookcorpus_frequency_subprocess, args=(part, parameters, queue)) for part in corpus_parts]
        pool.close()
        pool.join()

        word_freq = Counter()
        for res in results:
            word_freq.update(res.get())

        queue.put(None)
        listener.join()

    return word_freq

def compute_term_scores(corpus_freq, paper_freq, num_docs_corpus, num_docs_paper, min_paper_count):
    """ Compute the term scores

    Parameters:
        corpus_freq (Counter): The frequency of the words in the corpus
        paper_freq (Counter): The frequency of the words in the papers
        num_docs_corpus (int): The number of documents in the corpus
        num_docs_paper (int): The number of documents in the papers
        min_paper_count (int): The minimum number of papers

    Returns:
        dict: A dictionary containing the term scores
    """
    # Now make a new dictionary that is the multiplication of the two. If a key is in corpus_freq but not in paper_freq, we set the value to 0
    # if a key is in paper_freq but not in corpus_freq, and the value in paper_freq is at least 10, we set the value to infinity.
    # if a key is in paper_freq but not in corpus_freq, and the value in paper_freq is less than 10, we set the value to 0
    term_scores = {}
    for key in corpus_freq:
        if key in paper_freq and paper_freq[key] >= min_paper_count:
            corpus_score = - np.log(corpus_freq[key]/num_docs_corpus)
            paper_score = np.log(paper_freq[key]/num_docs_paper)
            term_scores[key] = corpus_score + paper_score
        else:
            term_scores[key] = - np.inf
    for key in paper_freq:
        if key not in corpus_freq:
            if paper_freq[key] >= min_paper_count:
                term_scores[key] = np.inf
            else:
                term_scores[key] = - np.inf
    # From the dictionary, remove all keys where there is punctuation, and remove all keys existing of only numbers
    term_scores_filtered = {key: value for key, value in term_scores.items() if (not any(char in string.punctuation for char in key) and not key.isdigit() and len(key) >= 2)}
    # sort term_scores from highest score to lowest
    term_scores_filtered = dict(sorted(term_scores_filtered.items(), key=lambda x: x[1], reverse=True))
    return term_scores_filtered

def get_words_to_remove_bookcorpus(triplets_dict, term_scores, cutoff=0.1):
    all_words = get_words_from_subject_object(triplets_dict)
    term_scores_restricted = {}
    for word in all_words:
        if word not in term_scores:
            term_scores_restricted[word] = np.inf
        else:
            term_scores_restricted[word] = term_scores[word]
    # sort the words by their scores, lowest first
    term_scores_restricted = {k: v for k, v in sorted(term_scores_restricted.items(), key=lambda item: item[1])}
    # Now we get the first 10% of the words
    words_to_remove = list(term_scores_restricted.keys())[:int(cutoff * len(term_scores_restricted))]
    return words_to_remove


def filter_with_bookcorpus(triplets, general_logger):
    ###################################  LOAD BOOK CORPUS  ###################################################
    path_book_corpus = TRIPLET_FILTERING_CONTEXT.path_book_corpus_gutenberg
    if path_book_corpus.exists():
        general_logger.write_log("Loading book corpus from file")
        book_corpus_gutenberg = pickle.load(open(path_book_corpus, "rb"))
    else:
        general_logger.write_log("Loading book corpus from huggingface")
        book_corpus_gutenberg = load_dataset("sedthh/gutenberg_english")['train']['TEXT']
        book_corpus_gutenberg = truncate_book_corpus(book_corpus_gutenberg, TRIPLET_FILTERING_CONTEXT.max_length_book)
        # save book corpus
        pickle.dump(book_corpus_gutenberg, open(path_book_corpus, "wb"))
    
    # take the subset, use a numpy random generator
    subset_bookcorpus = TRIPLET_FILTERING_CONTEXT.subset_book_corpus
    random_indices = np.random.default_rng(seed=GENERAL_CONTEXT.seed).choice(len(book_corpus_gutenberg), size=int(len(book_corpus_gutenberg)*subset_bookcorpus), replace=False)
    book_corpus_gutenberg = [book_corpus_gutenberg[i] for i in random_indices]
    general_logger.write_log(f"Number of books in the book corpus, with subset size {subset_bookcorpus}: {len(book_corpus_gutenberg)}")
    ################################### PROCESS BOOK CORPUS  ###################################################
    path_term_frequency_bookcorpus = TRIPLET_FILTERING_CONTEXT.path_term_frequency_bookcorpus
    if not path_term_frequency_bookcorpus.exists():
        general_logger.write_log("Computing book frequency")
        parameters = {
            'use_lemmatization': GENERAL_CONTEXT.use_lemmatization,
            'min_num_counts_term_in_single_book': TRIPLET_FILTERING_CONTEXT.min_num_counts_term_in_single_book,
            'spacy_model': GENERAL_CONTEXT.spacy_model,
            'num_cpus': num_cpus,
            'path_log': GENERAL_CONTEXT.path_log.joinpath('filter_triplets.txt'),
            'log_level_file': GENERAL_CONTEXT.logging_level_file,
            'log_level_cmd': GENERAL_CONTEXT.logging_level_cmd
        }
        word_freq_book = get_bookcorpus_frequency(book_corpus_gutenberg, parameters)
        with open(path_term_frequency_bookcorpus, 'wb') as f:
            pickle.dump(word_freq_book, f)
    else:
        general_logger.write_log("Loading book frequency")
        with open(path_term_frequency_bookcorpus, 'rb') as f:
            word_freq_book = pickle.load(f)
    

    ################################### PROCESS PAPERS  ###################################################
    path_paper_freq = ENTROPY_CONTEXT.path_save_category_counts
    if not path_paper_freq.exists():
        raise ValueError("Paper frequency file does not exist, please compute it by running entropy.py")
    else:
        general_logger.write_log("Loading paper frequency")
        with open(path_paper_freq, 'rb') as f:
            word_freq_papers = pickle.load(f)
        word_freq_papers = pool_categories(word_freq_papers)
            
    ################################### COMPUTE TERM SCORES  ###################################################
    path_save_term_scores = TRIPLET_FILTERING_CONTEXT.path_save_term_scores_bookcorpus
    if not path_save_term_scores.exists():
        num_docs_corpus = len(book_corpus_gutenberg)
        num_files_per_category = pickle.load(open(ENTROPY_CONTEXT.path_num_files_per_category, 'rb'))
        num_docs_paper = sum(num_files_per_category[key] for key in num_files_per_category.keys())
        general_logger.write_log(f'Number of papers used for the term score computation based on bookcorpus and academic corpus: {num_docs_paper}')
        general_logger.write_log("Computing term scores based on the bookcorpus and academic corpus.")
        term_scores = compute_term_scores(word_freq_book, word_freq_papers, num_docs_corpus, num_docs_paper, TRIPLET_FILTERING_CONTEXT.min_num_paper_appearances_term_bookcorpus_scores)
        with open(path_save_term_scores, 'wb') as f:
            pickle.dump(term_scores, f)

    else:
        general_logger.write_log("Loading term scores")
        with open(path_save_term_scores, 'rb') as f:
            term_scores = pickle.load(f)

    words_to_remove = get_words_to_remove_bookcorpus(triplets, term_scores, TRIPLET_FILTERING_CONTEXT.cutoff_bookcorpus)
    general_logger.write_log(f"Number of words to remove by bookcorpus: {len(words_to_remove)}")
    return words_to_remove 

def get_words_to_filter_entropy(triplets, logger):
    all_words = get_words_from_subject_object(triplets)

    logger.write_log("Loading entropy dictionary")
    with open(ENTROPY_CONTEXT.path_save_entropy_file, 'rb') as f:
        entropy = pickle.load(f)
    
    logger.write_log("Calculating percentage of words missed in the entropy dictionary")
    #calculate perc_missed simpler than above
    perc_missed = sum([1 for word in all_words if word not in entropy.keys()]) / len(all_words)
    logger.write_log(f"Percentage of words missed in the entropy dictionary: {perc_missed}")

    # new_entropy = {word: ent for word, ent in entropy.items() if word in all_words}
    # logger.write_log(f"Number of words in the filtered entropy dictionary: {len(new_entropy)}")


    
    # Now we get the top perc_entropy percent of the words with the highest entropy
    threshold = np.percentile(list(entropy.values()), TRIPLET_FILTERING_CONTEXT.entropy_percentile_threshold * 100)

    #entropy threshold log
    logger.write_log(f"Entropy threshold: {threshold}")

    words_to_remove = set([word for word, ent in entropy.items() if ent > threshold])

    logger.write_log(f"Number of words to remove by entropy: {len(words_to_remove)}")
    return words_to_remove


def filter_by_word_list_subprocess(triplets, parameters, queue):
    words_to_remove = parameters['words_to_remove']
    local_logger = logger(queue=queue, logging_level_file = parameters['logging_level_file'], logging_level_cmd = parameters['logging_level_cmd'], verbose=parameters['verbose'])
    removed_triplets = []
    num_items = len(triplets)
    local_logger.write_log("Starting filtering")
    for i, (key, value) in enumerate(triplets.items()):
        if i % 250 == 0:
            local_logger.write_log(f"Filtering triplets based on the combined lists from frequency/bookcorpus/entropy, processing number {i} out of {num_items}")
        triplet_list = value['triplets']
        kept_triplets = []
        for triplet in triplet_list:
            subject, _, obj = triplet
            object_words = obj.split()
            subject_words = subject.split()
            # Now we remove the words that are in words_to_remove, if no words are left, we remove the triplet
            if all(word in words_to_remove for word in object_words) or all(word in words_to_remove for word in subject_words):
                removed_triplets.append(triplet)
            else:
                kept_triplets.append(triplet)

        triplets[key]['triplets'] = kept_triplets

    # Now remove all keys that have no triplets left
    triplets = {key: value for key, value in triplets.items() if len(value['triplets']) > 0}

    return triplets, removed_triplets

def filter_by_word_list(triplets, parameters):
    num_processes = parameters['num_cpus']
    # use multiprocessing to speed up the process
    num_triplets = len(triplets)
    num_triplets_per_process = num_triplets // num_processes

    with mp.Manager() as manager:
        queue = manager.Queue()
        listener = mp.Process(target=listener_process, args=(queue, listener_configurer, parameters['path_log'], parameters['logging_level_file'], parameters['logging_level_cmd']))
        listener.start()
        pool = mp.Pool(num_processes)
        res = []
        for i in range(num_processes):
            start = i * num_triplets_per_process
            end = (i + 1) * num_triplets_per_process
            if i == num_processes - 1:
                end = num_triplets
            triplets_sub = {key: value for idx, (key, value) in enumerate(triplets.items()) if idx >= start and idx < end}
            res.append(pool.apply_async(filter_by_word_list_subprocess, args=(triplets_sub, parameters, queue)))
    
        pool.close()
        pool.join()

        triplets = {}
        removed_triplets = []
        for i in range(num_processes):
            res_sub = res[i].get()
            triplets.update(res_sub[0])
            removed_triplets.extend(res_sub[1])

        queue.put(None)
        listener.join()

    # save the removed triplets in a txt file
    with open(parameters['path_removed_triplets'], 'w', encoding='utf-8') as f:
        for triplet in removed_triplets:
            f.write(str(triplet) + '\n')
            
    # return the triplets
    return triplets



def pool_categories(category_counts):
    overall_counter = Counter()
    # category counts has diction
    for _, counts in category_counts.items():
        overall_counter += counts
    return overall_counter

def save_triplets(triplets_dict, path_save_triplets):
    # first save the dict as a pkl file
    with open(path_save_triplets, 'wb') as file:
        pickle.dump(triplets_dict, file)
    # Now also save the triplets as txt files
    path_save_triplets_txt_suffix = path_save_triplets.with_suffix('.txt')
    with open(path_save_triplets_txt_suffix, 'w', encoding='utf-8') as file:
        for key, values in triplets_dict.items():
            file.write(key + '\n')
            for triplet in values['triplets']:
                file.write(str(triplet) + '---')
            file.write('\n')
            file.write('---'*20 + '\n')

def save_removed_words(words_to_remove, path_save_removed_words):
    with open(path_save_removed_words, 'w') as f:
        for word in words_to_remove:
            f.write(word + '\n')

def load_removed_words(path_save_removed_words):
    with open(path_save_removed_words, 'r') as f:
        words_to_remove = f.readlines()
        words_to_remove = [word.strip() for word in words_to_remove]
    return words_to_remove

def regression_test(time_taken):
    results_regression_test = {}
    path_regression_test_folder = GENERAL_CONTEXT.path_regression_test_folder
    results_regression_test['time_taken'] = time_taken
    
    #load the filtered triplets, and check if the number of papers is correct
    with open(GENERAL_CONTEXT.path_save_final_triplets_file, 'rb') as f:
        triplets = pickle.load(f)
        results_regression_test['num_files'] = len(triplets)

    with open(path_regression_test_folder.joinpath('filter_triplets_test_output.json'), 'w') as f:
        json.dump(results_regression_test, f)

def main():
    ###################################   SETTINGS  ###################################################
    begin_time = time.time()
    run_regression_test = True if os.getenv("REGRESSION_TEST") == "1" else False
    global GENERAL_CONTEXT, ENTROPY_CONTEXT, TRIPLET_FILTERING_CONTEXT, num_cpus
    GENERAL_CONTEXT = GeneralContext()
    TRIPLET_FILTERING_CONTEXT = TripletFilteringContext(GENERAL_CONTEXT)
    ENTROPY_CONTEXT = EntropyContext(GENERAL_CONTEXT)
    num_cpus = int(os.getenv("NUM_CPUS_TRIPLET_PROCESSING"))

    path_save_triplet_filtering_folder = GENERAL_CONTEXT.path_save_triplet_filtering_folder
    path_save_final_triplets = GENERAL_CONTEXT.path_save_final_triplets_file

    ####################################################################################################
    general_logger = logger('triplet_filtering', GENERAL_CONTEXT.path_log.joinpath('filter_triplets.txt'), logging_level_file=GENERAL_CONTEXT.logging_level_file, logging_level_cmd=GENERAL_CONTEXT.logging_level_cmd, verbose=GENERAL_CONTEXT.verbose)
    refilter_triplets = TRIPLET_FILTERING_CONTEXT.refilter_triplets
   
    ################### LOAD TRIPLETS  ############################
    with open(GENERAL_CONTEXT.path_save_processed_triplets_file, 'rb') as f:
        triplets = pickle.load(f)

    ################### FILTERING ############################
    if (not path_save_final_triplets.exists()) or refilter_triplets:
        
        ################### FILTER TRIPLETS BY FREQUENCY ############################
        path_removed_words_frequency = path_save_triplet_filtering_folder.joinpath('removed_words_frequency.txt')
        if not path_removed_words_frequency.exists() or refilter_triplets:
            general_logger.write_log("Words to remove by frequency do not exist, computing them")
            words_to_remove_frequency = filter_by_frequency(triplets, TRIPLET_FILTERING_CONTEXT.path_word_frequency, general_logger, TRIPLET_FILTERING_CONTEXT.min_num_appearances_for_frequency)
            save_removed_words(words_to_remove_frequency, path_removed_words_frequency)

        else:
            general_logger.write_log("Words to remove by frequency exist, loading them")
            words_to_remove_frequency = load_removed_words(path_removed_words_frequency)

        ################### FILTER TRIPLETS USING BOOKCORPUS ############################
        path_removed_words_bookcorpus = path_save_triplet_filtering_folder.joinpath('removed_words_bookcorpus.txt')
        if not path_removed_words_bookcorpus.exists() or refilter_triplets:
            general_logger.write_log("Words to remove by bookcorpus do not exist, computing them")
            words_to_remove_bookcorpus = filter_with_bookcorpus(triplets, general_logger)
            save_removed_words(words_to_remove_bookcorpus, path_removed_words_bookcorpus)
        else:
            general_logger.write_log("Words to remove by bookcorpus exist, loading them")
            words_to_remove_bookcorpus = load_removed_words(path_removed_words_bookcorpus)

        ################### FILTER TRIPLETS BY ENTROPY ############################
        if TRIPLET_FILTERING_CONTEXT.use_entropy:
            path_removed_words_entropy = path_save_triplet_filtering_folder.joinpath('removed_words_entropy.txt')
            if not path_removed_words_entropy.exists() or refilter_triplets:
                general_logger.write_log("Words to remove by entropy do not exist, computing them")
                words_to_remove_entropy = get_words_to_filter_entropy(triplets, general_logger)
                save_removed_words(words_to_remove_entropy, path_removed_words_entropy)
            else:
                general_logger.write_log("Words to remove by entropy exist, loading them")
                words_to_remove_entropy = load_removed_words(path_removed_words_entropy)

        ################### COMBINE THE REMOVED WORDS ############################
        if TRIPLET_FILTERING_CONTEXT.use_entropy:
            words_to_remove_combined = set(words_to_remove_frequency).union(set(words_to_remove_bookcorpus)).union(set(words_to_remove_entropy))
        else:
            words_to_remove_combined = set(words_to_remove_frequency).union(set(words_to_remove_bookcorpus))
        save_removed_words(words_to_remove_combined, path_save_triplet_filtering_folder.joinpath('removed_words_combined.txt'))

        parameters = {
            'words_to_remove': words_to_remove_combined,
            'path_removed_triplets': TRIPLET_FILTERING_CONTEXT.path_removed_triplets,
            'path_log': GENERAL_CONTEXT.path_log.joinpath('filter_triplets.txt'),
            'logging_level_file': GENERAL_CONTEXT.logging_level_file,
            'logging_level_cmd': GENERAL_CONTEXT.logging_level_cmd,
            'num_cpus': num_cpus,
            'verbose': GENERAL_CONTEXT.verbose,
        }

        general_logger.write_log('Number of CPUs according to multiprocessing (mp.cpu_count): ' + str(mp.cpu_count()))
        general_logger.write_log('Number of CPUs according to the context (user input): ' + str(num_cpus))

        triplets = filter_by_word_list(triplets, parameters)

        save_triplets(triplets, path_save_final_triplets)

        general_logger.write_log('Triplets are filtered and saved in ' + str(path_save_final_triplets))

    else:
        general_logger.write_log("Final triplets already exist, skipping filtering")
    
    end_time = time.time()
    if run_regression_test:
        regression_test(end_time-begin_time)

if __name__ == "__main__":
    main()



