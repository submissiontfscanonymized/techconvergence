from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from collections import defaultdict
import Levenshtein
import json
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

def subprocess_subsample_from_each_period(files, keywords_for_paper_filtering=[]):
    resulting_files = []
    for idx, file in enumerate(files):
        if idx % 1000 == 0:
            print(f"Processing file {idx} of {len(files)}", flush=True)
        with open(file, "r") as f:
            text = f.read()

        if len(keywords_for_paper_filtering) > 0:
            abstract = get_abstract(text)
            if not any(keyword in abstract for keyword in keywords_for_paper_filtering):
                continue
        resulting_files.append(file)
    return resulting_files

def subsample_from_each_period(path_texts, n_samples=100, keywords_for_paper_filtering=[], num_processes=50, period_dict=None):
    resulting_files = []
    all_files = list(path_texts.rglob("*.txt"))
    chunk_size = len(all_files) // num_processes
    chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]

    pool = mp.Pool(num_processes)
    results = [pool.apply_async(subprocess_subsample_from_each_period, args=(chunk, keywords_for_paper_filtering)) for chunk in chunks]
    pool.close()
    pool.join()

    for result in results:
        resulting_files.extend(result.get())

    final_files = []
    counts = defaultdict(int)
    for file in resulting_files:
        if n_samples == None:
            final_files.append(file)
            continue
        # get the first 4 characters of the filename
        period = file.stem[:4]
        if period_dict is not None:
            period = period_dict[file]
        if counts[period] < n_samples:
            final_files.append(file)
            counts[period] += 1
    print(f"Number of files after filtering: {len(final_files)}")
    return final_files

def get_abstract(text, average_abstract_length=2000):
    abstract_pos = text.find("abstract")
    introduction_pos = text.find("introduction")
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

    abstract = text[starting_pos:starting_pos + abstract_length]
    return abstract

def get_keywords_subprocess(files, kw_model, max_ngram):
    keywords_lists = defaultdict(list)
    for i, file in enumerate(files):
        if i % 25 == 0:
            print(f"Processing file {i} of {len(files)}", flush=True)
        with open(file, "r") as f:
            text = f.read()
        abstract = get_abstract(text)
        keywords = kw_model.extract_keywords(abstract, keyphrase_ngram_range=(1, max_ngram), top_n=5)
        file_name_str = file.stem
        keywords_lists[file_name_str] = keywords
    return keywords_lists


def main():
    # use specter model
    run_name = "patents_llm_2018_to_2024"
    global patents
    patents = True
    sentence_model = SentenceTransformer('allenai-specter')
    kw_model = KeyBERT(model=sentence_model)
    path_texts = Path(f"data/{run_name}/raw_papers_{run_name}/")
    N_SAMPLES = 20
    max_ngram = 3
    num_processes = 50
    path_save_keywords = Path(f"results_{run_name}/keywords")
    keywords_for_paper_filtering = []

    if not path_save_keywords.exists():
        path_save_keywords.mkdir(parents=True, exist_ok=True)

    path_save_keywords_file = path_save_keywords / "keywords_counts.json"
    
    if path_save_keywords_file.exists():
        print('Loading keywords from file', flush=True)
        keyword_counts = json.load(open(path_save_keywords_file, "r"))
    
    else:
        period_dict = None
        if patents:
            print('Making period dictionary')
            # in path_texts we have subfolders 1801, 1802, 1803, etc. For every file in each folder we want to know the period
            period_dict = {}
            for period in path_texts.iterdir():
                for file in period.iterdir():
                    period_dict[file] = period.stem


        print('Extracting keywords', flush=True)
        files = subsample_from_each_period(path_texts, N_SAMPLES, keywords_for_paper_filtering, num_processes, period_dict)


        print('Total number of files:', len(files), flush=True)
        keywords_lists = defaultdict(list)

        # split the files in chunks
        chunk_size = len(files) // num_processes
        chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

        pool = mp.Pool(num_processes)
        results = [pool.apply_async(get_keywords_subprocess, args=(chunk, kw_model, max_ngram)) for chunk in chunks]
        pool.close()
        pool.join()

        for result in results:
            keywords_lists.update(result.get())

        # now we want to have one dictionary, with for every keyword the number of times it appears
        keyword_counts = defaultdict(int)
        for keywords in keywords_lists.values():
            for keyword in keywords:
                keyword_counts[keyword[0]] += 1

        path_pickle = path_save_keywords / "keyword_counts.pkl"
        with open(path_pickle, "wb") as f:
            pickle.dump(keyword_counts, f)
        # save as json
        with open(path_save_keywords_file, "w") as f:
            json.dump(keyword_counts, f)

if __name__ == "__main__":
    main()
