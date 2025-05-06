import os
import logging
import logging.handlers
from pathlib import Path
import json
import pickle
import random
import spacy
from time import sleep
from transformers import set_seed
from spacy.tokenizer import Tokenizer
from spacy.cli.download import download
import torch

logging.basicConfig(level=logging.DEBUG)

def remove_folder(path_folder):
    """Remove a folder and its content

    Parameters:
        path_folder (Path): path to the folder to remove
    """
    if path_folder.exists() and path_folder.is_dir():
        for file in path_folder.iterdir():
            if file.is_file():
                file.unlink()
            else:
                remove_folder(file)
        path_folder.rmdir()

def get_paths_armasuisse_cluster(months, years):
    path_raw_pdf = []
    path_base_pdf = Path(r'/cluster/raid/data/arxiv/arxiv/pdf/')
    path_base_pdf_recent = Path(r'/cluster/raid/data/stea/arxiv')

    for year in years:
        for month in months:
            date_path = year + month
            if (year in ['22', '23', '24']) or (year == '21' and month in ['10', '11', '12']):
                path_raw_pdf.append(path_base_pdf_recent.joinpath(date_path))
            else:
                path_raw_pdf.append(path_base_pdf.joinpath(date_path))

    return path_raw_pdf

####################### Logging tools for multiprocessing #######################
def listener_configurer(path_log, logging_level_file, logging_level_cmd):
    root = logging.getLogger()
    # file_handler = logging.FileHandler(path_log, 'a', 'utf-8')
    # formatter_file_handler = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    # file_handler.setFormatter(formatter_file_handler)
    # file_handler.setLevel(logging_level_file)

    # stream_handler = logging.StreamHandler()
    # formatter_stream_handler = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    # stream_handler.setFormatter(formatter_stream_handler)
    # stream_handler.setLevel(logging_level_cmd)

    # root.addHandler(file_handler)
    # root.addHandler(stream_handler)
    root.setLevel(logging.DEBUG)

def listener_process(queue, configurer, path_log, logging_level_file, logging_level_cmd):
    configurer(path_log, logging_level_file, logging_level_cmd)
    while True:
        record = queue.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


####################### Logging for single process #######################

class logger:
    def __init__(self, name=None, path_log=None, logging_level_file=None, logging_level_cmd=None, queue=None, verbose=True):
        if queue is not None:
            self._logger = logging.getLogger(name)
            self._logger.setLevel(logging.DEBUG)
            self.queue = queue
            self.queuehandler = logging.handlers.QueueHandler(queue)
            self.queuehandler.setFormatter(logging.Formatter('%(asctime)s %(processName)-10s %(levelname)-8s %(message)s'))
            self._logger.addHandler(self.queuehandler)
            self.verbose = verbose

        else:
            self.queue = None
            self._logger = logging.getLogger(name)
            self._logger.setLevel(logging.DEBUG)

            # File handler
            self.file_handler = logging.FileHandler(path_log, mode='a', encoding='utf-8')
            self.file_handler.setLevel(logging_level_file)
            self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s'))
            self._logger.addHandler(self.file_handler)

            # Stream handler
            self.stream_handler = logging.StreamHandler()
            self.stream_handler.setLevel(logging_level_cmd)
            self.stream_handler.setFormatter(logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s'))
            self._logger.addHandler(self.stream_handler)

            self.verbose = verbose

    def get_log_changing_sentence(self, old_sentence, new_sentence, line, start_positions, end_positions, type):
        """ Write a log file with the changes made to the sentences in the corpus.

            Parameters
                old_sentence (str): the original sentence
                new_sentence (str): the sentence after the changes
                line (int): the line number in the file
                start_positions (list): the starting positions of the words that were changed
                end_positions (list): the ending positions of the words that were changed
                type (str): the type of change that was made
                log_file (str): the name of the log file

            Returns
                None

        """
        to_log = old_sentence + '\n'
        # start and end positions are lists of integers indicating the positions of the words that were changed in the sentence
        next_line = ''
        for i in range(len(old_sentence)):
            # if i is between two start and end positions, then put a * in the log file. E..g between start_positions[1] and end_positions[1]
            for k in range(len(start_positions)):
                if i >= start_positions[k] and i < end_positions[k]:
                    next_line += '*'
                    break
            else:
                next_line += '-'
        to_log += next_line + '\n'
        to_log += 'Line: ' + str(line) + ', start: ' + str(start_positions) + ', end: ' + str(end_positions) + ', type: ' + type + '\n'
        to_log += new_sentence + '\n\n\n'
        return to_log

    def write_log(self, text, level=logging.INFO, prepend=''):
        """
        Log a text in the log file

        Parameters:
            text (str): text to log
        """
        if prepend:
            text = prepend + ' ' + text
        match level:
            case logging.DEBUG:
                if self.verbose:
                    self._logger.debug(text)
            case logging.INFO:
                if self.verbose:
                    self._logger.info(text)
            case logging.WARNING:
                self._logger.warning(text)
            case logging.ERROR:
                self._logger.error(text)
            case logging.CRITICAL(text):
                self._logger.critical(text)
            case _:
                self._logger.warning(f'The following message has been logged with an invalid log level: {level}:{text}')
        
    def close(self):
        if self.queue is None:
            self._logger.removeHandler(self.file_handler)
            self._logger.removeHandler(self.stream_handler)
            self.file_handler.close()
            self.stream_handler.close()
        else:
            self._logger.removeHandler(self.queuehandler)
            self.queuehandler.close()
        


def get_device(gpu_id=None):
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def get_category_dictionary(path_metadata, path_cat_dict):
    if path_cat_dict.exists():
        with open(path_cat_dict, 'r') as file:
            category_dict = json.load(file)

    else:
        with open(path_metadata, 'rb') as f:
            metadata = pickle.load(f)

            # filter for the rows with the column id that do not contain any letters
            metadata = metadata[metadata['id'].str.contains(r'[a-zA-Z]') == False]

            # The categories column is a string "math.CA math.FA" etc. We want to split it into a list of categories
            metadata['categories'] = metadata['categories'].apply(lambda x: x.split(' '))

            # Now we want to make a dictionary with the id as the key and the categories as the value
            category_dict = metadata.set_index('id')['categories'].to_dict()

            with open(path_cat_dict, 'w') as file:
                json.dump(category_dict, file)
            
    return category_dict

def initialize_random_state(seed=-1):
    """Initialize random state

    Parameters:
        seed (int): random seed if you want to set it to replicate results (default: -1)
                    -1 mean ask for a new seed

    Returns:
        random state (int)
    """
    if seed == -1:
        seed = random.randint(0, 4294967295)
    
    set_seed(seed)

    return seed

def check_spacy_installed(spacy_model: str):
    """Check if spacy model is installed

    Parameters:
        spacy_model (str): spacy model to check

    Returns:
        None
    """
    try:
        spacy.load(spacy_model)
    except OSError:
        # load the model
        download(spacy_model)


def load_spacy(spacy_model: str, disable_list: list[str] = ['parser', 'senter', 'ner'], use_special_tokenizer=True) -> spacy.language.Language:
    """Load the spacy model

    Parameters:
        disable_list (list): list of components to disable in spacy (default: ['parser', 'senter', 'ner'])
        use_special_tokenizer (bool): whether to use special tokenizer or not (default: True

    Returns:
        spacy model
    """
    #spacy.prefer_gpu()
    nlp = spacy.load(spacy_model, disable=disable_list)

    if use_special_tokenizer:
        infixes = nlp.Defaults.prefixes + [r"[-]~"]

        infix_re = spacy.util.compile_infix_regex(infixes)

        def custom_tokenizer(nlp):
            return Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

        nlp.tokenizer = custom_tokenizer(nlp)

    return nlp

def lemmatize_text(text: str, nlp: spacy.language.Language) -> str:
    """Lemmatize the text

    Parameters:
        text (str): text to lemmatize
        nlp (spacy model): spacy model to use for lemmatization

    Returns:
        lemmatized text (str)
    """
    # return lemmatized text as a string, not a list
    return ' '.join([token.lemma_ for token in nlp(text) if not token.is_punct and not token.is_space])
