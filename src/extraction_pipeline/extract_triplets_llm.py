import csv
import gc
import json
import math
import os
import regex
import sys
import warnings
from pathlib import Path
from dotenv import load_dotenv

import time
import multiprocessing as mp

import logging
import torch
import time
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from huggingface_hub import login

from contexts import GeneralContext, TripletExtractionContext
from helpers import logger, get_device, listener_configurer, listener_process
from estimation import find_total_usable_memory, estimate_memory, infer_best_batch_size_by_heuristics, parameters_count, find_total_usable_memory_device

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

mp.set_start_method('spawn', force=True)
set_seed(0)

def get_prompt_string_starling(raw_text, tokenizer, output=None, use_fewshot_prompting=True, few_shot_examples=None):
    """
    Function to get the prompt string for the Starling model, this is separate as it is not included in the apply_chat_template function

    Parameters:
        raw_text (str): The raw text to be used in the prompt
        tokenizer (AutoTokenizer): The tokenizer to be used
        output (str): The output to be included, this is only required when doing fine-tuning and the output is already known
        use_fewshot_prompting (bool): Whether to use few-shot prompting or not
        few_shot_examples (list): The few-shot examples to be used

    Returns:
        prompt (str): The prompt string to be used
    """

    if use_fewshot_prompting:
        chat = few_shot_examples.copy()
        chat.append({"role": "user", "content": "The next text is: \n {}".format(raw_text)})
    else:
        chat = [{"role": "user", "content": "You will extract the subject-predicate-object triplets from the text and return them in the form [(subject_1; predicate_1; object_1), (subject_2; predicate_2; object_2), ..., (subject_n; predicate_n; object_n)]. We want the subjects and objects in the triplets to be specific, they cannot be pronouns or generic nouns. The first text is: \n {}".format(raw_text)}]

    prompt = ''
    for item in chat:
        if item['role'] == 'user':
            prompt += f'GPT4 Correct User: {item["content"]}<|end_of_turn|>'
        else:
            prompt += f'GPT4 Correct Assistant: {item["content"]}<|end_of_turn|>'

    if output is not None:
        prompt += f'GPT4 Correct Assistant: {output}'
    else:
        prompt += 'GPT4 Correct Assistant:'

    return prompt

def get_prompt_string(raw_text, tokenizer, output=None, few_shot_examples=None, use_fewshot_prompting=True):
    """
    Function to get the prompt string for the model

    Parameters:
        raw_text (str): The raw text to be used in the prompt
        tokenizer (AutoTokenizer): The tokenizer to be used
        output (str): The output to be included, this is only required when doing fine-tuning and the output is already known
        few_shot_examples (list): The few-shot examples to be used
        use_fewshot_prompting (bool): Whether to use few-shot prompting or not
        summarize (bool): Whether to summarize the text or not

    Returns:
        prompt (str): The prompt string to be used
    """
    if use_fewshot_prompting:
        assert few_shot_examples is not None, 'Few shot examples must be provided when using few-shot prompting'

    if use_fewshot_prompting:
        chat = few_shot_examples.copy()
        chat.append({"role": "user", "content": "The next text is: \n {}".format(raw_text)})
    else:
        chat = [{"role": "user", "content": "You will extract the subject-predicate-object triplets from the text and return them in the form [(subject_1; predicate_1; object_1), (subject_2; predicate_2; object_2), ..., (subject_n; predicate_n; object_n)]. We want the subjects and objects in the triplets to be specific, they cannot be pronouns or generic nouns. The first text is: \n {}".format(raw_text)}]
        
    if output is not None:
        chat.append({"role": "assistant", "content": output})
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    else:
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    return prompt

def load_test_data_folder_text_files(path, existing_triplets=None, logger=None):
    """
    Function to load the test data from a folder containing text files

    Parameters:
        path (str): The path to the folder containing the test data

    Returns:
        test_files (list): A list containing the test files
        names (list): A list containing the names of the test files
    """
    # get all the test files in a list, test files are .txt
    test_files = []
    paper_ids = []
    total_num_files = len(list(path.rglob('*.txt')))

    if existing_triplets is not None:
        # get the names of the files that have already been processed
        existing_files = existing_triplets['paper_id'].unique()
        num_existing_files = len(existing_files)
        if logger is not None:
            logger.write_log(f'Total number of files: {total_num_files}, number of files already processed: {num_existing_files}, number of files to process: {total_num_files - num_existing_files}')

    for file in path.rglob('*.txt'):
        if existing_triplets is not None:
            if file.name in existing_files:
                continue

        test_files.append(file)
        # get the name of the file, in the format name.txt
        paper_ids.append(file.name)
    return test_files, paper_ids

def get_data_splits(processed_text_files, paper_ids, num_gpus):
    """
    Function to split the data into the number of gpus

    Parameters:
        processed_text_files (list): A list containing the processed text files
        num_gpus (int): The number of gpus

    Returns:
        data_splits (list): A list containing the data splits
    """
    data_splits = []
    paper_ids_splits = []
    for i in range(num_gpus):
        data_splits.append(processed_text_files[i::num_gpus])
        paper_ids_splits.append(paper_ids[i::num_gpus])
    return data_splits, paper_ids_splits


def prepare_test_data(test_files, size_per_model_call, logger=None):
    """
    Function to prepare the test data

    Parameters:
        test_files (list): A list containing the test files
        size_per_model_call (int): The size of the batch
        logger (Logger): The logger to be used

    Returns:
        test_data (list): A list containing lists of the test data, the inner lists are in batches of size size_per_batch
    """

    # split the text into chunks of max_len, split on \n
    test_data = []

    for file in test_files:
        subfiles = []
        with open(file, 'r') as f:
            # read the lines
            lines = f.readlines()

        # split it into chunks of lines size_per_batch
        for i in range(0, len(lines), size_per_model_call):
            # make a string with the lines, separated by \n
            text = ''.join(lines[i:i+size_per_model_call])
            subfiles.append(text)
            
        test_data.append(subfiles)
    return test_data

def get_prompts(processed_text_files, tokenizer, model_id, few_shot_examples, use_fewshot_prompting):
    prompts = []
    for i, file in enumerate(processed_text_files):
        prompts_temp = []
        for j, text in enumerate(file):
            if model_id == "Nexusflow/Starling-LM-7B-beta":
                prompts_temp.append(get_prompt_string_starling(text, tokenizer, use_fewshot_prompting=use_fewshot_prompting, few_shot_examples=few_shot_examples))
            else:
                prompts_temp.append(get_prompt_string(text, tokenizer, use_fewshot_prompting=use_fewshot_prompting, few_shot_examples=few_shot_examples))
        prompts.append(prompts_temp)
    return prompts



def load_model_and_tokenizer(parameters):
    attn_impl = 'flash_attention_2' if torch.cuda.is_available() else 'sdpa'

    tokenizer = AutoTokenizer.from_pretrained(parameters['model_type'], model_max_length=parameters['max_input_length'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    if parameters['quantize']:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            parameters['model_type'],
            quantization_config=bnb_config,
            attn_implementation = attn_impl,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            parameters['model_type'],
            torch_dtype=torch.bfloat16,
            attn_implementation = attn_impl,
        )
    return model, tokenizer

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        torch.mps.empty_cache()

def save_triplets(path_save_triplets_file, results_list, intermediate=False):
    with open(path_save_triplets_file, 'a') as f:
        writer = csv.writer(f)
        for row in results_list:
            writer.writerow(row.values())
        results_list = []

    if intermediate:
        return results_list

def generate_triplets(inputs, model, tokenizer, path_save_triplets_file, device, terminators, parameters, general_logger):

    texts = inputs['text']
    prompts = inputs['prompts']
    paper_ids = inputs['paper_ids']
    gpu_id = inputs['gpu_id']

    num_files = len(prompts)
    count_too_long = 0
    count_memory_estimation_fails = 0

    general_logger.write_log(f'Maximum number of new tokens: {parameters["max_new_tokens"]}')

    with torch.inference_mode():
        results_list = []
        # check if the file exists, if not, create it
        if not path_save_triplets_file.exists():
            general_logger.write_log(f'The file {path_save_triplets_file} does not exist, creating it.')
            with open(path_save_triplets_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['text', 'triplets', 'paper_id'])
        else:
            general_logger.write_log(f'The file {path_save_triplets_file} already exists, appending to it.')
        
        #start time tracking
        time_start = time.time()

        for i, prompts_for_one_paper in enumerate(prompts):
            memory_estimate_failed = False

            if (i+1) % 25 == 0:
                general_logger.write_log(f'Processing file {i+1} of {num_files}')
            if (i+1) % 100 == 0:
                # take the time
                time_end = time.time()
                # get minutes
                general_logger.write_log(f'Time taken in minutes for the last 100 files: {(time_end - time_start) / 60}')
                time_start = time.time()

            max_input_length = parameters['max_input_length']
            if max(map(len, prompts_for_one_paper)) > max_input_length:
                count_too_long += 1
                general_logger.write_log(f'The max length of the input is {max(map(len, prompts_for_one_paper))}, which is larger than the max_input_length of {max_input_length}. The input will be truncated.', level=logging.WARNING)

            input_ids = tokenizer(prompts_for_one_paper, return_tensors='pt', padding='longest', truncation=True)
            num_tokens = input_ids['input_ids'].shape[1]

            # estimated_memory = estimate_memory(num_tokens, parameters['max_new_tokens'])
            # batch_size = int(find_total_usable_memory() / estimated_memory)
            # # if the batch_size is negative, set it to 8
            # if batch_size < 1:
            #     general_logger.write_log(f'The batch size is negative: {batch_size}. Setting it to 8.', level=logging.WARNING)
            #     batch_size = 8
            attention_mask = input_ids['attention_mask'].to(device)
            input_ids = input_ids['input_ids'].to(device)

            estimated_memory, r2_test = estimate_memory(parameters['path_memory_estimate'], num_tokens, parameters['max_new_tokens'])
            if r2_test:
                #logger.info(f"Estimated memory: {estimated_memory:.2f} GB")
                available_memory = find_total_usable_memory_device(gpu_id)
                batch_size = int(available_memory / estimated_memory)
            else:
                general_logger.write_log('The memory estimation did not pass the R2 test', level=logging.WARNING)
                batch_size = infer_best_batch_size_by_heuristics(parameters_count(model), find_total_usable_memory_device(gpu_id), num_tokens, parameters['max_new_tokens'])


            process_inputs = 0
            while process_inputs < len(input_ids):
                output, batch_size_final = generate_answer(model, input_ids, attention_mask, process_inputs, batch_size, terminators, parameters, gpu_id, general_logger)
                output = output.to('cpu')

                if batch_size_final != batch_size and not memory_estimate_failed:
                    count_memory_estimation_fails += 1
                    memory_estimate_failed = True

                if output is None:
                    process_inputs += batch_size_final
                    continue

                result = tokenizer.batch_decode(output, skip_special_tokens=True)
                # log the length
                match parameters['model_type']:
                    case "Nexusflow/Starling-LM-7B-beta":
                        extracted_triplets_list = [item.split('GPT4 Correct Assistant:')[-1] for item in result]
                    case "mistralai/Mistral-7B-Instruct-v0.2":
                        extracted_triplets_list = [item.split('[/INST]')[-1] for item in result]
                    case "google/gemma-2b-it":
                        extracted_triplets_list = [item.split('<start_of_turn>model')[-1] for item in result]
                    case "meta-llama/Meta-Llama-3-8B-Instruct":
                        extracted_triplets_list = [item.split('assistant')[-1] for item in result]
                    case "CohereForAI/c4ai-command-r-v01":
                        extracted_triplets_list = [item.split('<|CHATBOT_TOKEN|>')[-1] for item in result]
                    case _:
                        raise ValueError('Model id not recognized')

                # Now we loop over the individual texts
                for k, extracted_triplets in enumerate(extracted_triplets_list):
                    if isinstance(extracted_triplets, str):
                        results_list.append({'text': texts[i][k], 'triplets': extracted_triplets, 'paper_id': paper_ids[i]})

                process_inputs += batch_size_final
            
            # Only need to save intermediately when there are many files
            if num_files > 200 and i % 200 == 0:
                results_list = save_triplets(path_save_triplets_file, results_list, intermediate=True)
                general_logger.write_log(f'Saved intermediate results at file {i} of {num_files}')


    general_logger.write_log(f'Finished processing all files, saving the results at {path_save_triplets_file}')
    save_triplets(path_save_triplets_file, results_list, intermediate=False)
    return [num_files, count_too_long, count_memory_estimation_fails, gpu_id]

@torch.inference_mode()
def generate_answer(model, input_ids, attention_mask, process_input, batch_size, terminators, parameters, gpu_id, logger):
    is_oom = False
    batch = input_ids[process_input:process_input+batch_size]

    mask = attention_mask[process_input:process_input+batch_size]

    max_new_tokens = parameters['max_new_tokens']
    model_id = parameters['model_type']
    temperature = parameters['temperature']
    repetition_penalty = parameters['repetition_penalty']
    top_p = parameters['top_p']

    clear_memory()
    try:
        if parameters['use_sample_generation_strategy']:
            if model_id == "meta-llama/Meta-Llama-3-8B-Instruct":
                output = model.generate(input_ids=batch, max_new_tokens=max_new_tokens, attention_mask=mask, do_sample=True, temperature=temperature, eos_token_id=terminators, repetition_penalty=repetition_penalty, top_p = top_p)
            else:
                output = model.generate(input_ids=batch, max_new_tokens=max_new_tokens, attention_mask=mask, do_sample=True, temperature=temperature, repetition_penalty=repetition_penalty, top_p = top_p)

        else:
            if model_id == "meta-llama/Meta-Llama-3-8B-Instruct":
                output = model.generate(input_ids=batch, max_new_tokens=max_new_tokens, attention_mask=mask, eos_token_id=terminators)
            else:
                output = model.generate(input_ids=batch, max_new_tokens=max_new_tokens, attention_mask=mask)

    except RuntimeError as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            is_oom = True
            # log the complete error
            logger.write_log(f'OOM error at GPU: {gpu_id}, the error is: {e}', level=logging.ERROR)
            logger.write_log(torch.cuda.memory_summary())
            # show nvidia-smi
            os.system('nvidia-smi')
        else:
            raise e

    if is_oom:
        if batch_size == 1:
            if logger is not None:
                logger.write_log('Even a batch size of 1 causes an OOM. Skipping this input.')
            return None, 1
        new_batch_size = max(1, math.floor(batch_size*0.8))
        warn_text = f'Reducing batch size from {batch_size} to {new_batch_size} due to memory overflow (OOM). Input size here is {batch.shape[1]}.'
        logger.write_log(warn_text, level=logging.WARNING)
        
        clear_memory()
        return generate_answer(model, input_ids, attention_mask, process_input, new_batch_size, terminators, parameters, gpu_id, logger)
    else:
        clear_memory()
        return output, batch_size

def extract_triplets_one_gpu(batch_data, paper_ids, gpu_id, parameters, fewshot_examples, path_save_triplets_folder, queue):
    local_logger = logger(queue=queue, logging_level_file=parameters['logging_level_file'], logging_level_cmd=parameters['logging_level_cmd'], verbose=parameters['verbose'])


    # assert that torch only sees one GPU
    device = torch.device(f'cuda:{gpu_id}') # Important: since torch should only see one GPU, the ID is always 0
    local_logger.write_log(f'Process: {os.getpid()} is using GPU {gpu_id}, device: {device}')
    model, tokenizer = load_model_and_tokenizer(parameters)
    model.to(device)
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    if parameters['model_type'] == "meta-llama/Meta-Llama-3-8B-Instruct":
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    else:
        terminators = [tokenizer.eos_token_id]

    processed_text_files = prepare_test_data(batch_data, parameters['num_lines_per_model_call'], local_logger)
    local_logger.write_log(f'Number of processed text files on GPU {gpu_id}: {len(processed_text_files)}')
    prompts = get_prompts(processed_text_files, tokenizer, parameters['model_type'], fewshot_examples, parameters['use_fewshot_prompting'])
    local_logger.write_log(f'Number of papers to process on GPU {gpu_id}: {len(prompts)}')
    model.config.pad_token_id = tokenizer.pad_token_id
    local_logger.write_log(f'Available memory on GPU {gpu_id}: {find_total_usable_memory_device(gpu_id):.2f} GB')
    path_save_triplets_file_subprocess = path_save_triplets_folder.joinpath(f'triplets_{gpu_id}.csv')

    inputs = {
        'text': processed_text_files,
        'prompts': prompts,
        'paper_ids': paper_ids,
        'gpu_id': gpu_id,
    }

    assert len(inputs['text']) == len(inputs['prompts']), 'The number of texts and prompts must be the same'
    assert len(inputs['text']) == len(inputs['paper_ids']), 'The number of texts and paper_ids must be the same'

    counts = generate_triplets(inputs, model, tokenizer, path_save_triplets_file_subprocess, device, terminators, parameters, local_logger)
    return prompts, counts

def merge_results(results, logger, path_save_triplets_file, num_gpus):
    # paths to merge should be all the files in the save_triplets_folder
    # use rglob to get all the csv files
    paths_to_merge = [path for path in GENERAL_CONTEXT.path_save_triplets_folder.rglob('*.csv')]
    #paths_to_merge = [GENERAL_CONTEXT.path_save_triplets_folder.joinpath(f'triplets_{i}.csv') for i in range(num_gpus)]
    final_df = pd.DataFrame()
    for path in paths_to_merge:
        with open(path, 'r') as f:
            df = pd.read_csv(f)
        final_df = pd.concat([final_df, df], ignore_index=True)
    final_df.to_csv(path_save_triplets_file, index=False)

    for result in results:
        _, counts = result
        if counts[0] == 0:
            continue
        logger.write_log(f'GPU ID: {counts[3]}, total number of files: {counts[0]}, percentage of files that were too long: {counts[1] / counts[0] * 100:.2f}%, percentage of files where the memory estimation failed: {counts[2] / counts[0] * 100:.2f}%')

    # now remove intermediate files
    for i in range(num_gpus):
        path_save_triplets_file_subprocess = path_save_triplets_file.parent.joinpath(f'triplets_{i}.csv')
        path_save_triplets_file_subprocess.unlink()

    return results

def regression_test(time_taken):
    results_regression_test = {}
    path_regression_test_folder = GENERAL_CONTEXT.path_regression_test_folder
    results_regression_test['time_taken'] = time_taken
    
    #load triplets file
    with open(GENERAL_CONTEXT.path_save_triplets_file, 'r') as f:
        triplets = pd.read_csv(f)

    # get the number of unique paper_ids
    num_files = len(triplets['paper_id'].unique())

    results_regression_test['num_files'] = num_files
    results_regression_test['time_taken'] = time_taken
    # save as json
    with open(path_regression_test_folder.joinpath('extract_triplets_test_output.json'), 'w') as f:
        json.dump(results_regression_test, f)

def main():
    #################### SETTINGS ####################
    begin_time = time.time()
    run_regression_test = True if os.getenv("REGRESSION_TEST") == "1" else False
    global GENERAL_CONTEXT, TRIPLET_EXTRACTION_CONTEXT, num_cpus
    GENERAL_CONTEXT = GeneralContext()
    TRIPLET_EXTRACTION_CONTEXT = TripletExtractionContext(GENERAL_CONTEXT)
    num_cpus = int(os.getenv("NUM_CPUS_TRIPLET_EXTRACTION"))

    load_dotenv()
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    num_gpus = torch.cuda.device_count()
    path_save_triplets_folder = GENERAL_CONTEXT.path_save_triplets_folder

    if GENERAL_CONTEXT.path_log.joinpath('triplet_extraction.txt').exists():
        GENERAL_CONTEXT.path_log.joinpath('triplet_extraction.txt').unlink()
    general_logger = logger('triplet_extraction', GENERAL_CONTEXT.path_log.joinpath('triplet_extraction.txt'), GENERAL_CONTEXT.logging_level_file, GENERAL_CONTEXT.logging_level_cmd, verbose=GENERAL_CONTEXT.verbose)
    path_save_triplets_file = GENERAL_CONTEXT.path_save_triplets_file

    if path_save_triplets_file.exists():
        general_logger.write_log(f'The file {path_save_triplets_file} already exists. Stop the script to avoid overwriting the file.')
        return
    
    if run_regression_test:
        general_logger.write_log('This is a regression test run')

    general_logger.write_log(os.environ["PYTORCH_CUDA_ALLOC_CONF"])

    general_logger.write_log(f'Number of GPUs: {num_gpus}')

    # check if there is something in path_triplets_folder
    if len(list(path_save_triplets_folder.rglob('*.csv'))) > 0:
        general_logger.write_log(f'There are already results saved, specifically {len(list(path_save_triplets_folder.rglob("*.csv")))} files. The script will continue from where it left off.')
        #load the csv files, then combine them, we have the columns: text, triplets, paper_id
        existing_triplets = pd.concat([pd.read_csv(file) for file in path_save_triplets_folder.rglob('*.csv')], ignore_index=True)
        general_logger.write_log('There are already results saved, the script will continue from where it left off.')
    else:
        existing_triplets = None

    # open the fewshot examples
    with open(TRIPLET_EXTRACTION_CONTEXT.path_fewshot_examples, 'r') as f:
        few_shot_examples = json.load(f)

    ##################################################
    max_input_length = TRIPLET_EXTRACTION_CONTEXT.max_input_length
    general_logger.write_log(f'Max input length: {max_input_length}, everything after will be truncated.')

    processed_text_files, paper_ids = load_test_data_folder_text_files(GENERAL_CONTEXT.path_save_processed_texts, existing_triplets, general_logger)
    splits, paper_ids_splits = get_data_splits(processed_text_files, paper_ids, num_gpus)

    parameters = {
        'model_type': TRIPLET_EXTRACTION_CONTEXT.model_type,
        'use_fixed_input_length': TRIPLET_EXTRACTION_CONTEXT.use_fixed_input_length,
        'num_lines_per_model_call': TRIPLET_EXTRACTION_CONTEXT.num_lines_per_model_call,
        'fixed_input_length': TRIPLET_EXTRACTION_CONTEXT.fixed_input_length,
        'use_fewshot_prompting': TRIPLET_EXTRACTION_CONTEXT.use_fewshot_prompting,
        'quantize': TRIPLET_EXTRACTION_CONTEXT.quantize,
        'use_sample_generation_strategy': TRIPLET_EXTRACTION_CONTEXT.use_sample_generation_strategy,
        'repetition_penalty': TRIPLET_EXTRACTION_CONTEXT.repetition_penalty,
        'temperature': TRIPLET_EXTRACTION_CONTEXT.temperature,
        'top_p': TRIPLET_EXTRACTION_CONTEXT.top_p,
        'logging_level_file': GENERAL_CONTEXT.logging_level_file,
        'logging_level_cmd': GENERAL_CONTEXT.logging_level_cmd,
        'max_input_length': max_input_length,
        'max_new_tokens': TRIPLET_EXTRACTION_CONTEXT.max_new_tokens,
        'path_memory_estimate': TRIPLET_EXTRACTION_CONTEXT.path_memory_estimate,
        'verbose': GENERAL_CONTEXT.verbose
    }

    # Log the memories before starting the process
    for i in range(num_gpus):
        general_logger.write_log(f'Available memory on GPU {i}: {find_total_usable_memory_device(i):.2f} GB')

    with mp.Manager() as manager:
        queue = manager.Queue()
        listener = mp.Process(target=listener_process, args=(queue, listener_configurer, GENERAL_CONTEXT.path_log.joinpath('extract_triplets.txt'), GENERAL_CONTEXT.logging_level_file, GENERAL_CONTEXT.logging_level_cmd))
        listener.start()

        pool = mp.Pool(num_gpus)
        results = [pool.apply_async(extract_triplets_one_gpu, args=(split, paper_ids_splits[i], i, parameters, few_shot_examples, path_save_triplets_folder, queue)) for i, split in enumerate(splits)]
        pool.close()
        pool.join()

        # get the results
        results = [r.get() for r in results]

        queue.put(None)
        listener.join()

    # merge the results
    merge_results(results, general_logger, path_save_triplets_file, num_gpus)
    general_logger.write_log('Finished processing all files, saved the results.')
    end_time = time.time()
    if run_regression_test:
        regression_test(end_time-begin_time)



if __name__ == '__main__':
    main()
