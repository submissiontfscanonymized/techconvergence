
import json
import os
import pandas as pd
from pathlib import Path
from helpers import logger, listener_configurer, listener_process
import multiprocessing as mp
from contexts import GeneralContext, ParsePatentsContexts
from collections import defaultdict
import json

mp.set_start_method('spawn', force=True)


    
# # Now make a dictionary, with as the key the publication_number, and as the values all columns except the 'abstract_title_claims', 'publication_title', 'abstract', 'claims', 'descriptions'
# dict_attributes = defaultdict(lambda x: {})
# for idx, row in patent_files.iterrows():
#     dict_attributes[row['publication_number']] = {key: row[key] for key in row.keys() if key not in ['abstract_title_claims', 'publication_title', 'abstract', 'claims', 'descriptions']}

def text_contains_keywords(text, keywords):
    
    if text is None:
        return False
    text_lower = text.lower()
    keywords_lower = [keyword.lower() for keyword in keywords]
    if any(keyword in text_lower for keyword in keywords_lower):
        return True
    return False

def filter_by_keywords(patent_dict, keywords, local_logger):
    if 'abstract' in patent_dict:
        patent_dict['abstract'] = ' '.join(abst for abst in patent_dict['abstract'])
        assert type(patent_dict['abstract']) == str, f'abstract is not a string: {patent_dict["abstract"]}'
    else:
        patent_dict['abstract'] = ''
    if 'claims' in patent_dict:
        patent_dict['claims'] = ' '.join(claims for claims in patent_dict['claims'])
        assert type(patent_dict['claims']) == str, f'claims is not a string: {patent_dict["claims"]}'
    else:
        patent_dict['claims'] = ''

    assert type(patent_dict['publication_title']) == str, f'publication_title is not a string: {patent_dict["publication_title"]}'

    # add a column 'abstract_title_claims' that contains the concatenation of all text fields (publication_title, abstract, claims, descriptions), if they exist, if they don't, fill with empty string
    patent_dict['abstract_title_claims'] = patent_dict['publication_title'] + ' ' + patent_dict['abstract'] + ' ' + patent_dict['claims']

    if len(keywords) > 0:
        if not text_contains_keywords(patent_dict['abstract_title_claims'], keywords):
            return {}
    
    return patent_dict

def load_parsed_patents_subprocess(filename_chunk, queue, parameters):
    local_logger = logger(queue=queue, logging_level_file=parameters['logging_level_file'], logging_level_cmd=parameters['logging_level_cmd'])
    res = {}
    suc_count = 0
    fail_count = 0
    for idx, filename in enumerate(filename_chunk):
        local_logger.write_log(f'Processing file number {idx + 1} out of {len(filename_chunk)}')
        period = filename.stem[3:7]
        total_num_lines = 0
        with open(filename, 'r') as f:
            for line in f:
                total_num_lines += 1
        with open(filename, 'r') as f:
            idy = 0
            for line in f:
                idy += 1
                if idy % 100 == 0:
                    local_logger.write_log(f'Processed {idy} lines out of {total_num_lines} in file number {idx + 1} out of {len(filename_chunk)}')
                    local_logger.write_log(f'Current number of patents in the result dictionary: {len(res)}')
                try:
                    parsed_line = json.loads(line)
                except Exception as e:
                    local_logger.write_log(f'Error parsing a line in file {filename}: {e}')
                    fail_count += 1
                    continue
                suc_count += 1
                if parsed_line['publication_number'] in parameters['existing_files']:
                    continue

                filtered_dict = filter_by_keywords(parsed_line, parameters['keywords'], local_logger)
                if len(filtered_dict) == 0:
                    continue
                filtered_dict['descriptions_concatenated'] = '\n'.join(filtered_dict['descriptions'])
                filtered_dict['full_text'] = filtered_dict['abstract_title_claims'] + ' ' + filtered_dict['descriptions_concatenated']
                filtered_dict['referential_documents'] = [inner_dict['publication_number'] for inner_dict in filtered_dict['referential_documents']]
                subdirectory = parameters['path_save_patent_text'].joinpath(period)
                with open(subdirectory.joinpath(f'{filtered_dict["publication_number"]}.txt'), 'w') as f:
                    f.write(filtered_dict['full_text'])
                #take he values that are in parameters['keys_attribute_dict'], the key will be the publication_number
                res[filtered_dict['publication_number']] = {key: filtered_dict[key] for key in parameters['keys_attribute_dict']}

        if idx % 1 == 0:
            local_logger.write_log(f'Processed {idx + 1} files out of {len(filename_chunk)}')
    return res, suc_count, fail_count

def load_parsed_patents(folder: Path, existing_files, attributes_dict, path_log) -> pd.DataFrame:
    parsed_patents = []
    total_num_files = len(list(folder.rglob('*.jsonl')))
    num_files_per_process = max(1, total_num_files // num_processes)
    files_per_process = []
    all_files = list(folder.rglob('*.jsonl'))

    # Now we want to make subdirectories
    for filename in all_files:
        # find the first 4 digits of the filename, they have to be numbers, so for a filename ipg20010101.jsonl, we want to get 2001
        year = filename.stem[3:7]
        # make a subdirectory with this date
        subdirectory_name = parse_patent_context.save_raw_patent_text.joinpath(year)
        if not subdirectory_name.exists():
            subdirectory_name.mkdir()

    for i in range(0, len(all_files), num_files_per_process):
        files_per_process.append(all_files[i:i + num_files_per_process])
    
    parameters = {
        "logging_level_file": parse_patent_context.logging_level_file,
        "logging_level_cmd": parse_patent_context.logging_level_cmd,
        'keywords': parse_patent_context.keywords_for_patent_filtering,
        'path_save_patent_text': parse_patent_context.save_raw_patent_text,
        'keys_attribute_dict': parse_patent_context.keys_attribute_dict,
        'existing_files': existing_files,
    }

    general_logger.write_log(f'Filtering for the keywords: {parameters["keywords"]}')

    with mp.Manager() as manager:
        queue = manager.Queue()
        listener = mp.Process(target=listener_process, args=(queue, listener_configurer, path_log, parse_patent_context.logging_level_file, parse_patent_context.logging_level_cmd))
        listener.start()
        res = []
        pool = mp.Pool(num_processes)
        for filename_chunk in files_per_process:
            res.append(pool.apply_async(load_parsed_patents_subprocess, args=(filename_chunk, queue, parameters)))
        pool.close()
        pool.join()

        if attributes_dict is None:
            attributes_dict = {}
            suc_count = 0
            fail_count = 0
            for r in res:
                res_chunk, suc_count_chunk, fail_count_chunk = r.get()
                suc_count += suc_count_chunk
                fail_count += fail_count_chunk
                # assert that all the keys from res_chunk are not yet in attributes_dict
                for key in res_chunk.keys():
                    assert key not in attributes_dict, f'Key {key} is already in attributes_dict'
                    attributes_dict[key] = res_chunk[key]

            # save as json
            with open(parse_patent_context.path_save_patent_attributes.joinpath('patent_attributes.json'), 'w') as f:
                json.dump(attributes_dict, f)

            if suc_count + fail_count == 0:
                general_logger.write_log(f'No patents were processed, exiting...')
            else:
                general_logger.write_log(f'Finished processing patents, succes percentage: {suc_count / (suc_count + fail_count)}, total succes: {suc_count}, total fail: {fail_count}')
        else:
            assert isinstance(attributes_dict, dict), f'attributes_dict is not a dictionary: {attributes_dict}'
            general_logger.write_log(f'Attributes dict already existed, not saving it again, exiting now')


        queue.put(None)
        listener.join()

    return pd.DataFrame(parsed_patents)
        


def main():
    global general_context, parse_patent_context, general_logger, num_processes
    general_context = GeneralContext()
    parse_patent_context = ParsePatentsContexts(general_context)
    path_parsed_patents = parse_patent_context.save_json_patents
    path_to_save = parse_patent_context.save_raw_patent_text
    path_log = parse_patent_context.path_log

    # Find all files in the path_to_save folder
    existing_files = [f for f in path_to_save.rglob('*.txt')]

    patent_attributes = None
    if parse_patent_context.path_save_patent_attributes.joinpath('patent_attributes.json').exists():
        patent_attributes = json.load(open(parse_patent_context.path_save_patent_attributes.joinpath('patent_attributes.json'), 'r'))
    
    num_processes = int(os.getenv("NUM_CPUS_PATENT_PARSING"))

    path_log = Path(path_log.joinpath('process_patents.txt'))
    general_logger = logger('process_patents', path_log, logging_level_file = general_context.logging_level_file, logging_level_cmd = general_context.logging_level_cmd)

    general_logger.write_log(f'Number of existing files: {len(existing_files)}')

    general_logger.write_log('Loading parsed patents...')
    parsed_patents = load_parsed_patents(path_parsed_patents, existing_files, patent_attributes, path_log)
    general_logger.write_log(f'Columns of the parsed patents: {parsed_patents.columns}')

    general_logger.write_log(f'Finished! Raw patent text saved at {path_to_save}')

if __name__ == '__main__':
    main()