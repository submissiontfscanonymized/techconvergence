'''
This code is obtained from https://github.com/lettergram/parse-uspto-xml
It is adapted and simplified to serve our needs 
'''

from __future__ import annotations

import datetime
import html
import json
import os
import re
import sys
import logging
from typing import Union, Callable
from pathlib import Path
import numpy as np

from bs4 import BeautifulSoup
import psycopg2.extras
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

from helpers import logger, listener_configurer, listener_process
from contexts import ParsePatentsContexts, GeneralContext



def get_filenames_from_dir(dirpaths: Path | list[Path]) -> list[Path]:
    """Get filenames from directory, the directory is a Path"""

    if isinstance(dirpaths, Path):
        dirpaths = [dirpaths]

    filenames = []
    for dirpath in dirpaths:
        for xml_file in dirpath.rglob("*.xml"):
            filenames.append(xml_file)
    return filenames

def parse_uspto_file(bs, keep_log: bool = False, local_logger: logger = None):
    """
    Parses a USPTO patent in a BeautifulSoup object.
    """

    patent_office = "uspto"
    grant_date = None

    try:
        publication_num = bs['file'].split("-")[0]
    except KeyError:
        raise KeyError("No file attribute found in the BeautifulSoup object.")
    application_status = "pending"
    if bs.name == ('us-patent-grant'):
        grant_date = bs.get("date-produced", None)
        application_status = "granted"

    publication_title = bs.find('invention-title').text
    publication_date = bs.find('publication-reference').find('date').text
    application_ref_bs = bs.find('application-reference')

    try:
        application_type = application_ref_bs['appl-type']
    except KeyError:
        raise KeyError("No application type found in the BeautifulSoup object.")

    application_date = application_ref_bs.find('date').text
    application_number = application_ref_bs.find('doc-number').text

    referential_documents = []
    # {uspto_patents.publication_number,reference,cited_by_examiner,document_type,country,metadata (JSON)

    related_docs_bs = bs.find("us-related-documents")
    for related_doc_bs in (related_docs_bs.find_all(recursive=False) if related_docs_bs else []):
        related_doc = {
            "publication_number": publication_num,
            "patent_office": patent_office,
            "application_number": application_number,
            "reference": None,
            "cited_by_examiner": None,
            "document_type": None,
            "country": None,
            "kind": None,
            "metadata": {}
        }
        if related_doc_bs.name in ["continuation", "division", "continuation-in-part", "reissue", "substitution", "us-reexamination-reissue-merger", "continuing-reissue"]:
            document_type = related_doc_bs.name
            if document_type in ["us-reexamination-reissue-merger", "continuing-reissue"]:
                document_type = "reissue"
            related_doc["document_type"] = document_type
            related_doc["cited_by_examiner"] = False
            for documents_bs in related_doc_bs.find_all(re.compile("(parent|child)-doc(ument)?$")):
                for doc_bs in documents_bs.find_all("document-id"):
                    if doc_bs.parent.name == "parent-grant-document":
                        related_doc["reference"] = doc_bs.find("doc-number").text
                    elif doc_bs.parent.name == "parent-pct-document":
                        related_doc["metadata"]["parent_pct_number"] = doc_bs.find("doc-number").text
                        related_doc["metadata"]["parent_pct_country"] = doc_bs.find("country").text
                        related_doc["metadata"]["parent_pct_date"] = getattr(doc_bs.find("date"), "text", None)
                    elif doc_bs.parent.name == "parent-doc":
                        related_doc["country"] = doc_bs.find("country").text
                        related_doc["metadata"]["application_number"] = doc_bs.find("doc-number").text
                        related_doc["metadata"]["application_date"] = getattr(doc_bs.find("date"), "text", None)
                    elif doc_bs.parent.name == "child-doc":
                        related_doc["metadata"]["child_application_number"] = doc_bs.find("doc-number").text
                        related_doc["metadata"]["parent_country"] = doc_bs.find("country").text
        elif related_doc_bs.name in ["us-provisional-application"]:
            related_doc["document_type"] = "provisional"
            related_doc["cited_by_examiner"] = False
            related_doc["country"] = related_doc_bs.find("country").text
            related_doc["reference"] = related_doc_bs.find("doc-number").text
            related_doc["metadata"]["application_date"] = related_doc_bs.find("date").text
        elif related_doc_bs.name in ["related-publication"]:
            related_doc["document_type"] = "prior"
            related_doc["cited_by_examiner"] = False
            related_doc["reference"] = related_doc_bs.find("doc-number").text
            related_doc["country"] = related_doc_bs.find("country").text
            related_doc["kind"] = related_doc_bs.find("kind").text
            related_doc["metadata"]["date"] = related_doc_bs.find("date").text
        else:
            raise KeyError(f"'{related_doc_bs.name}' is not setup to be included in referential documents.")
        referential_documents.append(related_doc)

    references = []
    refs_cited_bs = bs.find(re.compile(".*-references-cited"))
    if refs_cited_bs:
        for ref_bs in refs_cited_bs.find_all(re.compile(".*-citation")):
            doc_bs = ref_bs.find("document-id")
            if doc_bs:
                reference = {
                    "publication_number": publication_num,
                    "patent_office": patent_office,
                    "application_number": application_number,
                    "reference": doc_bs.find("doc-number").text,
                    "cited_by_examiner": "examiner" in ref_bs.find("category").text,
                    "document_type": "patent-reference",
                    "country": getattr(doc_bs.find("country"), "text", None),
                    "kind": getattr(doc_bs.find("kind"), "text", None),
                    "metadata":{
                        "name": getattr(doc_bs.find("name"), "text", None),
                        "date": getattr(doc_bs.find("date"), "text", None),
                    }
                }
            else:
                reference = {
                    "publication_number": publication_num,
                    "patent_office": patent_office,
                    "application_number": application_number,
                    "reference": ref_bs.find("othercit").text,
                    "cited_by_examiner": "examiner" in ref_bs.find("category").text,
                    "document_type": "other-reference",
                    "country": getattr(ref_bs.find("country"), "text", None),
                    "kind": None,
                    "metadata": {},
                }
            references.append(reference)
        referential_documents += references

    priority_claims = []
    priority_docs_bs = bs.find("priority-claims")
    if priority_docs_bs:
        for doc_bs in priority_docs_bs.find_all("priority-claim"):
            priority_claims.append({
                "publication_number": publication_num,
                "patent_office": patent_office,
                "application_number": application_number,
                "reference": doc_bs.find("doc-number").text,
                "cited_by_examiner": False,
                "document_type": "other-reference",
                "country": getattr(doc_bs.find("country"), "text", None),
                "kind": None,
                "metadata":{
                    "date": getattr(doc_bs.find("date"), "text", None),
                },
            })
        referential_documents += priority_claims

    # check to make sure all keys are proper -- TODO: this should be a test.
    for reference in referential_documents:
        expected_keys = {
            "publication_number",
            "patent_office",
            "application_number",
            "reference",
            "cited_by_examiner",
            "document_type",
            "country",
            "kind",
            "metadata",
        }
        missing_keys = expected_keys - set(reference.keys())
        bad_keys =  set(reference.keys()) - expected_keys
        if missing_keys or bad_keys:
            raise KeyError(
                f"referential_documents has missing_keys: "
                f"{missing_keys} and bad_keys: {bad_keys} "
                f"for {reference}"
            )

    # International Patent Classification (IPC) Docs:
    # https://www.wipo.int/classifications/ipc/en/
    sections = {}
    section_classes = {}
    section_class_subclasses = {}
    section_class_subclass_groups = {}
    for classes in bs.find_all('classifications-ipcr'):
        for el in classes.find_all('classification-ipcr'):

            section = el.find('section').text

            classification  = section
            classification += el.find('class').text
            classification += el.find('subclass').text

            group = el.find('main-group').text + "/"
            group += el.find('subgroup').text

            sections[section] = True
            section_classes[section+el.find('class').text] = True
            section_class_subclasses[classification] = True
            section_class_subclass_groups[classification+" "+group] = True

    if not sections:
        re_classification = re.compile(
            "(?P<section>[A-Z])"
            + "(?P<class>[0-9]{2})"
            + "(?P<subclass>[A-Z])"
            + "\s?(?P<maingroup>[0-9]{1,4})"
            + "\s?/\s?"
            + "(?P<subgroup>[0-9]{2,6})"
        )
        re_classification_tag = re.compile(
            "(classification-ipc(r)?)|(classification-cpc(-text)?)"
        )
        for classes in bs.find_all(re.compile("us-bibliographic-data-(grant|application)")):
            for el in classes.find_all(re_classification_tag):
                if "citation" in el.parent.name:
                    continue  # skip anything that's not the patent itself
                classification = getattr(el.find('main-classification'), "text", el.text)
                re_value = re_classification.match(classification)
                if re_value is not None:
                    section = re_value.group("section")
                    section_class = section + re_value.group("class")
                    section_subclass = section_class + re_value.group("subclass")

                    group = re_value.group("maingroup") + "/" + re_value.group("subgroup")

                    sections[section] = True
                    section_classes[section_class] = True
                    section_class_subclasses[section_subclass] = True
                    section_class_subclass_groups[section_subclass + " " + group] = True

    def build_name(bs_el):
        """Creates a name '<First> <Last>'"""
        # [First Name, Last Name]
        name_builder = []
        for attr_name in ["first-name", "last-name"]:
            value = getattr(bs_el.find(attr_name), "text", "")
            if value and value != "unknown":
                name_builder.append(value)
        name = ""
        if name_builder:
            name = " ".join(name_builder).strip()
        return name

    def build_org(bs_el):
        """Creates an organization '<org>, <city>, <country>'"""
        # org_builder: [organization, city, country]
        org_builder = []
        for attr_name in ["orgname", "city", "country"]:
            value = getattr(bs_el.find(attr_name), "text", "")
            if value and value != "unknown":
                org_builder.append(value)
        org_name = ""
        if org_builder:
            org_name = ", ".join(org_builder).strip()
        return org_name

    authors = []
    organizations = []
    attorneys = []
    attorney_organizations = []
    for parties in bs.find_all(re.compile('^.*parties')):
        for inventors in parties.find_all(re.compile('inventors|applicants')):
            for el in inventors.find_all('addressbook'):
                # inventor_name: " ".join([first, last])
                inventor_name = build_name(el)
                if inventor_name:
                    authors.append(inventor_name)

        for applicants in parties.find_all(re.compile('^.*applicants')):
            for el in applicants.find_all('addressbook'):
                # org_name: ", ".join([organization, city, country])
                org_name = build_org(el)
                if org_name:
                    organizations.append(org_name)

        for agents in parties.find_all(re.compile('^.*agents')):
            for agent in agents.find_all("agent", attrs={"rep-type": "attorney"}):
                for el in agent.find_all("addressbook"):
                    # attorney_name: " ".join([first, last])
                    attorney_name = build_name(el)
                    if attorney_name:
                        attorneys.append(attorney_name)

                    # org_name: ", ".join([organization, city, country])
                    org_name = build_org(el)
                    if org_name:
                        attorney_organizations.append(org_name)

    abstracts = []
    for el in bs.find_all('abstract'):
        abstracts.append(el.text.strip('\n'))

    descriptions = []
    for el in bs.find_all('description'):
        descriptions.append(el.text.strip('\n'))

    claims = []
    for el in bs.find_all('claim'):
        claims.append(el.text.strip('\n'))

    uspto_patent = {
        "publication_title": publication_title,
        "publication_number": publication_num,
        "publication_date": publication_date,
        "grant_date": grant_date,
        "application_number": application_number,
        "application_type": application_type,
        "application_date": application_date,
        "application_status": application_status,
        "patent_office": patent_office,
        "authors": authors, # list
        "organizations": organizations, # list
        "attorneys": attorneys, # list
        "attorney_organizations": attorney_organizations, # list
        "referential_documents": referential_documents,
        "sections": list(sections.keys()),
        "section_classes": list(section_classes.keys()),
        "section_class_subclasses": list(section_class_subclasses.keys()),
        "section_class_subclass_groups": list(section_class_subclass_groups.keys()),
        "abstract": abstracts, # list
        "descriptions": descriptions, # list
        "claims": claims # list
    }

    if keep_log:

        local_logger.write_log("Filename:", bs['file'])
        local_logger.write_log("\n\n")
        local_logger.write_log("\n--------------------------------------------------------\n")

        local_logger.write_log("USPTO Invention Title:", publication_title)
        local_logger.write_log("USPTO Publication Number:", publication_num)
        local_logger.write_log("USPTO Publication Date:", publication_date)
        local_logger.write_log("USPTO Application Type:", application_type)

        count = 1
        for classification in section_class_subclass_groups:
            local_logger.write_log("USPTO Classification #"+str(count)+": " + classification)
            count += 1
        local_logger.write_log("\n")

        count = 1
        for author in authors:
            local_logger.write_log("Inventor #"+str(count)+": " + author)
            count += 1

        count = 1
        for org in organizations:
            local_logger.write_log("Organization #"+str(count)+": " + org)
            count += 1

        count = 1
        for attorney in attorneys:
            local_logger.write_log("Attorney #"+str(count)+": " + attorney)
            count += 1

        count = 1
        for org in attorney_organizations:
            local_logger.write_log("Attorney Organization #"+str(count)+": " + org)
            count += 1

        local_logger.write_log("\n--------------------------------------------------------\n")

        local_logger.write_log("Abstract:\n-----------------------------------------------")
        for abstract in abstracts:
            local_logger.write_log(abstract)

        local_logger.write_log("Description:\n-----------------------------------------------")
        for description in descriptions:
            local_logger.write_log(description)

        local_logger.write_log("Claims:\n-----------------------------------------------")
        for claim in claims:
            local_logger.write_log(claim)

    return uspto_patent

def load_batch_from_data(
        xml_text_list: list[str],
        keep_log: bool = False,
        local_logger: logger = None
    ):

    count = 0
    success_count = 0
    errors = []
    patent_list = []

    for patent in xml_text_list:

        if patent is None or patent == "":
            continue

        bs = BeautifulSoup(patent, "lxml")

        if bs.find('sequence-cwu') is not None:
            continue # Skip DNA sequence documents

        application = bs.find('us-patent-application')
        if application is None: # If no application, search for grant
            application = bs.find('us-patent-grant')
        title = "None"

        if application is None:
            local_logger.write_log(f"Error at {count}: No application or grant found.", level=logging.ERROR)
            continue

        try:
            title = application.find('invention-title').text
        except Exception as e:
            local_logger.write_log(f"Error at {count}: {str(e)}", level=logging.ERROR)

        try:
            uspto_patent = parse_uspto_file(
                bs=application,
                keep_log=keep_log,
                local_logger=local_logger
            )
            patent_list.append(uspto_patent)
            success_count += 1
        except Exception as e:
            exception_tuple = (count, title, e)
            errors.append(exception_tuple)
            local_logger.write_log(f"Error: {exception_tuple}", level=logging.ERROR)
        count += 1

    return count, success_count, patent_list, errors

def parse_batch(xml_splits, local_logger, filename, jsonl_filename, parameters):
    keep_log = parameters["keep_log"]
    batch_size = parameters["batch_size"]

    count = 0
    success_count = 0
    errors = []
    total_patents = len(xml_splits)
    local_logger.write_log(f"Total patents in {filename}: {total_patents}")
    for i in range(0, len(xml_splits), batch_size):
        local_logger.write_log(f"Processing split {i} to {i + batch_size} of {total_patents}")
        last_index = i + batch_size
        xml_batch = xml_splits[i : last_index]
        batch_count, batch_success_count, patents, batch_errors = \
            load_batch_from_data(xml_batch, keep_log, local_logger)
        count += batch_count

        recent_title = None
        if len(patents):
            recent_title = patents[0].get("publication_title")
        
        try:
            push_to_jsonl(patents, jsonl_filename)
        except Exception as e:
            exception_tuple = (count, recent_title, e)
            errors.append(exception_tuple)
            local_logger.write_log(f"Error: {exception_tuple}", level=logging.ERROR)
            batch_success_count = 0

        local_logger.write_log(f"{count} of {total_patents}, {filename}, {recent_title}")

        success_count += batch_success_count
        errors += batch_errors
    return count, success_count, errors

def parse_patent_subprocess(
        filename_chunk: list[str],
        queue: mp.Queue,
        parameters: dict,
    ):

    logging_level_file = parameters["logging_level_file"]
    logging_level_cmd = parameters["logging_level_cmd"]
    local_logger = logger(queue=queue, logging_level_file=logging_level_file, logging_level_cmd=logging_level_cmd)

    count = 0
    success_count = 0
    errors = []
    local_logger.write_log(f"Subprocess {os.getpid()}: processing {len(filename_chunk)} patents.")
    
    for i, filename in enumerate(filename_chunk):
        jsonl_filename = Path(filename).name
        jsonl_filename = Path(parameters['folder_to_save']).joinpath(jsonl_filename).with_suffix(".jsonl")
        if jsonl_filename.exists():
            local_logger.write_log(f"Skipping {i+1} of {len(filename_chunk)}: {filename} already exists.")
            continue

        local_logger.write_log(f"Processing {i+1} of {len(filename_chunk)}: {filename}")
        filename = filename_chunk[i]
        with open(filename, "r") as fp:
            xml_text = html.unescape(fp.read())
        xml_splits = xml_text.split("<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
        # if there is at least 1 patent which is not an empty string
        if len(xml_splits) and not xml_splits[0]:
            xml_splits = xml_splits[1:]
        
        local_logger.write_log(f"Subprocess {os.getpid()}: {len(xml_splits)} patents found in {filename}")

        count, success_count, errors = parse_batch(xml_splits, local_logger, filename, jsonl_filename, parameters)

        local_logger.write_log(f"Subprocess {os.getpid()}: {success_count} of {count} patents successfully parsed from {filename}")
        local_logger.write_log(f"Saving to: {jsonl_filename}")
        
    return count, success_count, errors

def parse_patents(
        dirpath_list:  list,
        batch_size: int = 50,
        num_cpus: int = 5,
        keep_log: bool = False,
        folder_to_save: str = "parsed_patents",
        path_log: str = "logs",
):
    """Load all files from local directory"""
    general_logger.write_log("LOADING FILES TO PARSE\n----------------------------")
    filenames = get_filenames_from_dir(dirpath_list)
    general_logger.write_log(f"Total number of XML files: {len(filenames)}")

    parameters = {
        "keep_log": keep_log,
        "logging_level_file": parse_patents_context.logging_level_file,
        "logging_level_cmd": parse_patents_context.logging_level_cmd,
        'batch_size': batch_size,
        'num_cpus': num_cpus,
        'folder_to_save': folder_to_save,
        'path_log': path_log
    }

    general_logger.write_log("Checking the suffix of the files and reading them.")
    # remove the files that are not xml
    filenames = [f for f in filenames if f.suffix == ".xml"]

    general_logger.write_log("Now starting the subprocesses to parse the files.")
    # split into num_cpus chunks
    filename_chunks = []
    chunk_size = max(len(filenames) // num_cpus, 1)
    for i in range(0, len(filenames), chunk_size):
        last_index = i + chunk_size
        if last_index > len(filenames):
            last_index = len(filenames)
        filename_chunks.append(filenames[i : last_index])

    with mp.Manager() as manager:
        queue = manager.Queue()
        listener = mp.Process(target=listener_process, args=(queue, listener_configurer, path_log, parse_patents_context.logging_level_file, parse_patents_context.logging_level_cmd))
        listener.start()
        res = []
        pool = mp.Pool(num_cpus)
        for i, filename_chunk in enumerate(filename_chunks):
            res.append(pool.apply_async(parse_patent_subprocess, args=(filename_chunk, queue, parameters)))

        pool.close()
        pool.join()

        count = 0
        success_count = 0
        errors = []

        for r in res:
            batch_count, batch_success_count, batch_errors = r.get()
            count += batch_count
            success_count += batch_success_count
            errors += batch_errors

        queue.put(None)
        listener.join()

    if errors:
        general_logger.write_log("\n\nErrors\n------------------------\n", level=logging.ERROR)
        for e in errors:
            general_logger.write_log(str(e), level=logging.ERROR)
    general_logger.write_log("=" * 50)
    general_logger.write_log("=" * 50)
    general_logger.write_log(f"Success Count: {success_count}")
    general_logger.write_log(f"Error Count: {count - success_count}")


def push_to_jsonl(patents: list[dict], push_to: str):
    patent_dumps_list = []
    for uspto_patent in patents:
        patent_dumps_list.append(json.dumps(uspto_patent) + "\n")
    with open(push_to, "a") as fp:
        fp.writelines(patent_dumps_list)

def main():
    global parse_patents_context, general_logger
    general_context = GeneralContext()
    parse_patents_context = ParsePatentsContexts(general_context)
    general_logger = logger('parse_patents', parse_patents_context.path_log.joinpath("parse_patents.log"), logging_level_file=parse_patents_context.logging_level_file, logging_level_cmd=parse_patents_context.logging_level_cmd)
    folder_to_save = parse_patents_context.save_json_patents    

    general_logger.write_log("Starting to parse USPTO XML files.")
    folder_xml_patents = parse_patents_context.path_xml_patents
 
    general_logger.write_log(f"Loading files from: {folder_xml_patents}")
    parse_patents(
        dirpath_list=folder_xml_patents,
        batch_size=50,
        num_cpus= int(os.getenv("NUM_CPUS_PATENT_PARSING")),
        keep_log=False,
        folder_to_save=folder_to_save,
        path_log = parse_patents_context.path_log.joinpath("parse_patents.log")
    )

if __name__ == "__main__":
    main()