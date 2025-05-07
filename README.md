## LLM-Resilient Bibliometrics: Factual Consistency Through Entity Triplet Extraction
TO BE UPDATED
**Last update: ...**

This github repository provides the code that belongs to the project "Monitoring Transformative Technological Convergence Through LLM-Extracted Semantic Entity Triple Graph". The code provides the full pipeline from raw arXiv pdf's or raw patents to processed entity triplets of the shape (subject, predicate, object). The triplets are extracted through a LLM.

### Structure :books:

-- **src** \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *analyses* \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- TBD \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *configs* \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```config_template.yaml```: In this file, you specify all the settings for the triplet extraction and processing. For more information and an overview of all parameters see the folder ```code_structure/parameters.pdf```  \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *extraction_pipeline* \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```contexts.py```: Here, the contexts classes are defined based on the user settings in the configuration file.  \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```helpers.py```: This file provides several helper functions, such as loggers, that are used throughout the pipeline. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```estimation.py```: This file provides memory estimations for specific input sizes, such that the batch size can be set dynamically. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```calculate_entropy.py```: This file provides the code for calculating the cross-categorical entropy, used for filtering the triplets. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```load_data.py```: This file loads the pdf's and converts them to text files \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```preprocessing.py```: This file preprocesses the text files \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```extract_triplets_llm.py```: This file extracts the triplets from the text files using a LLM \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```process_triplets.py```: This file processes the triplets \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```filter_triplets.py```: This file filters the triplets, after which the triplets are finalized! \

-- **scripts** \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *general*  \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```install_anaconda.sh```: installing anaconda in your home directory in the cluster \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```create_environment.sh```: creating the conda environment that is used to run the code \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```pipeline_load_data.sh```: pipeline for running extraction with spaCy on cluster \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```pipeline_llm_cluster.sh```: pipeline for running extraction with LLMs on cluster \

&nbsp;&nbsp;&nbsp;&nbsp;|--- *general* \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```download_arxiv_data.sh```: sample bash script to download arXiv papers from 2023. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```patent_downloading.sh```: a script to download patent applications from 2018 - 2024. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```regression_test.sh```: a regression test for the triplet extraction pipeline. \

&nbsp;&nbsp;&nbsp;&nbsp;|--- *patents* \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```parse_patents.sh```: Parse the raw XML patent files into text files. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```process_patents.sh```: Select only those patents with the desired keywords. \

\
-- **code_structure** \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```code_overview.pdf```: A schematic overview of the full pipeline.  \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```parameter_description.pdf```: An overview and description of all parameters that can be set in ```configs.py``` \

\
-- **data** \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *fewshot_examples*:  \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```fewshotsamples_nlp.json```: Fewshot examples for the domain of NLP. For new domains one will have to construct the examples themself and put them in this folder. \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *memory_estimation*: \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```llama3_memory_estimate.json```: The memory estimation for the Llama3 model, for a specific setting that can be found in ```infos.json```. To estimate the memory usage of a different set-up, one can use the [Textwiz] (https://github.com/Cyrilvallez/TextWiz/tree/main) library.  \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```infos.json```: The set-up for which the ```llama3_memory_estimate.json``` holds. \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *entropy*: \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```category_counts.pickle```: The count per arXiv category for each term. The values are based on a *control corpus* from the period 2015-2023, with 200 papers sampled per month.  \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```entropy.pickle```: The cross-categorical entropy for each term. The values are based on a *control corpus* from the period 2015-2023, with 200 papers sampled per month.  \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```num_files_per_cat.pickle```: The total number of files for each arXiv category. The values are based on a *control corpus* from the period 2015-2023, with 200 papers sampled per month.  \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *target papers*: Here the user can put target papers that he or she wants.  \

\
-- ```requirements.txt```: File with environment requirements

\


### Requirements :mag:

#### Environment
The file ```requirements.txt``` contains the requirements needed, which are compatible with python 3.11.7. Using the following code snippet, an environment can be created:

```
conda create --name <env_name> python=3.11.7
pip install -r requirements.txt
```

If pip does not install the packages in your conda environment (you can check this with ```conda list```), you may need to explicitly call pip within the environment. Therefore, first install pip with ```conda install pip```, then check what your path is, it may look something like ```/anaconda/envs/venv_name/bin/pip```. You can then install all packages with ```/anaconda/envs/venv_name/bin/pip install -r requirements.txt```.

After installing these requirements, we still want to separately install PyTorch and Flash-attention. Depending on the Cuda driver, you will want to install the compatible Torch version, look for the versions [here](https://pytorch.org/get-started/previous-versions/). For Cuda 11.8, we run:

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install flash-attn --no-build-isolation
```

#### Data
The triplet extraction pipeline can be used for any type of textual data. However, for each different data source, there will be different preprocessing requirements. The code in this repository provides loading and preprocessing for arXiv papers and patents. The code is designed to be modular, such that the user can easily add new data sources.

* The arXiv preprints are publicly available in a Google Cloud bucket, for more information read [this Kaggle page](https://www.kaggle.com/datasets/Cornell-University/arxiv). Here, one can also download the arXiv metadata.

* Since the creation of this work, the Bulk Data Storage System (BDSS) of the USPTO has retired. There is now a new system to download the patent data, specifically the [Open Data Portal's Bulk Data Directory feature](https://data.uspto.gov/apis/transition-guide/bdss). We still provide the script ```scripts/general/patent_downloading.sh```, which shows how to unzip the raw patent data, however, the source link to the USPTO should be adapted following the new interface. 

* To download the arXiv data on a remote cluster, one can use [Nix](https://github.com/DavHau/nix-portable) with the Google-cloud-sdk package, which can be installed as follows:

```
wget https://github.com/DavHau/nix-portable/releases/download/v012/nix-portable-x86_64
chmod +x nix-portable-x86_64
```

Then, the data can be downloaded through Nix, an example script to download the data from 2023 is provided in the file ```scripts/general/download_arxiv_data.sh```.

### Usage of the code :memo:
The code is designed for the extraction of triplets from arXiv papers. 

#### Triplet extraction from arXiv papers

1. Clone the repository
2. Download the arXiv metadata, as available on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv). Put this under a folder with a name of your choice (the paths will be specified later in the ```configs.py``` file), it is recommended to simply put it directly in the ```data``` folder. 
3. For the triplet filtering, one needs the term entropy. Here you have two options:
    1. Use the entropy based on the sampled arXiv papers from 2015-2023, this is provided in teh folder ```data/entropy```.
    2. Recalculate the entropy with a corpus of your choice, specify the path to the corpus in ```configs.py``` and recalculate the entropy using the file ```calculate_entropy.py```.
4. Now define your set of _target papers_, which must be from arXiv, from which you want to extract triplets. Put these papers in a folder of your choice, for instance the folder ```data/target_papers``` .
5. To get access to the Llama-3-8b model, fill in the form on [the huggingface page](https://huggingface.co/meta-llama/Meta-Llama-3-8B) . Next, create a huggingface access token by going to your settings and into the access tokens part. Then, create a ```.env``` file in your root folder, with ```HUGGINGFACE_TOKEN = your_huggingface_api_token```. 
6. Now we are ready to start! Modify the settings as you like them in ```configs.py```. Note that - among others - you have to specify a folder to save the processed data, and a folder to save the results (e.g. triplets). We can then do the triplet extraction and comparison in the following steps:
    1. Run the triplet extraction and processing. If we run it locally, we run ```scripts/local/pipeline_llm.sh``` or ```scripts/local/pipeline_spacy.sh```. If we run it on a cluster, we run ```scripts/cluster/pipeline_llm.sh``` or ```scripts/local/pipeline_spacy.sh```.

#### Triplet extraction from patents
1. Clone the repository
2. Download the patent data from the USPTO, as described above. Put this under a folder with a name of your choice (the paths will be specified later in the ```configs.py``` file), it is recommended to simply put it directly in the ```data``` folder.
3. Parse and process the patent data, using the scripts ```scripts/general/parse_patents.sh``` and ```scripts/general/process_patents.sh```. One has to specify the path and the desired key terms in the ```configs.py``` file. The script ```parse_patents.sh``` will parse the raw XML files into text files, and the script ```process_patents.sh``` will select only those patents that contain the desired keywords. 
4. When using patents, one can either compute the entropy across different categories based on the CPC classes, or set the ```use_entropy``` variable in ```configs.py``` to ```False```. 
5. Now follow steps 4, 5 and 6 from the triplet extraction from arXiv papers.

### Regression test
To assert whether the whole triplet extraction pipeline is functioning (for instance after new updates), a simple regression test using **pytest** is available in ```src/regression_test```. To execute the test, one needs to have access to a GPU that fits a LLM with ~8B parameters.

* We provide a script to run the code on a cluster that uses slurm, through the script ```scripts/general/regression_test.sh```. If you want to run the test on a system that does not use slurm, you have to adjust the script accordingly.


Good luck, and if you have feedback, do not hesitate to reach out!
