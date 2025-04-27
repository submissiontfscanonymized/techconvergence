## LLM-Resilient Bibliometrics: Factual Consistency Through Entity Triplet Extraction

**Last update: 27/04/2025**

This github repository provides the code that belongs to the project "Monitoring Transformative Technological Convergence Through LLM-Extracted Semantic Entity Triple Graph". The code provides the full pipeline from raw arXiv pdf's to processed entity triplets of the shape (subject, predicate, object). The triplets are extracted through LLM's.

### Structure :books:

-- **src** \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```configs.py```: In this file, you specify all the settings for the triplet extraction and processing. For more information and an overview of all parameters see the folder ```code_structure/parameters.pdf```  \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```contexts.py```: Here, the contexts classes are defined based on the user settings in the configuration file.  \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```helpers.py```: This file provides several helper functions, such as loggers, that are used throughout the pipeline. \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```estimation.py```: This file provides memory estimations for specific input sizes, such that the batch size can be set dynamically. \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```load_data.py```: This file loads the pdf's and converts them to text files \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```preprocessing.py```: This file preprocesses the text files \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```extract_triplets_llm.py```: This file extracts the triplets from the text files using a LLM \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```process_triplets.py```: This file processes the triplets \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```filter_triplets.py```: This file filters the triplets, after which the triplets are finalized! \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```triplet_analysis.py```: ...work in process... \
\
-- **scripts** \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *armasuisse_cluster*  \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```install_anaconda.sh```: installing anaconda in your home directory in the cluster \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```create_environment.sh```: creating the conda environment that is used to run the code \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```pipeline_load_data.sh```: pipeline for running extraction with spaCy on cluster \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```pipeline_llm_cluster.sh```: pipeline for running extraction with LLMs on cluster \

&nbsp;&nbsp;&nbsp;&nbsp;|--- *general* \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```download_arxiv_data.sh```: sample bash script to download arXiv papers from 2023. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```patent_downloading.sh```: some work in progress on USPTO patents. \

&nbsp;&nbsp;&nbsp;&nbsp;|--- *hevs_cluster* \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- ```create_environment.sh```: Creating a conda environment on the HEVS cluster. \

\
-- **code_structure** \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```code_overview.pdf```: A schematic overview of the full pipeline.  \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```parameter_description.pdf```: An overview and description of all parameters that can be set in ```configs.py``` \
\
-- **data** \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *fewshot_examples*: The folder to put fewshot examples for triplet extraction, in the repo you will find fewshot examples for the domain of NLP and for the domain of quantum computing. For different domains the user has to make fewshot examples themself.  \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *memory_estimation*: The folder includes the memory estimation for one GPU set up, the exact set up is given in the file ```infos.json```. To estimate the memory usage of a different set-up, one can use the [Textwiz] (https://github.com/Cyrilvallez/TextWiz/tree/main) library.  \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *entropy*: The cross-categorical entropy values. The repository provides values based on a *control corpus* from the period 2015-2023, with 200 papers sampled per month.  \
&nbsp;&nbsp;&nbsp;&nbsp;|--- *target papers*: Here the user can put target papers that he or she wants.  \
\
-- ```requirements.txt```: File with environment requirements



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
Before using the code, you need to load data and the claim extraction model. The required data to run the pipeline is available in [this Google Drive](https://drive.google.com/drive/folders/1VzIWOI6PPWNSOLkZUZPCbuorUEroMW2W?usp=sharing).

* The code is designed for the extraction of triplets from arXiv papers. These articles are publicly available in a Google Cloud bucket, for more information read [this Kaggle page](https://www.kaggle.com/datasets/Cornell-University/arxiv). Here, one can also download the arXiv metadata.

* To download the arXiv data on a remote cluster, one can use [Nix](https://github.com/DavHau/nix-portable) with the Google-cloud-sdk package, which can be installed as follows:

```
wget https://github.com/DavHau/nix-portable/releases/download/v012/nix-portable-x86_64
chmod +x nix-portable-x86_64
```

Then, the data can be downloaded through Nix, an example script to download the data from 2023 is provided in the file ```download_arxiv_data.sh```.

### Usage of the code :memo:
The code is designed for the extraction of triplets from arXiv papers. You have to take the following steps to use the code:

1. Clone the repository
2. Download the arXiv metadata, as available on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv). Put this under a folder with a name of your choice (the paths will be specified later in the ```configs.py``` file).
3. For the triplet filtering, one needs the term entropy. Here you have two options:
    1. Use the entropy based on the arXiv papers from 2023, this is available in the [Google Drive](https://drive.google.com/drive/folders/1VzIWOI6PPWNSOLkZUZPCbuorUEroMW2W?usp=sharing). Put the files in the folder ```data/entropy```.
    2. Recalculate the entropy with a corpus of your choice, specify the path to the corpus in ```configs.py``` and recalculate the entropy using the file ```calculate_entropy.py```.
4. Now define your set of _target papers_, which must be from arXiv, from which you want to extract triplets. Put these papers in a folder of your choice.
5. To get access to the Llama-3-8b model, fill in the form on [the huggingface page](https://huggingface.co/meta-llama/Meta-Llama-3-8B) . Next, create a huggingface access token by going to your settings and into the access tokens part. Then, create a ```.env``` file in your root folder, with ```HUGGINGFACE_TOKEN = your_huggingface_api_token```. 
6. Now we are ready to start! Modify the settings as you like them in ```configs.py```. Note that - among others - you have to specify a folder to save the processed data, and a folder to save the results (e.g. triplets). We can then do the triplet extraction and comparison in the following steps:
    1. Run the triplet extraction and processing. If we run it locally, we run ```scripts/local/pipeline_llm.sh``` or ```scripts/local/pipeline_spacy.sh```. If we run it on a cluster, we run ```scripts/cluster/pipeline_llm.sh``` or ```scripts/local/pipeline_spacy.sh```.
    2. The pipeline outputs the nodes and edges corresponding to the entities and the relations. Through Neo4J, and the queries present in ```scripts/neo4j/match_triplets.cypher```, you can find the similar triplets.

In case you want to use the triplets not for a factual consistency comparison, but for another downstream analysis, this is also possible. In this case, you stop after step 6.1. The triplets will be saved in the folder that you have specified in ```configs.py```.

### Regression test
To assert whether the whole pipeline is functioning (for instance after new updates), a simple regression test using **pytest** is available in ```src/regression_test```. To execute the test, one needs to have access to a GPU. 

* If you are directly on a machine (e.g. on the HEVS cluster), from the root repository, run the command ```bash scripts/hevs_cluster/regression_test.sh```
* If you are using slurm (e.g. on the armasuisse cluster), submit the job ```sbatch scripts/armasuisse_cluster/regression_test.sh```


Good luck, and if you have feedback, do not hesitate to reach out!
