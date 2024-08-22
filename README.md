#**flm (Fluid Language Model)**
Contains collection of tools, chains agents and graphs to search the literature, summarize articles and create databases
useful for complex fluid research. 
All tools can be applied to any other field of technical research.
Additionally, more agents can be added for other complex tasks, such as running simulations. 

## Install the requirements
The easiest way to run this package is to use the flm_env.yml file to setup the virtual environment and install all packages. After lauching a terminal:

```console
conda env create -f flm_env.yml
```

this will create a virtual environment named 'flm' which can be activated by:
``` console
conda activate flm 
```
and deactivated by:
``` console
conda deactivate flm 
```
I also included requirements.txt file which can be used by:
```
pip3 install requirements.txt
```
After activation of the environment run the driver script _run.py_ by:

``` console
./run.py <name of the functions>
```

currently, three choices are supported:
_query_pdf_: which is used to query a pdf document using self-reflective Retrieval Augmented Generation 
_search_scholar_: which is used to search the google scholar and output the results in an interactive graph
_schema_from_pdf_: which is used to extract schema from pdf for building databases 





