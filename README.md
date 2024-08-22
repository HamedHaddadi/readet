# **flm (Fluid Language Model)**
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
### apply and obtain API_Keys
You will need API keys for the following services: <br/>
➡️ _OpenAI_ : for chat and embedding models <br/>
➡️ _Langchain_ : For using LangSmith <br/>
➡️ _Tavily search_: For Tavily search tools <br/>
➡️ _SerpAI_ : For Google Scholar tools <br />
The easiest way is to store your API keys with the following names in a _key.env_ files in a separate directory called _configs_. <br/>
I recommend using the following keys and formats: <br/>
_OPENAI_API_KEY_ = "your API key" <br/> 
_LANGCHAIN_API_KEY_ = "your API key" <br/>
_TAVILY_API_KEY_ = "your API key" <br/>
_SERP_API_KEY_ = "your API key" <br/>
Then define the path to your _keys.env_ in the _KEYS_ variable in run.py file. <br/>

### Instantiate classes and run functions
currently, these choices are supported but I am adding features weekly: <br/>
➡️ _query_pdf_: which is used to query a pdf document using self-reflective Retrieval Augmented Generation <br/> 
➡️ _search_scholar_: which is used to search the google scholar and output the results in an interactive graph <br/>
➡️ _schema_from_pdf_: which is used to extract schema from pdf for building databases <br/>
➡️ _query_pdf_and_search_: which is used to query a pdf file for specific information and perform a more comprehesive search on retrieved information. <br/>

### Example
Let's query a [scientific paper on interfacial rheology](https://pubs.acs.org/doi/full/10.1021/acs.langmuir.2c00460?casa_token=WZcUAM0NbEsAAAAA%3AgUutytC-cTT6fJud7B9Buuof3ZxObYdYhwJCa0nX6aVLogZjQwpjiSDtvvU-_yBDb_sbAJBSP1D5sFQ) and perform a google search on retrieved information. We can query this document for the raw materials used to conduct the experiments. <br/>

``` console
./run query_pdf_and_search
```
you will be prompted to enter the full path and name of the pdf file that you stored on your local machine. <br/>
Currently, I use path on Linux systems. <\br>








