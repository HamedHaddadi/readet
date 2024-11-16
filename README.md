# **peruse (Language models to dig into scientific literature)**
Contains collection of tools, runnables, agents and graphs to search the scientific and technical literature, including Google Scholar, Arxiv and Google Patents. You can also download and summarize articles and create knowledge graphs. It is also possible to 
query pdf files using different Retrieval Augmented Generation (RAG) systems. 
I have personally used this tool in my main area of expertise, complex fluids, computational fluid dynamics and computational materials science. 
All tools can be applied to other fields of technical research.
Additionally, more agents can be added for other complex tasks, such as running simulations.

## Install the requirements
The easiest way to run this package is to use the requirements.txt file to setup the virtual environment and install all packages. using conda:
``` console 
conda create -n <your environment name> python=3.10.0
```
Then install all dependencies by 

``` console
pip3 install -r requirements.txt
```
inside the activated environment.
in case you need to reactivate the environment use
```console
conda activate <your environment name>
```

it is also possible to install dependencies in a conda environment. create a .YAML file and copy the text below. choose a name for the file, for example peruse.yaml:

```console
name: <choose a name>
channels:
  - conda-forge
dependencies:
  - python=3.10.0
  - anaconda
  - pip
  - pip:
    - -r requirements.txt
```
Then use the command below to setup the environment:

```console
conda env create -f peruse.yaml
```

### apply and obtain API_Keys
You will need API keys for the following services: <br/>
➡️ _OpenAI_ : for chat and embedding models <br/>
➡️ _Langchain_ : For using LangSmith <br/>
➡️ _SerpAI_ : For Google Scholar tools <br />

The easiest way is to store your API keys with the following names in a _key.env_ files in a separate directory called _configs_. <br/>
I recommend using the following keys and formats: <br/>
_OPENAI_API_KEY_ = "your API key" <br/> 
_LANGCHAIN_API_KEY_ = "your API key" <br/>
_SERP_API_KEY_ = "your API key" <br/>

### run ResearchAssistant from a Jupyter notebook
If you have created your keys.env file and stored it in _peruse/configs_ directory, do the following in the jupyter notebook. 
``` python
from dotenv import load_dotenv
load_dotenv('./peruse/configs/keys.env')
```
This will return a _True_ or _False_. If _False_ make sure the path is correct. For now your jupyter notebook should be in a same directory as _peruse_. 
Then run the following 
```python
from peruse.bots.prebuilt import ResearchAssistant
```

### a few points to mention:
➡️ summaries are appended to a summary.txt file. So, if you summarize individual articles, and then summarize all, you perhaps summarize some papers twice. Remember to delete the old summary files. <br/>
➡️ I included several other tools for research. I have not included them in chatbots yet. Take a look at _peruse.core.tools_ and _peruse.core.rags_. 











