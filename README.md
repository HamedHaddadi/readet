# **peruse (Language Models for scientific perusing)**
Contains collection of tools, runnables, agents and graphs to search the literature. download and summarize articles and create knowledge graphs. It is also possible to 
query pdf files using different Retrieval Augmented Generation (RAG) systems. 
I have personally used this tool in my main area of expertise, complex fluids, computational fluid dynamics and computational materials science. 
All tools can be applied to other fields of technical research.
Additionally, more agents can be added for other complex tasks, such as running simulations.
Currently I am working on dofferent chatbots. But this repo is useful for many things already.

## Install the requirements
The easiest way to run this package is to use the peruse_requirements.yml file to setup the virtual environment and install all packages. After lauching a terminal:

```console
conda env create -f peruse_requirements.yml
```

this will create a virtual environment named 'peruse' which can be activated by:
``` console
conda activate peruse 
```
and deactivated by:
``` console
conda deactivate peruse
```
I also included requirements.txt file which can be used by:
```
python3 -m venv <peruse or any other name>
pip3 install requirements.txt
```
(if you prefer to set up a virtual environment of course). 

If you like to use this repository, the only class you need is _ResearchAssistant_ from _peruse/bots/prebuilt_. 

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
Then define the path to your _keys.env_ in the _KEYS_ variable in run.py file. <br/>










