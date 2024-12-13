# readet
🚧 _until I prepare a more comprehensive documentation, use this readme to work with the package_ </br>
⚠️ If you run this package on a Windows machine, make sure you define the paths to files accordingly. </br>
⚠️ this documentation explains how to use the functionalities using a minimal set of inputs and using default arguments. But you can control parameters if you want. I will add the details in the documentation soon.

readet is a package developed using _LangChain_ for perusing scientific and technical literature. But all tools are applicable to any context. </br>
Eventhough several functionalities are included in this package, such as multi-agent systems, these modules are used more frequently: </br>
➡️ summarizers that are used to summarize a text, mostly pdf files. </br>
➡️ RAGs or Retrieval Augmented Generation tools which can be used to ask questions about a document. </br>
➡️ prebuilt agents that are used to download papers and patents in bulk. </br>

here is the current directory tree of the package </br>
```console
readet
├── __init__.py
├── bots
│   ├── __init__.py
│   ├── agents.py
│   ├── chat_tools.py
│   ├── components.py
│   ├── multi_agents.py
│   └── prebuilt.py
├── core
│   ├── __init__.py
│   ├── chains.py
│   ├── knowledge_graphs.py
│   ├── rags.py
│   ├── retrievers.py
│   ├── summarizers.py
│   └── tools.py
└── utils
    ├── __init__.py
    ├── docs.py
    ├── models.py
    ├── prompts.py
    ├── save_load.py
    └── schemas.py
```
__How to install__ </br>
I recommend setting up a virtual environment with python version 3.10 </br>
```console
conda create -n <name> python=3.10
```
This will make sure the package dependencies remain inside the virtual environment. 
The package can be installed using 
```console
pip3 install readet
```
I also included the _requirements.txt_ file.

__How to use__ </br>
This package uses several _API_ s that need API keys. Fortunaletly, all of them are free for a while (or forever if you do not use them too often). Here is the list of APIs </br>
1️⃣ OpenAI </br>
2️⃣ Serp API </br>
3️⃣ Anthropic </br>
4️⃣ Tavily Search </br>
5️⃣ LangChain </br>
6️⃣ Hugging Face </br>
apply for 1️⃣ to 3️⃣ first. With these APIs you can use utilize most of the functionalities in this package. But it is good to obtain all APIs at some point. </br>
The easiest way is to define all API keys in a _keys.env_ file and load it in your environment. The keys.env file is structured as </br>
OPENAI_API_KEY ="<you key>" </br>
TAVILY_API_KEY="<your key>" </br>
SERP_API_KEY="<your key>" </br>
ANTHROPIC_API_KEY ="<your key>" </br> 


__quick example usage 1__ </br>
📖 _summarizers_ </br>
I use the _PlainSummarizer_ as an example: </br>
First, import necessary functions and classes </br> 
```python
# to define paths
from os import path
# for pretty prints of the summary

from readet.utils.io import load_keys
from readet.core.summarizers import PlainSummarizers
```
</br>
Now define parameters: </br>

```python
# you can define any model from openai. Include 'openai-' before the model name.
# example: 'openai-gpt-4o'
chat_model = 'openai-gpt-4o-mini'
# degree of improvisation given to the model; 0 is preferred
temperature = 0
# instantiate the summarizer
plain_summarizer = PlainSummarizer(chat_model = chat_model, temperature = temperature)
```
</br>
Now specify the path to your pdf file and run the summarizer: </br>

```python
# note that your path might be different. In Windows, MacOS or Linux. Choose the exact path
pdf_file = path.join('../files/my_file.pdf')
response = plain_summarizer(pdf_file)
```
</br>
You can print the response to see the summary </br>
Also, You may run the callable as much as you want to many pdf files: </br>

```python
pdf_files = ['./my_papers/paper.pdf', './my_patents/patent.pdf']
responses = {}
for count,pdf in enumerate(pdf_files):
    responses[f'summary_{count}'] = plain_summarizer(pdf)
```
</br>
Note that ingesting pdf files may take some time. For a general scientific paper it may take about 12 seconds. Later when I explain RAGs, I will describe a method to store ingested pdf files to avoid spending too much time reading pdf files from scratch. </br>

__quick example usage 2__
📑 _RAGS_ </br>
RAGS are used to ask questions about a document. Say you have a pdf file and you want to ask questions about the content without reading it. RAGS ingest the pdf file and store in a database (a vectorstore) and use LLMs to respond to your questions based on what they hold. All RAGs in this package can keep their database on your local computer. So you do not need to add pdf files from scratch all the time. </br>
readet contains several RAGs but working with all of them is the same. I start with the _PlainRAG_ which is the simplest model: </br>
```python
from readet.utils.io import load_keys
load_keys('./keys.env')
from readet.core.rags import PlainRAG
```
</br>
You can define a RAG from scratch, or initialize it from saved data. I start from the former case </br>

```python
pdf_file = './my_papers/fluidflow.pdf'
# define your RAG store path here
store_path = './myRAGS'
rag = PlainRAG(pdf_file, store_path = store_path)
```
</br>
This will give you a function for asking questions </br>:
```python
rag("who are the authors of this work?")
rag(""what is the relationship between fluid pressure and solid content?")
```
</br>







