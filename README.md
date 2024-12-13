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
│   ├── __init__.py
│   ├── agents.py
│   ├── chat_tools.py
│   ├── components.py
│   ├── multi_agents.py
│   └── prebuilt.py
├── core
│   ├── __init__.py
│   ├── chains.py
│   ├── knowledge_graphs.py
│   ├── rags.py
│   ├── retrievers.py
│   ├── summarizers.py
│   └── tools.py
└── utils
    ├── __init__.py
    ├── docs.py
    ├── io.py
    ├── models.py
    ├── save_load.py
    └── schemas.py
```
👉 __How to install__ </br>
I recommend setting up a virtual environment with python version 3.10 </br>
```console
conda create -n <name> python=3.10
```
</br>
Then you can activate the environment using </br>

```console
conda activate <name>
```
</br>
This will make sure the package dependencies remain inside the virtual environment. 
The package can be installed using 
```console
pip3 install readet
```
I also included the _requirements.txt_ file. </br>

👉 __How to use__ </br>
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


👉 __example use case 1__ </br>
📖 _summarizers_ </br>
I use the _PlainSummarizer_ as an example: </br>
First, import necessary functions and classes </br> 
```python
# use this function to load your API keys from keys.env file
from readet.utils.io import load_keys
load_keys('keys.env')
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
pdf_file = '../files/my_file.pdf'
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

👉 __example use case 2__ </br>
📑 _RAGS_ </br>

RAGS are used to ask questions about a document. Say you have a pdf file and you want to ask questions about the content without reading it. RAGS ingest the pdf file and store in a database (a vectorstore) and use LLMs to respond to your questions based on what they hold. All RAGs in this package can keep their database on your local computer. So you do not need to add pdf files from scratch all the time. </br>
readet contains several RAGs but working with all of them is the same. Here is a list </br>
1️⃣ _PlainRAG_: simple but useful RAG to ask questions about a pdf file </br>
2️⃣ _RAGWithCitations_: similar to plainRAG, but returns the reference as well (see an example below) </br>
3️⃣ _AgenticRAG_: RAG with extra checks to make sure the answer is relevant to the context of the document </br>
4️⃣ _SelfRAG_: RAG with introspection, to avoid hallucination </br>
5️⃣ _AdaptiveRAG_: RAG that screens the question based on the relevance to the document. If not relevant, it gives an answer by google search. For example, it does not allow you to answer question about salsa dancing from a fluid dynamics text </br>

I start with the _PlainRAG_ which is the simplest model: </br>
```python
from readet.utils.io import load_keys
load_keys('keys.env')
from readet.core.rags import PlainRAG
```
</br>
You can define a RAG from scratch, or initialize it from saved data. I start from the former case </br>

```python
pdf_file = './my_papers/fluidflow.pdf'
# define your RAG store path here
store_path = './myRAGS'
rag = PlainRAG(documents = pdf_file, store_path = store_path)
```
</br>
This will give you a function for asking questions: </br>

```python
rag("who are the authors of this work?")
rag("what is the relationship between fluid pressure and solid content?")
```
</br>
Let's start the RAG from the previously saved database (or "vector store"). This will allow you to add new pdf files, or keep asking question from the old files. </br>
here are parameters that you need to pass to the class: </br>

``` python
# this parameter can also be None, if you do not want to add any new pdf file
new_pdf_file = './my_papers/turbulence.pdf'
# directory path
store_path = './myRAGS'
# either use a version number, ex 0,1,.., or pass 'last'
load_version_number = 'last'
rag2 = PlainRAG(documents = new_pdf_file, store_path = store_path, load_version_number = load_version_number)
```
</br>
Now you can ask questions. </br>

```python
rag2("what is the relationship between inertia and viscosity?")
```
</br>

Let's use _RAGWithCitations_ as well: </br>

```python
from readet.utils.io import load_keys
load_keys('keys.env')
from readet.core.rags import RAGWithCitations
pdf_file = './files/HaddadiMorrisJFM2014.pdf'
store_path = './RAGStore'
rag = RAGWithCitations(pdf_file, store_path = store_path)
rag("what is the relationship between inertia and normal stress?")
```
</br>
And here is the answer: 

```console
'Inertia affects the normal stress in suspensions by influencing the distribution of particles and their interactions under shear flow. As inertia increases, it can lead to higher particle pressure and changes in the normal stress differences, particularly the first normal stress difference (N1), which becomes more negative with increasing inertia and volume fraction. This relationship highlights the complex interplay between inertia and stress in particle-laden fluids, where increased inertia amplifies the effects of excluded volume and alters the stress distribution within the suspension.',
 'Haddadi, H. & Morris, J. F. (2023). Microstructure and Rheology of Finite Inertia Suspensions. J. Fluid Mech.'
```
</br>

_I use one more example of the AdaptiveRAG and move on to the next example usage. All other RAGs mentioned above work the same_ </br>

```python
from readet.core.rags import AdaptiveRAG
from readet.utils.io import load_keys
load_keys('keys.env')

# can be None if you want to load from database
pdf_file = './files/fluidflow.pdf'
store_path = './RAGFluid'
# if you want to load from database, choose a verion number or 'last'; else None
load_version_number = None

rag = AdaptiveRAG(documents = None, store_path = store_path, load_version_number = 'last')
rag("what is relationship between Reynolds number and viscosity?")

```
</br>
And here is the answer: </br>

```console
The Reynolds number (Re) is a dimensionless quantity that characterizes the flow regime in fluid dynamics, influenced by factors such as velocity, characteristic length, and viscosity. Generally, as Re increases, the effects of inertia become more significant compared to viscous forces, which can lead to changes in flow behavior. However, the viscosity itself may not show significant changes with varying Re, as indicated in the context provided.

```

👉 __example use case 3__</br>
📚 _search and download several papers from Google Scholar and Arxiv_ </br>
This tool has been a real convenience for me and I hope it helps you as well. I explain how it works. But I included this tool as an agent in a _multi agent_ chat bot and I deploy that chatbot soon. You can use this tool, summary and RAGs to peruse a lot papers. </br>
⚠️To use the Download functionality , you need _OpenAI_ and _Serp API_ API keys. Use the links in the first part of this ReadMe document to obtain the API keys. </br>
⚠️ ⚠️ To use this agent, prompting is important. Make sure to mention _"search and download"_ if you want the agent to download the files for you. Otherwise, it will output a list of papers and their information and links to download the article. </br>

```console
from readet.utils.io import load_keys
load_keys('keys.env')
from readet.bots.prebuilt import Download
```
</br>
Now you can define the parameters. These parameters are a path to save the downloaded files and maximum number of papers to download. Note that if you connection to the download faces a publisher paywall, the pdf file is not downloaded. But you can use the list of papers that are found to identify those papers and ask some to download it for you. </br>

``` python
save_path = './pdfs'
max_results = 100
downloader = Download(save_path = save_path, max_results = max_results)

# NOTE: if you want to download the paper, explicitly mention the word 'download'
download("search and download all papers related to finite inertia suspension flow of ellipsoidal particles")
```
The downloaded files are stored in _save_path_. A '.txt' file containing information of the papers is also stored in the _save_path_ directory </br>
For example, the first record in this file is : </br>

```console
*******************
Title: Numerical study of filament suspensions at finite inertia
Authors: AA Banaei, ME Rosti, L Brandt
Citation Count: 36
PDF Link: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/5FA754F237DC68A6721F7C055FA08CEC/S0022112019007948a.pdf/div-class-title-numerical-study-of-filament-suspensions-at-finite-inertia-div.pdf
```
for example, you can send this file to colleagues via email. </br>

I am continuosly adding more functionalities. Hope this package is useful for your scientific discovery 🤞





















