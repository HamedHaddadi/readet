readet or _"read" "et"_ is a package developed using _LangChain_ to peruse scientific and technical literature. But all tools are applicable to any context. </br>
Eventhough this package has several tools, including multi-agent systems, but the current modules are used more frequently: </br>
➡️ summarizers that are used to summarize a text, mostly pdf files. </br>
➡️ RAGs or Retrieval Augmented Generation tools which can be used to ask questions about a document. </br>
➡️ prebuilt agents that are used to download papers and patents in bulk. </br>

here is the curtrent directory tree of the package </br>
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
    ├── image.py
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
apply for 1️⃣ to 3️⃣ first. With these APIs you can use most of the tools in the package. But it is good to obtain all APIs at some point. </br>


