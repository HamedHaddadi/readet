readet or _readet_ is a package developed using _LangChain_ to peruse scientific and technical literature. But all tools are applicable to any context. </br>
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
│   ├── **rags.py**
│   ├── retrievers.py
│   ├── **summarizers.py**
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
