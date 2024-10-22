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
ResearchAssistant can be instantiated using following keywords:
```python
save_in = 'a/path/in/your/desk/for/downloading/papers/and/storing/summaries'
# maximum number of papers to download
maximum_number_of_results=100
bot = ResearchAssistant(save_in, maximum_number_of_results = maximum_number_of_results)
```
currently, this bot is working in two modes. _search_ and _summary_. I am working on a bot that automatically routes tasks and I will push a new version soon. 
so, if you like to search, activate the search mode by 
``` python
bot.enable_search()
```
This will activate the chatbot. Try to ask relevant questions. But it will chat with you about any topic anyway. Here is an example conversation:
```console
User:  Hi
Assistant: Hello! How can I assist you today?
User:  what can you do for me?
Assistant: I can assist you in several ways, particularly with scientific research. Here are some of the things I can do for you:
1. **Search for Scientific Papers**: I can help you find research papers on specific topics by searching the arXiv database.
2. **Download Papers**: Once I find the papers you're interested in, I can download them for you.
3. **List Files**: I can show you the files that you have stored in a specific directory.

If you have a specific topic or paper in mind, just let me know, and I'll get started!
User:  yes, please search and download all papers related to finite inertia suspension flows
Assistant: I am working... wait!
Assistant: I have successfully downloaded all the papers related to finite inertia suspension flows. They are now stored in the specified directory. If you need any further assistance, feel free to ask!
User:  bye
Ciao!
```
Say bye, goodbye or quit, else search mode remains active for more chat.

Now you can summarize papers. You can summarize individual articles. For example:
```python
bot.enable_summary()
```
and this is a conversation

``` console
User:  Hi
Assistant: Hello! How can I assist you today?
User:  list all files
Assistant: I am working... wait!
Assistant: Here are the files listed in the specified directory:

1. Rheology of dense granular suspensions under extensional flow.pdf
2. Channel flow of rigid sphere suspensions: particle dynamics in the inertial regime.pdf
3. Rheology of concentrated suspension of fibers with load dependent friction coefficient.pdf
4. Unified Theory of Inertial Granular Flows and Non-Brownian Suspensions.pdf
5. Suspensions of deformable particles in Poiseuille flows at finite inertia.pdf
6. Shear thickening in non-Brownian suspensions: an excluded volume effect.pdf
7. Interplay of inertia and deformability on rheological properties of a suspension of capsules.pdf
8. Taylor-Couette flow of hard-sphere suspensions: Overview of current understanding.pdf
User:  summarize Rheology of concentrated suspension of fibers with load dependent friction coefficient.pdf

Assistant: I am working... wait!
Assistant: The document presents a numerical study on the rheology of concentrated suspensions of fibers, focusing on the effects of fiber aspect ratio, roughness, flexibility, and flow inertia. Key findings include:
1. **Modeling Approach**: Direct numerical simulations using the Euler-Bernoulli beam equation model flexible fibers suspended in a Newtonian fluid, employing an Immersed Boundary Method (IBM) for fiber motion and interactions.
2. **Friction Dynamics**: A load-dependent friction coefficient is introduced to model contact dynamics between fibers, capturing the shear-thinning behavior observed in experiments. Contact stresses between fibers significantly contribute to the suspension's viscosity.
3. **Viscosity Behavior**: The viscosity of the suspension increases with fiber volume fraction, roughness, and rigidity, especially at lower shear rates. The first normal stress difference is positive and increases with Reynolds number for flexible fibers.
4. **Jamming Phenomenon**: The study explores the jamming volume fraction, finding it decreases with increasing aspect ratio and roughness. A modified Maron-Pierce law is proposed to predict jamming behavior.
5. **Industrial Relevance**: Understanding the rheological behavior of fiber suspensions is crucial for various industrial applications, including material reinforcement and pulp production. Modifying fiber properties, such as reducing roughness or increasing flexibility, can enhance flowability in concentrated suspensions.
6. **Conclusion**: The research emphasizes the importance of modeling friction and contact forces in understanding the rheological behavior of fiber suspensions, highlighting the need for further studies to quantify the effects of these parameters in practical applications.
Overall, the study provides insights into the complex interactions governing the flow behavior of concentrated fiber suspensions, which are relevant for optimizing industrial processes.
```

or you can summarize all of them by 
``` console
User:  summarize all
Assistant: I am working... wait!
Assistant: I am working... wait!
```
### a few points to mention:
➡️ summaries are appended to a summary.txt file. So, if you summarize individual articles, and then summarize all, you perhaps summarize some papers twice. Remember to delete the old summary files. <br/>
➡️ I included several other tools for research. I have not included them in chatbots yet. Take a look at _peruse.core.tools_ and _peruse.core.rags_. 











