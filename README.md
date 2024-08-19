flm (Fluid Language Model)
Contains collection of tools, chains agents and graphs to search the literature, summarize articles and create databases
useful for complex fluid research. 
All tools can be applied to any other field of technical research.
Additionally, more agents can be added for other complex tasks, such as running simulations. 

The easiest way to run this package is to use the flm_env.yml file to setup the virtual environment and install all packages. After lauching a terminal:

conda env create -f flm_env.yml

this will create a virtual environment named 'flm' which can be activated by:

conda activate flm 

and deactivated by:

conda deactivate flm 

I also included requirements.txt file which can be used by:
pip3 install requirements.txt
 
After activation of the environment run the driver script 'run.py' by:
./run.py <'querty_pdf', 'scholar_search', 'schema_from_pdf'>





