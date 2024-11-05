
# ################################################## #
# Prebuilt functions and classes for agentic systems #
# ################################################## #
from os import PathLike  
from typing import Union, Sequence, Literal
# langchain, langgraph 
from . assistants import Assistant
from . agents import ReAct 

class ResearchAssistantLite:
	"""
	Assistant class the helps in search, download and summarzing pdf files
	"""
	def __init__(self, save_in: Union[str, PathLike], 
			  	search_in: Sequence[str] = ['arxiv', 'google_scholar'], max_results: int = 10) -> None:
		self.save_in = save_in 
		self.search_assistant = None 
		self.summary_assistant = None 

		self._configure_search_assistant(search_in, max_results)
		self._configure_summary_assistant()

	def _configure_search_assistant(self, search_in: Sequence[str], max_results: int) -> None:
		added_prompt = f"""you are a helpful assistant that helps in searching for scientific papers
		 	in {', '.join(search_in)} and downloading them to {self.save_in}"""
		tools = [search + '_search' for search in search_in] + ['pdf_download', 'list_files']
		search_agent = ReAct(tools, added_prompt = added_prompt, save_path = self.save_in, max_results = max_results)
		self.search_assistant = Assistant.from_graph(search_agent.runnable)
	
	def _configure_summary_assistant(self) -> None:
		added_prompt = f"""you are a helpful assistant that summarizes pdf files; remember to
			give a response using tools that you have. Do not speculate"""
		tools = ['summarize_pdfs', 'list_files']
		summary_agent = ReAct(tools, added_prompt = added_prompt, save_path = self.save_in)
		self.summary_assistant = Assistant.from_graph(summary_agent.runnable)
	
	def enable_search(self) -> None:
		self.search_assistant()
	
	def enable_summary(self) -> None:
		self.summary_assistant()

# ################################################## #
#  Research Assistant 								 #
# ################################################## #
class ResearchAssistant:
	"""
	Research assistant class that uses 
		multiple subgraphs to search, download, summarize and query pdf files
	the number of subgraphs depends on the tools that are used by the agent.
	The primary assistant automatically handles the conversation between agents.
	Note that the number of agents is fixed:
		search agent: uses arxiv and scholar to search for papers and downloads them
		summary agent: summarizes the downloaded pdfs
		query agent: queries the downloaded pdfs
	"""
	def __init__(self, max_results: int = 10) -> None:
		self.search_runnable = None 
		self.summary_runnable = None 
		self.query_runnable = None
		self._configure_search_assistant(max_results)
		self._configure_summary_assistant()
		self._configure_query_assistant()

	def build(self) -> None:
		pass 

	def run(self) -> None:
		pass 



	

