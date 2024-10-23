
# ################################################## #
# Prebuilt functions and classes for agentic systems #
# ################################################## #
from os import path, makedirs, PathLike  
from datetime import datetime  
from typing import Literal, Union, List, Sequence, Optional 
# langchain, langgraph 
from langgraph.graph.graph import CompiledGraph 
from langgraph.graph import StateGraph 
from langgraph.prebuilt import create_react_agent 
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
# peruse tools 
from .. core import tools as peruse_tools 
from . base import ReAct, Assistant 
from . multi_agent import Supervisor


class ResearchAssistant:
	"""
	Assistant class the helps in search, download and summarzing pdf files
	"""
	def __init__(self, save_in: Union[str, PathLike], 
			  	search_in: Sequence[str] = ['arxiv'], max_results: int = 10) -> None:
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

	

