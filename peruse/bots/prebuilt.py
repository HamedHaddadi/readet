
# ################################################## #
# Prebuilt functions and classes for agentic systems #
# ################################################## #
from os import path, makedirs 
from datetime import datetime  
from typing import Literal, Union, List, Sequence 
# langchain, langgraph 
from langgraph.graph.graph import CompiledGraph 
from langgraph.graph import StateGraph 
from langgraph.prebuilt import create_react_agent 
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
# peruse tools 
from .. core import tools as peruse_tools 
from . base import create_react_graph 
from . multi_agent import Supervisor

# ########################################################### #
# search_download_gist_graph: search in the search_in, download the pdf files and generate a gist of the material mentioned in the pdf and
# 	save them to a .txt file. 
# ########################################################### #
def search_download_query_by_keyword_graph(save_path: str, compile: bool = False, 
		agent_type: Literal['react'] = 'react',
			search_in: Sequence[str] = ['arxiv'], 
						max_results: int = 10, rag_type: str = 'agentic-rag-pdf', 
								summarizer_type: str = 'plain') -> Union[StateGraph, CompiledGraph]:

	save_path = path.join(save_path, f'{search_in}_search_download_query_results_on_' + datetime.now().strftime('%Y-%m-%d-%H-%M'))
	if not path.exists(save_path):
		makedirs(save_path)
	
	search_tools = []
	if search_in == 'scholar':
		search_tools.append(peruse_tools.GoogleScholarTool(api_wrapper = peruse_tools.GoogleScholarSearch(top_k_results = max_results)))
	elif search_in == 'patents':
		search_tools.append(peruse_tools.GooglePatentTool(api_wrapper = peruse_tools.PatentSearch(max_number_of_pages = max_results)))
	elif search_in == 'arxiv':
		search_tools.append(peruse_tools.ArxivTool(api_wrapper = peruse_tools.ArxivSearch(max_results = max_results)))

	download_tool = peruse_tools.PDFDownloadTool(downloader = peruse_tools.PDFDownload(save_path = save_path)) 
	gist_to_file_tool = peruse_tools.GistToFileTool(files_path = save_path, extractor = peruse_tools.ExtractKeywords(), 
						rag_type = rag_type, summarizer_type = summarizer_type) 	
	tools = [download_tool, gist_to_file_tool]
	tools.extend(search_tools)
	
	if agent_type == "react":
		return create_react_graph(tools, agent_use = f"""search in {search_in},
					 download the pdf files and generate a gist of the material mentioned in the pdf and
					 	save them to a .txt file. """, compile = compile)
	
# ########################################################### #
# search_download_summary_graph: search in the search_in, download the pdf files and generate a summary of the material mentioned in the pdf and
# 	save them to a .txt file. 
# ########################################################### #
def search_download_summary_graph(save_path: str, compile: bool = False, 
		agent_type: Literal['react', 'supervisor'] = 'react', search_in: Sequence[str] = ['arxiv'], 
			max_results: int = 10,  summarizer_type: Literal['plain'] = 'plain', 
				chat_model: str = 'openai-gpt-4o-mini', 
					added_prompt: str = "") -> Union[StateGraph, CompiledGraph, Runnable]:
	"""
	search in the search_in, download the pdf files and generate a summary of the material mentioned in the pdf and
		save them to a .txt file. 
	"""
	save_path = path.join(save_path, f'{search_in}_search_download_query_results_on_' + datetime.now().strftime('%Y-%m-%d-%H-%M'))
	if not path.exists(save_path):
		makedirs(save_path)
	
	search_tools = []
	if search_in == 'scholar':
		search_tools.append(peruse_tools.GoogleScholarTool(api_wrapper = peruse_tools.GoogleScholarSearch(top_k_results = max_results)))
	elif search_in == 'patents':
		search_tools.append(peruse_tools.GooglePatentTool(api_wrapper = peruse_tools.PatentSearch(max_number_of_pages = max_results),
				save_path = save_path))
	elif search_in == 'arxiv':
		search_tools.append(peruse_tools.ArxivTool(api_wrapper = peruse_tools.ArxivSearch(max_results = max_results)))

	download_tool = peruse_tools.PDFDownloadTool(downloader = peruse_tools.PDFDownload(save_path = save_path))
	summary_tool = peruse_tools.PDFSummaryTool(path_to_files= save_path, to_file = True,
		summarizer = peruse_tools.PDFSummary(chat_model = chat_model, summarizer_type = summarizer_type))	 
	
	tools = [download_tool, summary_tool]
	tools.extend(search_tools)
	if agent_type == "react":
		model = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
		return create_react_agent(model, tools = tools)
	
	elif agent_type == "supervisor":
		agents = {'search': search_tools, 'download': download_tool, 'summary': summary_tool}
		sub = Supervisor(agents, model = chat_model)
		graph = sub.build(compile = compile)
		return graph 

