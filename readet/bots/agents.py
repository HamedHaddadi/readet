# Generate classes and functions to run all chatbots #
from typing import (Dict, List, Sequence, Union, Literal, Callable, Optional)
from .. utils import models
from .. core import tools as readet_tools   
# langgraph imports
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool   
from langgraph.prebuilt import create_react_agent 
from functools import reduce
from os import getcwd, path, listdir, PathLike
from .. utils import docs

 
# ########################### #
# ReAct graph 				  #
# ########################## #
class ReAct:
	"""
	Class for creating a ReAct agent by wrapping create_react_agent
	of langgraph.prebuilt
	tools: a sequence of BaseTool objects or a tool object or a sequence of tool names
	"""
	def __init__(self, tools: Union[Sequence[BaseTool], BaseTool, Sequence[str]], 
			  chat_model: str = 'openai-gpt-4o-mini',
			   	added_prompt: str = "you are a helpful AI assistant", **tools_kwargs):
		
		tools = self._configure_tools(tools, tools_kwargs)
		model = models.configure_chat_model(chat_model, temperature = 0)
		self.runnable = create_react_agent(model, tools, state_modifier = added_prompt)

	# configure tools 
	def _configure_tools(self, tools: Union[Sequence[BaseTool], BaseTool, Sequence[str]], tools_kwargs: Dict) -> List[BaseTool]:
		if isinstance(tools, BaseTool):
			tools = [tools]
		elif isinstance(tools, Sequence) and all(isinstance(tool, str) for tool in tools):
			tools = [readet_tools.get_tool(tool, tools_kwargs) for tool in tools]
		return tools 

	def run(self, query: str):
		self.runnable.invoke({"messages": [HumanMessage(content = query)]})

	def __call__(self, query: str):
		self.run(query)

# ################################################## #
# 	Download agent using ReAct class 				#
# ################################################## #
class Download(Callable):
	"""
	Download agent using ReAct class. searches and downloads papers from arxiv and scholar
	"""
	PROMPT = """You are a specialized assistant for searching technical papers on google
				scholar and arxiv and downloading them. search and download all the papers that are relevant to the query."""
	def __init__(self, save_path: str, max_results: int = 100, 
			  		chat_model: str = 'openai-gpt-4o-mini', 
						search_in: List[Literal['google_scholar', 'arxiv', 'google_patent']] = ['arxiv', 'google_scholar']) -> None:
		self.save_path = save_path 
		tools = [tool_name + '_search' for tool_name in search_in] + ['pdf_download']
		self.agent = ReAct(tools = tools, chat_model = chat_model, added_prompt = self.PROMPT,  
								max_results = max_results, save_path = save_path)
	
	def __call__(self, query: str, list_files: bool = False) -> None:
		self.agent(query)
		replacements = {'_': ' ', '.pdf': ''}
		if list_files:
			pdf_files = [reduce(lambda a,kv: a.replace(*kv), replacements.items(), filename) for filename in listdir(self.save_path) if filename.endswith('.pdf')]
			if len(pdf_files) >  0:
				pdf_names = ''.join([f'{count + 1}.{filename}\n' for count, filename in enumerate(pdf_files)])
				newline = '\n'
				return f"**The following pdf files are downloaded: {newline} {pdf_names}"
			else:
				return "**No pdf files are downloaded! check your Serp API key; or change the search query!"

# ################################################## #
# 	Text to chart agent 							 #
# ################################################## #
class TextToCharts(Callable):
	"""
	Text to chart agent using PythonREPL and react agent
	chat_model: the chat model to use for the agent
	save_path: the path to save the charts
	"""
	PROMPT = f""" you are a python code developer that reads the text
	  and generates python code for generating charts. 
	"""
	def __init__(self, chat_model: str = 'openai-gpt-4o', 
			  		save_path: Optional[str] = None) -> None:
		self.save_path = {True: getcwd(), False: save_path}[save_path is None]
		self.agent = ReAct(tools = ['python_repl'], chat_model = chat_model, added_prompt = self.PROMPT)

	def __call__(self, text:str | PathLike) -> None:
		if path.isfile(text) and text.endswith('.pdf'):
			text = docs.text_from_pdf(text)
		
		message = f""" generate chart with nice styling and Times New Roman for font axis
                    from the following text. Text information may allow you
					  to generate more than one chart. If so, attach 
                        multiple figures as subfigures. 
						save the figures in the {self.save_path} directory with 'jpeg' format.
                        Do not generate reduntant charts. Here is the text: \n {text} \n
		"""
		self.agent(message)

	
	

