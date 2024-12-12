# Generate classes and functions to run all chatbots #
from typing import (Dict, List, Sequence, Union)
from .. utils import models
from .. core import tools as readet_tools   
# langgraph imports
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool   
from langgraph.prebuilt import create_react_agent 

 
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

