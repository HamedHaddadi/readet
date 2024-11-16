# ################################################## #
# chat bot assistants 								 #
# this module contains tools to run graphs           #
# in chatbot format 		 						 #
# ################################################## #
from typing import Literal, Union, List, Dict, Optional, TypeVar 
from pprint import pprint 
from collections.abc import Callable 
from langgraph.graph import StateGraph 
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable


# General class to run a graph in chatbot format 
Chat = TypeVar('Chat', bound = "SimpleChat")
class SimpleChat(Callable):
	"""
	Assistant class 
	runnable: can be a compiled graph or a chain or agent 
	thread: integer; id of a thread for adding memory to the conversation
	"""
	def __init__(self, runnable: Runnable, thread: int = 1,
			  	 memory: Optional[Literal["device"]] = None, 
				 	config: Optional[Dict] = None):
		
		if config is not None:
			self.config = config 
		else:
			self.config = {"configurable": {"thread_id": thread}}

		if isinstance(runnable, CompiledGraph):
			self.runnable = runnable 
		elif isinstance(runnable, StateGraph):		
			memory = {'device': MemorySaver(), None: None}[memory]
			self.runnable = runnable.compile(checkpointer = memory) 
	
	def _get_last_message(self, value: Union[List, Dict]):
		if isinstance(value, list):
			return value[-1]
		elif isinstance(value, dict):
			messages = value['messages']
			if isinstance(messages, list):
				return messages[-1]
			else:
				return messages
	
	def _run_chat_mode(self, stream_mode: Literal['updates', 'values'] = 'updates'):
		while True:
			user_input = input("User: ")
			if user_input.lower() in ['exit', 'quit', 'bye']:
				print("Ciao!")
				break 
			# note that with stream mode = values then value will be a list
			# if stream_mode = None then it will be a dictionary  
			for event in self.runnable.stream({"messages":[("user", user_input)]}, 
						self.config, stream_mode = stream_mode):
				for value in event.values():
					last_message = self._get_last_message(value)
					if isinstance(last_message, AIMessage):
						if last_message.content == "":
							print("Assistant: I am working... wait!")
						elif isinstance(last_message.content, str):
							print("Assistant:", last_message.content)
						elif isinstance(last_message.content, list):
							print("Assistant:", last_message.content[0]['text'])
	
	def _run_single_shot_mode(self, query: str, stream_mode: Literal['updates', 'values'] = 'updates') -> None:
		inputs  = {"messages": [query]}
		output = self.runnable.invoke(inputs, config = None)
		pprint(output["messages"][-1].content)

	def __call__(self, chat: bool = True,
			  	 query: Optional[str] = None, stream_mode: Literal['updates', 'values'] = 'updates') -> None:
		if not chat and query is None:
			raise ValueError("chat mode is False so a query is needed! ") 

		if chat:
			self._run_chat_mode(stream_mode = stream_mode)
		else:
			self._run_single_shot_mode(query, stream_mode = stream_mode)
	
	@classmethod
	def from_graph(cls, graph: Union[StateGraph, CompiledGraph], thread: int = 1) -> Chat:
		"""
		accept an uncompiled graph as input
		"""
		return cls(graph, thread = thread)