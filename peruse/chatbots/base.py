# Generate classes and functions to run all chatbots #
from typing import (TypedDict, Dict, Annotated, List, Optional, Literal, TypeVar, Sequence, Union)
from pprint import pprint 
from .. utils import models
from .. core import tools as peruse_tools   
from . import base 
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate 
# langgraph imports
from langgraph.graph import START, END, StateGraph, add_messages 
from langgraph.graph.message import add_messages 
from langgraph.managed import IsLastStep 
from langchain_core.language_models import LanguageModelLike 
from langchain_core.messages import ToolMessage, BaseMessage 
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.tools import BaseTool   
from langgraph.prebuilt import ToolNode 
from langgraph.graph.graph import CompiledGraph 
from langgraph.checkpoint.memory import MemorySaver 
from collections.abc import Callable 

 
def handle_tool_errors(state: TypedDict) -> Dict:
	"""
	function to handle tool errors during tool execution
	Args:
		state: the current state of the AI agent which contains AI Messages and tool_calls. 
	Returns:
		dict: A dictionary containing error messages for each tool  
	"""
	error = state.get("error")
	tool_calls = state["messages"][-1].tool_calls 

	return {"messages": [ToolMessage(content = f"Error: {repr(error)} \n Take steps to fix errors!", 
										tool_call_id = tc["id"]) for tc in tool_calls]}

def tool_node_with_error_handling(tools: List) -> ToolNode:
	"""
	function to generate tool nodes with error handling mechanism
	Args:
		tools: the list of tools
	Returns:
		Toolnode with error handling mechanism
	""" 
	return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_errors)], exception_key = "error")

# ########################### #
# ReAct graph 				  #
# ########################## #

class AgentState(TypedDict):
	"""The state of the agent """
	messages: Annotated[List, add_messages]
	is_last_step: IsLastStep

def create_react_graph(tools: Union[Sequence[BaseTool], BaseTool], agent_use: str,  model: Optional[LanguageModelLike] = None, 
				tool_error_handling: bool = True,  chat_model: str = 'openai-chat', 
						compile: bool = True) -> Union[StateGraph, CompiledGraph]:
	"""
	Creates a graph that works with a LLM for performing tool calling. 
	This function is similar to langgraph create_react_agent 
	Args:
		tools: a sequence of BaseTool objects or a tool object which will be added to a list
		model: LLM model (instance of LanguageModelLike) or None. This argument can be used to input
			a same LLMs for multiple agents
		agent_use: a string that shows application of the agent  
		chat_model: an string which will be used to choose a LLM from 'LangChain' that supports tool calling
					the default value is 'openai-chat' 
		tool_error_handling: boolean for adding fall_back. 
	Returns:
		a compiled graph
	"""
	if not isinstance(tools, list):
		tools = [tools]

	prompt = ChatPromptTemplate.from_messages(
    	[("system", f"""
			you are an AI assistant for {agent_use}. To come up with the best answer, 
			use relevant tools that are provided for you. Avoid giving any weird answer. 
			"""), ("human", "{messages}")])

	if model is None:
		model = models.configure_chat_model(chat_model, temperature = 0)
		model_runnable = prompt | model.bind_tools(tools)
	else:
		model_runnable = prompt | model.bind_tools(tools)
	
	if tool_error_handling:
		tool_node = base.tool_node_with_error_handling(tools)
	else:
		tool_node = ToolNode(tools)

	# define functions 
	def route_tools(state: AgentState) -> Literal["tools", "end"]:
		if isinstance(state, list):
			ai_message = state[-1]
		elif messages := state.get("messages", []):
			ai_message = messages[-1]
		if not ai_message.tool_calls:
			return "end"
		else:
			return "tools"
	
	def call_model(state: AgentState):
		response = model_runnable.invoke(state["messages"])
		if state["is_last_step"] and response.tool_calls:
			return {"messages": [AIMessage(id = response.id, content = "sorry we need more steps")]}
		return {"messages": [response]}
	
	# define the graph 
	workflow = StateGraph(AgentState)
	workflow.add_node("model", call_model)
	workflow.add_node("tools", tool_node)
	workflow.set_entry_point("model")
	workflow.add_conditional_edges("model", route_tools, 
									{"tools": "tools", "end": END})
	workflow.add_edge("tools", "model")
	workflow.add_edge(START, "model")
	if compile:
		return workflow.compile()
	else:
		return workflow 
	
# General assistante class to run a graph 
Assist = TypeVar('Assist', bound = "Assistant")
class Assistant(Callable):
	"""
	Assistant class 
	runnable: can be a compiled graph or a chain or agent 
	thread: integer; id of a thread for adding memory to the conversation
	"""
	def __init__(self, runnable: Runnable, thread: int = 1, memory: Optional[Literal["device"]] = None):
		
		self.config = None 
		if isinstance(runnable, CompiledGraph):
			self.runnable = runnable 
		elif isinstance(runnable, StateGraph):		
			memory = {'device': MemorySaver(), None: None}[memory]
			self.config = None 
			if memory is not None:
				self.config = {"configurable": {"thread_id": thread}}
			self.runnable = runnable.compile(checkpointer = memory) 
	
	def _run_chat_mode(self):
		while True:
			user_input = input("User: ")
			if user_input.lower() in ["quit", "exit", "q"]:
				print("Ciao!")
				break 
			# note that with stream mode = values then value will be a list
			# if stream_mode = None then it will be a dictionary  
			for event in self.runnable.stream({"messages":[("user", user_input)]}, 
						self.config, stream_mode = 'values'):
				for value in event.values():
					if isinstance(value[-1], BaseMessage):
						if value[-1].content == "":
							print("Assistant: I am thinking... wait!")
						else:
							print("Assistant:", value[-1].content)
	
	def _run_single_shot_mode(self, query: str) -> None:
		inputs  = {"messages": [query]}
		output = self.runnable.invoke(inputs, stream_mode = 'values')
		pprint(output["messages"][-1].content)

	def __call__(self, chat: bool = True, query: Optional[str] = None) -> None:
		if not chat and query is None:
			raise ValueError("chat mode is False so a query is needed! ") 

		if chat:
			self._run_chat_mode()
		else:
			self._run_single_shot_mode(query)
	
	@classmethod
	def from_graph(cls, graph: Union[StateGraph, CompiledGraph], thread: int = 1) -> Assist:
		"""
		accept an uncompiled graph as input
		"""
		return cls(graph, thread = thread)






