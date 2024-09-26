# ###################################################################  #
# chatbot tools built using LangGraph for indentifying and analysing   #
#   trends 											                   #
# ###################################################################  #
from typing import (TypedDict, Annotated, List, Optional, Literal, Sequence)
from .. utils import models
from .. core import tools as peruse_tools   
from . import base 
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate 
# langgraph imports
from langgraph.graph import START, END, StateGraph, add_messages 
from langgraph.graph.graph import CompiledGraph 
from langgraph.graph.message import add_messages 
from langgraph.prebuilt import ToolNode 
from langgraph.managed import IsLastStep 
from langgraph.checkpoint.memory import MemorySaver 


class AgentState(TypedDict):
	"""The state of the agent """
	messages: Annotated[Sequence[BaseMessage], add_messages]
	is_last_step: IsLastStep 


def create_scouting_graph(max_number_of_pages: int = 10, agent_use: str = 'patents',
					document_save_path: Optional[str] = None,
						tool_error_handling: bool = False,  
						chat_model: str = 'openai-chat', 
							tools: List[str] = ['search', 'download', 'summarize'], 
								memory: str = 'device') -> CompiledGraph:
	"""
	creates a graph that works with a LLM for performing tool calling. 
	all tools are used for scouting pdf files, for example US patents. 
	Args:
		max_number_of_pages: an int for the number of pages to search by the search tool
		document_save_path: path to the save folder or directory 
		chat_model: an string which will be used to choose a LLM from 'LangChain' that supports tool calling
					the default value is 'openai-chat'
		tools: a list of strings of tool names. more tools can be appended 
				current entries are: [search, download, summarize]
	Returns:
		a compiled graph
	"""
	agent_tools = []
	for tool_name in tools:
		if tool_name == 'search':
			agent_tools.append({'patents': peruse_tools.GooglePatentTool(api_wrapper = 
							peruse_tools.PatentSearch(max_number_of_pages = max_number_of_pages))}[agent_use])
		
		elif tool_name == 'download':
			agent_tools.append(peruse_tools.PDFDownloadTool(downloader = peruse_tools.PDFDownload(save_path = document_save_path)))
		
		elif tool_name == 'summarize':
			agent_tools.append(peruse_tools.summarizer_tool)
	
	prompt = ChatPromptTemplate.from_messages(
    [("system", f"""
		you are an AI assistant for {agent_use}. use the tools provided to you. 
			"""), ("human", "{messages}")])

	model = models.configure_chat_model(chat_model, temperature = 0)
	model_with_tools = model.bind_tools(agent_tools)
	model_runnable = prompt | model_with_tools  

	if tool_error_handling:
		tool_node = base.tool_node_with_error_handling(agent_tools)
	else:
		tool_node = ToolNode(agent_tools)

	# define functions 
#	def route_tools(state: AgentState) -> Literal["tools", "end"]:
	#	if isinstance(state, list):
		#	ai_message = state[-1]
	#	elif messages := state.get("messages", []):
		#	ai_message = messages[-1]
	#	if not ai_message.tool_calls:
		#	return "end"
	#	else:
		#	return "tools"
	
	def route_tools(state):
		messages = state["messages"]
		last_message = messages[-1]
		if not last_message.tool_calls:
			return "end"
		else:
			return "continue"
	
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
									{"continue": "tools", "end": END})
	workflow.add_edge("tools", "model")
	workflow.add_edge(START, "model")
	
	if memory == 'device':
		graph = workflow.compile(checkpointer = MemorySaver())
	elif memory == 'none':
		graph = workflow.compile()

	return graph 
