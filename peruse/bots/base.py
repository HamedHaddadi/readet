# Generate classes and functions to run all chatbots #
from typing import (TypedDict, Dict, Annotated, List, Tuple, Optional, Literal, TypeVar, Sequence, Union)
from pprint import pprint 
import operator 
from pydantic import BaseModel, Field 
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
from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.tools import BaseTool   
from langgraph.prebuilt import ToolNode, create_react_agent 
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
				tool_error_handling: bool = True,  chat_model: str = 'openai-gpt-4o-mini', 
						compile: bool = True, 
							added_prompt: str = "") -> Union[StateGraph, CompiledGraph]:
	"""
	Creates a graph that works with a LLM for performing tool calling. 
	This function is similar to langgraph create_react_agent 
	Args:
		tools: a sequence of BaseTool objects or a tool object which will be added to a list
		model: LLM model (instance of LanguageModelLike) or None. This argument can be used to input
			a same LLMs for multiple agents
		agent_use: a string that shows application of the agent  
		chat_model: an string which will be used to choose a LLM from 'LangChain' that supports tool calling
					the default value is 'openai-gpt-4o-mini' 
		tool_error_handling: boolean for adding fall_back. 
	Returns:
		a compiled graph
	"""
	if not isinstance(tools, list):
		tools = [tools]

	prompt = ChatPromptTemplate.from_messages(
    	[("system", f"""
			You are an AI assistant for {agent_use}. To come up with the best answer, 
			use relevant tools that are provided for you. Avoid giving any weird answer. 
			{added_prompt}
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

# ########################### #
# Plan and execute graph 	  #
# ########################### #

class PlanExecuteState(TypedDict):
	"""The state of the plan and execute graph"""
	input: str 
	plan: List[str]
	past_steps: Annotated[List[Tuple], operator.add]
	response: str 

class Plan(BaseModel):
	"""The plan of the task"""
	steps: List[str] = Field(description = """different steps to follow in order to complete the task.
			should be in sorted order""")

class Response(BaseModel):
	"""The response of the user"""
	response: str = Field(description = "The final respone to the user")

class Action(BaseModel):
	"""The action to be taken"""
	action: Union[Response, Plan] = Field(description = """ Action to perform. If you want to respond to user, 
			use Response. If you want to use tools to come up with the final answer, use Plan""") 

class PlanExecute:
	"""
	Class for planning and executing a task
	different agents are used for planning and executing the task
	attribuites:
		executor_agent: the agent for executing the task
		planner_agent: the agent for planning the task 
		replanner_agent: the agent for replanning the task
		graph: the graph of the plan and execute
		async: boolean for async mode	
	each agent can be invoked separately to test output 
	the best way to run this tool is using run method. But the graph can be compiled and used within a larger graph. 
	
	How to use:
		plan_execute = PlanExecute(tools, executor_chat_model, other_chat_model, async_mode)
		plan_execute.run(query)
		tools: a list of tools or a single tool. 
		executor_chat_model: the chat model for the executor agent
		other_chat_model: the chat model for the planner and replanner agents
		async_mode: boolean for async mode. If true then the run method is async. 
	
	"""
	def __init__(self, tools: Union[List[BaseTool], BaseTool], 
					executor_chat_model: str = 'openai-gpt-4o', 
						other_chat_model: str = 'openai-gpt-4o', 
							async_mode: bool = True):
		
		self._async = async_mode 
		self.executor_agent = None 
		self.planner_agent = None 
		self.replanner_agent = None 
		self.graph = None 

		self._configure_executor(tools, executor_chat_model)
		self._configure_planner(other_chat_model)
		self._configure_replanner(other_chat_model)
		self._compiled = False
		self._built = False 

	@property
	def built(self) -> bool:
		return self._built 

	@built.setter
	def built(self, value: bool):
		if self._built is False and value is True:
			self._built = True 

	@property
	def compiled(self) -> bool:
		return self._compiled

	@compiled.setter
	def compiled(self, value: bool):
		if self._compiled is False and value is True:
			self._compiled = True 

	def _configure_executor(self, tools: Union[List[BaseTool], BaseTool], 
							executor_chat_model: str):
		if isinstance(tools, BaseTool):			
			tools = [tools]
		executor_prompt = ChatPromptTemplate.from_messages([
			("system", """
			You are a helpful AI assistant. 
			You are given a task and a set of tools to complete the task.
			"""), ("placeholder", "{messages}")])
		executor_llm = models.configure_chat_model(executor_chat_model, temperature = 0)
		self.executor_agent = create_react_agent(executor_llm, tools, state_modifier = executor_prompt)
	
	def _configure_planner(self, other_chat_model: str):
		planner_prompt = ChatPromptTemplate.from_messages([(
            "system", """For the given objective, come up with a simple step by step plan. 
			This plan should involve individual tasks, that if executed correctly will result in the correct answer. 
			Do not add any superfluous steps. The result of the final step should be the final answer. 
			Make sure that each step has all the information needed - do not skip steps."""),
			("placeholder", "{messages}")])
		planner_llm = models.configure_chat_model(other_chat_model, temperature = 0)
		self.planner_agent = planner_prompt | planner_llm.with_structured_output(Plan) 
	
	def _configure_replanner(self, other_chat_model: str):
		replanner_prompt = ChatPromptTemplate.from_template("""For the given objective, 
			come up with a simple step by step plan. 
			This plan should involve individual tasks, that if executed correctly will yield the correct answer. 
			Do not add any superfluous steps. The result of the final step should be the final answer. 
			Make sure that each step has all the information needed - do not skip steps.
			Your objective was this:
			{input}
			Your original plan was this:
			{plan}
			You have currently done the follow steps:
			{past_steps}
			Update your plan accordingly. If no more steps are needed and you can return to the user, 
			then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. 
			Do not return previously done steps as part of the plan.""")
		replanner_llm = models.configure_chat_model(other_chat_model, temperature = 0)
		self.replanner_agent = replanner_prompt | replanner_llm.with_structured_output(Action) 

	async def _a_execute_step(self, state: PlanExecuteState) -> Dict:
		plan = state["plan"]
		all_plans = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
		tasks = f"""For the following plans: {all_plans} \n\n You are tasked with executing step {1}, {plan[0]}"""
		response = await self.executor_agent.ainvoke({"messages": [HumanMessage(content = tasks)]})
		return {"past_steps": [(plan[0], response["messages"][-1].content)]}
	
	def _execute_step(self, state: PlanExecuteState) -> Dict:
		plan = state["plan"]
		all_plans = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
		tasks = f"""For the following plans: {all_plans} \n\n You are tasked with executing step {1}, {plan[0]}"""
		response = self.executor_agent.invoke({"messages": [HumanMessage(content = tasks)]})
		return {"past_steps": [(plan[0], response["messages"][-1].content)]}
	
	async def _a_plan_step(self, state: PlanExecuteState) -> Dict:
		plan = await self.planner_agent.ainvoke({"messages": [HumanMessage(content = state["input"])]})
		return {"plan": plan.steps}
	
	def _plan_step(self, state: PlanExecuteState) -> Dict:
		plan = self.planner_agent.invoke({"messages": [HumanMessage(content = state["input"])]})
		return {"plan": plan.steps}
	
	async def _a_replan_step(self, state: PlanExecuteState) -> Dict:
		results = await self.replanner_agent.ainvoke(state)
		if isinstance(results.action, Response):
			return {"response": results.action.response}
		else:
			return {"plan": results.action.steps}
	
	def _replan_step(self, state: PlanExecuteState) -> Dict:
		results = self.replanner_agent.invoke(state)
		if isinstance(results.action, Response):
			return {"response": results.action.response}
		else:
			return {"plan": results.action.steps}

	def _should_end(self, state: PlanExecuteState):
		return END if "response" in state and state["response"] else "agent"
	
	def _build(self) -> StateGraph:
		workflow = StateGraph(PlanExecuteState)
		workflow.add_node("agent", self._execute_step)
		workflow.add_node("planner", self._plan_step)
		workflow.add_node("replannner", self._replan_step)
		workflow.set_entry_point("planner")
		workflow.add_edge("planner", "agent")
		workflow.add_edge("agent", "replannner")
		workflow.add_conditional_edges("replannner", self._should_end, {"agent": "agent", END: END})
		return workflow 
	
	def _build_async(self) -> StateGraph:
		workflow = StateGraph(PlanExecuteState)
		workflow.add_node("agent", self._a_execute_step)
		workflow.add_node("planner", self._a_plan_step)
		workflow.add_node("replannner", self._a_replan_step)
		workflow.set_entry_point("planner")
		workflow.add_edge("planner", "agent")
		workflow.add_edge("agent", "replannner")
		workflow.add_conditional_edges("replannner", self._should_end, {"agent": "agent", END: END})
		return workflow 
	
	def build(self, compile: bool = True) -> Union[StateGraph, CompiledGraph]:
		if self._async:
			graph = self._build_async()
		else:
			graph = self._build()
		
		if compile:
			self.graph = graph.compile()
			self._compiled = True 
		else:
			self.graph = graph
		self._built = True 
		return self.graph  
	
	
	def run(self, query: str, recursion_limit: int = 50) -> None:
		config = {'recusrion_limit': recursion_limit}
		inputs = {"input": query}
	
		if not self._built:
			self.build(compile = True)

		if self._async:
			iterable = self.graph.astream(inputs, config, stream_mode = 'updates')
		else:
			iterable = self.graph.stream(inputs, config, stream_mode = 'updates')

		for event in iterable:
				for step, value in event.items():
					if step != '__end__':
						pprint(value)
			







	
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
	
	def _get_last_message(self, value: Union[List, Dict]):
		if isinstance(value, list):
			return value[-1]
		elif isinstance(value, dict):
			return value['messages'][-1]
	
	def _run_chat_mode(self, stream_mode: Literal['updates', 'values'] = 'updates'):
		while True:
			user_input = input("User: ")
			if user_input.lower() in ["quit", "exit", "q"]:
				print("Ciao!")
				break 
			# note that with stream mode = values then value will be a list
			# if stream_mode = None then it will be a dictionary  
			for event in self.runnable.stream({"messages":[("user", user_input)]}, 
						self.config, stream_mode = stream_mode):
				for value in event.values():
					last_message = self._get_last_message(value)
					if isinstance(last_message, BaseMessage):
						if last_message.content == "":
							print("Assistant: I am thinking... wait!")
						else:
							print("Assistant:", last_message.content)
	
	def _run_single_shot_mode(self, query: str, stream_mode: Literal['updates', 'values'] = 'updates') -> None:
		inputs  = {"messages": [query]}
		output = self.runnable.invoke(inputs, stream_mode = stream_mode)
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
	def from_graph(cls, graph: Union[StateGraph, CompiledGraph], thread: int = 1) -> Assist:
		"""
		accept an uncompiled graph as input
		"""
		return cls(graph, thread = thread)






