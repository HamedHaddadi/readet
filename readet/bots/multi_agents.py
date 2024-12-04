# ########################### #
# Multi-agent systems         #
# ########################### #
from typing import Annotated, Sequence, Dict, TypedDict, Union
from functools import partial  

from langchain_core.messages import HumanMessage, BaseMessage 
from langchain_core.tools import BaseTool 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

from langgraph.graph import START, END, StateGraph, add_messages 
from langgraph.graph.graph import CompiledGraph 
from langgraph.prebuilt import create_react_agent 

from .. utils import models 
from ..core.tools import get_tool  

# ########################################################### #
class AgentState(TypedDict):
	messages: Annotated[Sequence[BaseMessage], add_messages]
	next: str 

# ########################################################### #
class Supervisor:
	"""
	The class create a Supervisor multi-agent and returns a Runnable (CompiledGraph)
	Args:
		agents: a dictionary of agent name and a list of tools. Example: {'research':[GooglePatentsTool]}
	model:
		str: name of the llm model 
	
	Return:
		build() method returns a Runnable(a compiled graph)
	Run or __call__ method execute the graph
	NOTE: this agent can not be run in a chatbot 
	"""
	def __init__(self, agents: Dict[str, Union[Sequence[BaseTool], BaseTool, str, Sequence[str]]], model: str = 'openai-gpt-4o-mini', 
			  **agents_kwargs):
		
		self.agents = None 
		self._configured_agents(agents, agents_kwargs)
		self._compiled = False 
		self._built = False 
		self.llm = models.configure_chat_model(model, temperature = 0)

		self.agent_names = list(self.agents.keys())
		system_prompt1 = f"""You are a supervisor tasked with managing a conversation between the
          	following workers: {self.agent_names}. Given the following user request,
         	respond with the worker to act next. Each worker will perform a
        	task and respond with their results and status. When finished,
        	respond with FINISH."""

		self.options = self.agent_names + ['FINISH']
		
		system_prompt2 = f""" Given the conversation above, who should act next?
			Or, should we FINISH? select one of: {self.options}"""
		
		self.router_schema = {'properties': {'next': {'enum': self.options,
   				'title': 'Next',
   				'type': 'string'}},
 				'required': ['next'],
    			'description': 'options that are available for routing the response by the supervisor',
 				'title': 'RouteResponse',
 				'type': 'object'}

		self.supervisor_prompt = ChatPromptTemplate.from_messages(
			[("system", system_prompt1), MessagesPlaceholder(variable_name = "messages"), 
				("system", system_prompt2)]).partial(options = str(self.options),
						 agent_names = ', '.join(self.agent_names))
	
	def _configure_agents(self, agents: Dict[str, Union[Sequence[BaseTool], BaseTool, str, Sequence[str]]], agents_kwargs: Dict):
		agents = {}
		for agent_name, tools in agents.items():
			if isinstance(tools, str):
				agents[agent_name] = [get_tool(tools, agents_kwargs)]
			elif isinstance(tools, Sequence) and all(isinstance(tool, str) for tool in tools):
				agents[agent_name] = [get_tool(tool, agents_kwargs) for tool in tools]
			elif isinstance(tools, Sequence) and all(isinstance(tool, BaseTool) for tool in tools):
				agents[agent_name] = tools 
			elif isinstance(tools, BaseTool):
				agents[agent_name] = [tools]
			else:
				raise ValueError(f"Invalid tools input for agent {agent_name}")
		self.agents = agents 
				
	@property 
	def built(self):
		return self._built 
	
	@built.setter 
	def built(self, status: bool):
		if self._built is False and status is True:
			self._built = True 
	
	@property 
	def compiled(self):
		return self._compiled 
	
	@compiled.setter 
	def compiled(self, status: bool):
		if self._compiled is False and status is True:
			self._compiled = True 
		
	def _supervisor_agent(self, state):
		chain = (self.supervisor_prompt | self.llm.with_structured_output(self.router_schema))
		return chain.invoke(state)

	@staticmethod
	def _agent_node(state, agent: CompiledGraph, name: str) -> Dict:
		result = agent.invoke(state)
		return {"messages": [HumanMessage(content = result["messages"][-1].content, name = name)]}
	
	def _route_agents(self, state):
		return state['next']
	
	def build(self, compile = True) -> Union[CompiledGraph, StateGraph]:
		workflow = StateGraph(AgentState)
		workflow.add_node("supervisor", self._supervisor_agent)
		for agent_name, tools in self.agents.items():
			setattr(self, agent_name, create_react_agent(self.llm, tools))
			node = partial(self._agent_node, agent = getattr(self, agent_name), name = agent_name)
			workflow.add_node(agent_name, node)
		for agent_name in self.agent_names:
			workflow.add_edge(agent_name, "supervisor")
		conditional_map = {k:k for k in self.agent_names}
		conditional_map["FINISH"] = END 
		workflow.add_conditional_edges("supervisor", self._route_agents, conditional_map)
		workflow.add_edge(START, "supervisor")
		if compile:
			self.graph = workflow.compile()
			self.compiled = True  
		else:
			self.graph = workflow
		self.built = True 
		return self.graph 
	
	def run(self, query: str) -> None:
		if not self.built:
			self.build(compile = True)
		inputs = {"messages": [query]}
		output = self.graph.invoke(inputs, stream_mode = 'updates')
		print('finished execution of the graph')
	
	def __call__(self, query: str) -> None:
		self.run(query)

	
	

		 



	

