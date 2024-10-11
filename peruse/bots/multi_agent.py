# ########################### #
# Multi-agent systems         #
# ########################### #
from typing import Annotated, Sequence, Dict, TypedDict, Union
from functools import partial  
from pprint import pprint 

from langchain_core.messages import HumanMessage, BaseMessage 
from langchain_core.tools import BaseTool 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

from langgraph.graph import START, END, StateGraph, add_messages 
from langgraph.graph.graph import CompiledGraph 
from langgraph.prebuilt import create_react_agent 

from .. utils import models 
from . base import create_react_graph 


class AgentState(TypedDict):
	messages: Annotated[Sequence[BaseMessage], add_messages]
	next: str 

class Supervisor:
	"""
	The class create a Supervisor multi-agent patent and returns a Runnable (CompiledGraph)
	Args:
		agents: a dictionary of agent name and a list of tools. Example: {'research':[GooglePatentsTool]}
	model:
		str: name of the llm model 
	
	Return:
		build() method returns a Runnable(a compiled graph)
		_build() constructs that graph. each agent can be invoked by its name. 
	"""
	def __init__(self, agents: Dict[str, Sequence[BaseTool]], model: str = 'openai-chat'):
		self.agents = agents 
		self._compiled = False  
		self.llm = models.configure_chat_model(model, temperature = 0)

		self.agent_names = list(self.agents.keys())
		print(f"the agent names are {self.agent_names}")
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
			setattr(self, agent_name, create_react_agent(self.llm, tools = [tools]))
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
		return self.graph 
	
	def run(self, query: str) -> None:
		inputs = {"messages": [query]}
		output = self.graph.invoke(inputs, stream_mode = 'updates')
		pprint(output["messages"][-1].content)

	
	

		 



	

