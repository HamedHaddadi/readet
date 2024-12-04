# ################################################## #
# module contains components that are used in the    #
# other chatbots 								     #
# ################################################## #
from typing import Annotated, List, TypedDict, Dict, Callable, Optional 
from pydantic import BaseModel 
from langgraph.graph.message import AnyMessage, add_messages 
from langgraph.prebuilt import ToolNode, tools_condition 
from langgraph.graph import END
from langchain_core.runnables import Runnable, RunnableLambda 
from langchain_core.messages import ToolMessage

# #################### #
# graph states		   #
# #################### #
class BaseState(TypedDict):
	"""
	base state of a simple graph
	other states can inherit this class and add more keys
	"""
	messages: Annotated[List[AnyMessage], add_messages]

# ################## #
# Assistants 		 #
# ################## #
class Assistant:
	"""
	Plain assistant that can be used to create an assistant using a runnable
	runnable can be chain of prompt | llm or agent 
	"""
	def __init__(self, runnable: Runnable):
		self.runnable = runnable 

	def __call__(self, state: BaseState):
		while True:
			result = self.runnable.invoke(state)
			if not result.tool_calls and (not result.content or isinstance(result.content, list) and not result.content[0].get("text")):
				messages = state['messages'] + [("user", "Respond with a real output.")]
				state = {**state, 'messages': messages}
			else:
				break 
		return {"messages": result}

# ##################  #
# tool error handlers #
# ##################  #
def handle_tool_error(state) -> Dict:
	error = state.get("error")
	tool_calls = state['messages'][-1].tool_calls
	return {"messages": [ToolMessage(content = f"Error: {repr(error)} \n please fix your errors", 
				tool_call_id = tc["id"]) for tc in tool_calls]}

def create_tool_node_with_fallback(tools: List) -> Dict:
	return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key = "error")

# ################# #
# dialog handling   #
# ################# #

def pop_dialog_state(state: BaseState) -> Dict:
	"""Pop the dialog stack and return to the main assistant.
	This lets the full graph explicitly track the dialog flow and delegate control
	to specific sub-graphs.
	"""
	messages = []
	if state["messages"][-1].tool_calls:
		messages.append(ToolMessage(content = """Resuming dialog with the host assistant. 
			Please reflect on the past conversation and assist the user as needed.""", tool_call_id = state["messages"][-1].tool_calls[0]["id"]))
	return {"dialog_state": "pop", "messages": messages}

def update_dialog_stack(left: str, right: str | None) -> List[str]:
	if right is None:
		return left
	if right == 'pop':
		return left[:-1]
	return left + [right]

# ######################## #
# Entry node        	   #
# ######################## #
def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
	"""
	creates and entry node; it needs a
	CompleteOrEscalate tool to be defined in the graph ecosystem
	"""
	def entry_node(state: BaseState) -> Dict:
		tool_call_id = state['messages'][-1].tool_calls[0]['id']
		return {
			"messages": [ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the task is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id)], 
			"dialog_state": new_dialog_state}
	return entry_node


# ######################### #
# pydantic models 		 #
# ######################### #
class CompleteOrEscalate(BaseModel):
	""" A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs. """
	cancel: bool = True
	reason: str
	class Config:
		jason_schema_extra = {"example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need more time to provide more information.",
            },
            "example 4": {
                "cancel": True,
                "reason": "My tools are not sufficient to complete the task.",
            }}

# ########################### #
# Router creator 			   #
# ########################### #
class RouterMeta(type):
	"""
	Langgraph uses function names 
	objects of a callable do not have __name__ attribute
	"""
	def __new__(cls, name, bases, attributes):
		attributes.update({'__name__': attributes.get("name")})
		return super().__new__(cls, name, bases, attributes)

class Router(metaclass = RouterMeta):
	def __init__(self, tools: List[BaseModel], return_options: List[str], name: Optional[str] = None):
		self.tools = tools 
		self.return_options = return_options
		self.name = name  
	
	def __call__(self, state: BaseState):
		route = tools_condition(state)
		if route == END:
			return END 
		tool_calls = state["messages"][-1].tool_calls
		if tool_calls:
			for tool, option in zip(self.tools, self.return_options):
				if tool_calls[0]["name"] == tool.__name__:
					return option 
		raise ValueError(f"invalid route")

class ToolRouter(metaclass = RouterMeta):
	def __init__(self,  continue_message: str, router_tool: BaseModel = CompleteOrEscalate,
			cancel_message: str = "leave_skill", name: Optional[str] = None):		
		self.router_tool = router_tool
		self.continue_message = continue_message 
		self.cancel_message = cancel_message 
		self.name = name 

	def __call__(self, state: BaseState):
		route = tools_condition(state)
		if route == END:
			return END 
		tool_calls = state["messages"][-1].tool_calls
		did_cancel = any(tc.get("name") == self.router_tool.__name__ for tc in tool_calls)
		if did_cancel:
			return self.cancel_message
		return self.continue_message
		














