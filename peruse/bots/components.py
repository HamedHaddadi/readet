# ################################################## #
# module contains components that are used in the    #
# other chatbots 								     #
# ################################################## #
from typing import Annotated, List, TypedDict, Dict 
from pydantic import BaseModel 
from langgraph.graph.message import AnyMessage, add_messages 
from langgraph.prebuilt import ToolNode 
from langchain_core.runnables import Runnable, RunnableLambda 
from langchain_core.messages import ToolMessage
from collections.abc import Callable

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
class PlainAssistant(Callable):
	"""
	Plain assistant that can be used to create an assistant using a runnable
	runnable can be chain of prompt | llm or agent 
	"""
	def __init__(self, runnable: Runnable):
		self.runnable = runnable 

	def __call__(self, state: BaseState):
		while True:
			state = {**state}
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
def update_dialog_state(left: List[str], right: str) -> List[str]:
	if right is None:
		return left
	if right == 'pop':
		return left[:-1]
	else:
		return left + [right]

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














