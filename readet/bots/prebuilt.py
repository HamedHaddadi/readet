
# ################################################## #
# Prebuilt functions and classes for agentic systems #
# ################################################## #
from pydantic import BaseModel, Field   
from typing import Literal, Annotated, List
# langchain, langgraph 
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END 
from langgraph.checkpoint.memory import MemorySaver 
# readet modules 
from .. utils import models  
from .. core import tools as readet_tools 
from . agents import ReAct 
from . components import *  


# ################################################## #
#  Research Assistant 								 #
# ################################################## #
class RAState(BaseState):
	"""
	state of the research assistant graph
	"""
	dialog_state: Annotated[List[Literal["primary_assistant", "search", "list_files", "summary", "rag"]],
						  update_dialog_stack]

class ToSearch(BaseModel):
	"""
	transfers the work to the special assistant responsbile for searching google
	scholar and arxiv and downloading technical documents
	"""
	search_query: str = Field(description = """The query to search for in google scholar and arxiv""")
	request: str = Field(description = """Any necessary follow up questions to update 
					  the search before moving formward""")
	class Config:
		jason_schema_extra = {"example": {"search_query": "search papers on fluid dynamics",
									 "request": "download papers with the pdf link"}}

class ToListFiles(BaseModel):
	"""
	transfers the work to the special assistant responsible for listing the downloaded pdf files
	"""
	request: str = Field(description = """Any additional information by user that helps the special assistant""")

class ToSummarize(BaseModel):
	"""
	transfers the work to the special assistant responsible for summarizing the pdf files
	"""
	request: str = Field(description = """ the title of the paper to summarize or 'all' to summarize all papers""")
	class Config:
		jason_schema_extra = {"example": {"request": "summarize all papers"}}

class ToRAG(BaseModel):
	"""
	transfers the work to the special assistant responsible for asking questions about the pdf files
	using RAGs
	"""
	query: str = Field(description = """a user question""")
	request: str = Field(description = """Any additional information by user that helps the special assistant""")
	class Config:
		jason_schema_extra = {"example": {"query": "what is the main idea of the paper?",
									 "request": "give a concise answer"}}

class ResearchAssistant:
	"""
	Research assistant class that uses 
		multiple subgraphs to search, download, summarize and query pdf files
	the number of subgraphs depends on the tools that are used by the agent.
	The primary assistant automatically handles the conversation between agents.
	Note that the number of agents is fixed:
		search agent: uses arxiv and scholar to search for papers and downloads them
		file agent: lists the downloaded pdfs 
		summary agent: summarizes the downloaded pdfs
		rag agent: queries the pdfs using Retrieval Augmented Generation 
	"""
	SEARCH_TOOL_NAMES = ["arxiv_search", "google_scholar_search", "pdf_download"]
	SEARCH_MESSAGE = [("system", """
						You are a specialized assistant for searching technical papers on google
						scholar and arxiv and downloading them. The primary assistant delegates the task to you when they need to search for papers on arxiv
							and scholar and downloading them. Use your tools to complete the task. If user changes their mind, escalate the task back to the primary assistant. 
							Do not waste user's time. Do not make up tools or functions."""),
							  ("placeholder", "{messages}")]
	LIST_FILES_MESSAGE = [("system", """
					You are a specialized assistant for listing the files. 
				  The primary assistant delegates the task to you when they need to list the files in a folder.
					If user changes their mind, escalate the task back to the primary assistant
						Do not waste user's time. Do not make up tools or functions. The path to the files
				  		is provided to your tool. do not make up the path"""),
							  ("placeholder", "{messages}")]
	
	SUMMARY_MESSAGE = [("system", """
						You are a specialized assistant for summarizing pdf files. 
						The primary assistant delegates the task to you when they need to summarize pdf files.
						If user changes their mind, escalate the task back to the primary assistant
						Do not waste user's time. Do not make up tools or functions. The path to the files
						is provided to your tool. do not make up the path"""),
							  ("placeholder", "{messages}")]

	RAG_MESSAGE = [("system", """
					You are a specialized assistant to answer user's question about pdf files. 
					The primary assistant delegates the task to you when they need to answer user's question about pdf files.
					If user changes their mind, escalate the task back to the primary assistant
					Do not waste user's time. Do not make up tools or functions."""),
							  ("placeholder", "{messages}")]

	PRIMARY_MESSAGE = [("system", """
						You are a helpful assistant for searching technical documents on
					 	google cholar and arxiv, downloading them, listing them, summarizing them and
					 			asking user questions about them. If user asks to search documents, download
					 				documents, summarize them or ask questions about them, delegate the task 
                                                to the special agent by invoking the appropriate tool. Only specialized 
                                                    agents are aware of different tasks so do not mention them to users. """)
						,("placeholder", "{messages}")]

	def __init__(self, save_path: str, max_results: int = 10, 
			  	arxiv_page_size: int = 10,
			  		special_agent_llm: str = "openai-gpt-4o-mini", 
					  primary_agent_llm: str = 'openai-gpt-4o-mini',
					  summarizer_type: str = 'plain',
					   summary_chat_model: str = 'openai-gpt-4o-mini',
							checkpointer: Literal["memory", "sqlite"] = "sqlite") -> None:
		
		#self.llm = models.configure_chat_model(special_agent_llm, temperature = 0)
		self.save_path = save_path 
		self.spacial_agent_llm = special_agent_llm 
		
		self.search_runnable = None
		self.search_tools = None 

		self.list_files_runnable = None 
		self.list_files_tools = None 

		self.summary_runnable = None 
		self.summary_tools = None 

		self.rag_runnable = None
		self.rag_tools = None 

		self.primary_assistant_runnable = None 
		# main runnable 
		self.runnable = None  

		self._configure_search_runnable(max_results, arxiv_page_size)
		self._configure_list_files_runnable()
		self._configure_summary_runnable(summarizer_type, summary_chat_model)
		self._configure_rag_runnable()
		self._configure_primary_assistant_runnable(primary_agent_llm)

		self.checkpointer = None  
		if checkpointer == "memory":
			self.checkpointer = MemorySaver()
		
		
	def _configure_search_runnable(self, max_results: int = None,
								 	 page_size: int = None) -> None:
		llm = models.configure_chat_model(self.spacial_agent_llm, temperature = 0)
		self.search_tools = [readet_tools.get_tool(tool, {'save_path': self.save_path, 
						'max_results': max_results, 'page_size': page_size}) for tool in self.SEARCH_TOOL_NAMES]
		search_prompt = ChatPromptTemplate.from_messages(self.SEARCH_MESSAGE)
		self.search_runnable = search_prompt | llm.bind_tools(self.search_tools + [CompleteOrEscalate]) 

	def _configure_list_files_runnable(self) -> None:
		llm = models.configure_chat_model(self.spacial_agent_llm, temperature = 0)
		self.list_files_tools = [readet_tools.get_tool("list_files", tools_kwargs = {'save_path': self.save_path, 'suffix': '.pdf'})]
		list_files_prompt = ChatPromptTemplate.from_messages(self.LIST_FILES_MESSAGE)
		self.list_files_runnable = list_files_prompt | llm.bind_tools(self.list_files_tools + [CompleteOrEscalate]) 

	def _configure_summary_runnable(self, summarizer_type: str, summary_chat_model: str) -> None:
		llm = models.configure_chat_model(summary_chat_model, temperature = 0)
		self.summary_tools = [readet_tools.get_tool("summarize_pdfs", tools_kwargs = {'save_path': self.save_path, 'chat_model': summary_chat_model, 
														"summarizer_type": summarizer_type})]
		summary_prompt = ChatPromptTemplate.from_messages(self.SUMMARY_MESSAGE)
		self.summary_runnable = summary_prompt | llm.bind_tools(self.summary_tools + [CompleteOrEscalate]) 

	def _configure_rag_runnable(self) -> None:
		llm = models.configure_chat_model(self.spacial_agent_llm, temperature = 0)
		self.rag_tools = [readet_tools.get_tool("rag", tools_kwargs = {'save_path': self.save_path})] 
		rag_prompt = ChatPromptTemplate.from_messages(self.RAG_MESSAGE)
		self.rag_runnable = rag_prompt | llm.bind_tools(self.rag_tools + [CompleteOrEscalate]) 

	def _configure_primary_assistant_runnable(self, primary_agent_llm: str) -> None:
		primary_llm = models.configure_chat_model(primary_agent_llm, temperature = 0)
		primary_prompt = ChatPromptTemplate.from_messages(self.PRIMARY_MESSAGE)
		self.primary_assistant_runnable = primary_prompt | primary_llm.bind_tools([ToSearch, ToListFiles, ToSummarize, ToRAG])

	def build(self) -> None:
		workflow = StateGraph(BaseState) 
		#primary assistant
		workflow.add_node("primary_assistant", Assistant(self.primary_assistant_runnable))
		workflow.add_edge(START, "primary_assistant")

		primary_assistant_router_options = ['enter_search', 'enter_list_files',
					'enter_summary', 'enter_rag']
		primary_assistant_router = Router([ToSearch, ToListFiles, ToSummarize, ToRAG],
									    primary_assistant_router_options, name = "primary_assistant_router")
		workflow.add_conditional_edges("primary_assistant", primary_assistant_router, 
								 primary_assistant_router_options + [END])
		workflow.add_node("leave_skill", pop_dialog_state)
		workflow.add_edge("leave_skill", "primary_assistant")
		
		# search entry 
		workflow.add_node("enter_search", create_entry_node("searching on arxiv and scholar", "search"))
		workflow.add_node("search", Assistant(self.search_runnable))
		workflow.add_edge("enter_search", "search")
		workflow.add_node("search_tools", create_tool_node_with_fallback(self.search_tools))
				
		search_router = ToolRouter(continue_message="search_tools", cancel_message = "leave_skill", name = "search_router")
		workflow.add_edge("search_tools", "search")
		workflow.add_conditional_edges("search", search_router, ["search_tools", "leave_skill", END])
							
		# adding the file agent 
		workflow.add_node("enter_list_files", create_entry_node("list files assistant", "list_files"))
		workflow.add_node("list_files", Assistant(self.list_files_runnable))
		workflow.add_edge("enter_list_files", "list_files")
		workflow.add_node("list_files_tools", create_tool_node_with_fallback(self.list_files_tools))
		workflow.add_edge("list_files_tools", "list_files")

		list_files_router = ToolRouter(continue_message="list_files_tools", cancel_message = "leave_skill", name = "list_files_router")
		workflow.add_conditional_edges("list_files", list_files_router, ["list_files_tools", "leave_skill", END])

		
		# adding the summary agent 
		workflow.add_node("enter_summary", create_entry_node("summary assistant", "summary"))
		workflow.add_node("summary", Assistant(self.summary_runnable))
		workflow.add_edge("enter_summary", "summary")
		workflow.add_node("summary_tools", create_tool_node_with_fallback(self.summary_tools))
		workflow.add_edge("summary_tools", "summary")

		summary_router = ToolRouter(continue_message="summary_tools", cancel_message = "leave_skill", name = "summary_router")
		workflow.add_conditional_edges("summary", summary_router, ["summary_tools", "leave_skill", END])

		# adding the rag agent 
		workflow.add_node("enter_rag", create_entry_node("RAG assistant", "rag"))
		workflow.add_node("rag", Assistant(self.rag_runnable))
		workflow.add_edge("enter_rag", "rag")
		workflow.add_node("rag_tools", create_tool_node_with_fallback(self.rag_tools))
		workflow.add_edge("rag_tools", "rag")

		rag_router = ToolRouter(continue_message="rag_tools", cancel_message = "leave_skill", name = "rag_router")
		workflow.add_conditional_edges("rag", rag_router, ["rag_tools", "leave_skill", END])
		self.runnable = workflow.compile(checkpointer = self.checkpointer)

	def run(self) -> None:
		pass

# ################################################## #
# 	Download agent using ReAct class 				#
# ################################################## #
class Download(Callable):
	"""
	Download agent using ReAct class. searches and downloads papers from arxiv and scholar
	"""
	PROMPT = """You are a specialized assistant for searching technical papers on google
				scholar and arxiv and downloading them. use the tools provided to you to complete the task."""
	def __init__(self, save_path: str, max_results: int = 100, 
			  		chat_model: str = 'openai-gpt-4o-mini', 
						search_in: List[Literal['google_scholar', 'arxiv', 'google_patent']] = ['arxiv', 'google_scholar']) -> None:
		tools = [tool_name + '_search' for tool_name in search_in] + ['pdf_download']
		self.agent = ReAct(tools = tools, chat_model = chat_model, added_prompt = self.PROMPT,  
								max_results = max_results, save_path = save_path)
	def __call__(self, query: str) -> None:
		self.agent(query)


		
		



	



	

