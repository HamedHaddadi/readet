
# ################################################## #
# Prebuilt functions and classes for agentic systems #
# ################################################## #
from os import PathLike  
from pydantic import BaseModel, Field   
from typing import Union, Sequence, Literal, Annotated, List, Optional
from functools import partial 
# langchain, langgraph 
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END 
from langgraph.checkpoint.memory import MemorySaver 
# peruse modules 
from . assistants import Assistant
from .. utils import models  
from .. core import tools as peruse_tools 
from . agents import ReAct 
from . components import *  


class ResearchAssistantLite:
	"""
	Assistant class the helps in search, download and summarzing pdf files
	"""
	def __init__(self, save_in: Union[str, PathLike], 
			  	search_in: Sequence[str] = ['arxiv', 'google_scholar'], max_results: int = 10) -> None:
		self.save_in = save_in 
		self.search_assistant = None 
		self.summary_assistant = None 

		self._configure_search_assistant(search_in, max_results)
		self._configure_summary_assistant()

	def _configure_search_assistant(self, search_in: Sequence[str], max_results: int) -> None:
		added_prompt = f"""you are a helpful assistant that helps in searching for scientific papers
		 	in {', '.join(search_in)} and downloading them to {self.save_in}"""
		tools = [search + '_search' for search in search_in] + ['pdf_download', 'list_files']
		search_agent = ReAct(tools, added_prompt = added_prompt, save_path = self.save_in, max_results = max_results)
		self.search_assistant = Assistant.from_graph(search_agent.runnable)
	
	def _configure_summary_assistant(self) -> None:
		added_prompt = f"""you are a helpful assistant that summarizes pdf files; remember to
			give a response using tools that you have. Do not speculate"""
		tools = ['summarize_pdfs', 'list_files']
		summary_agent = ReAct(tools, added_prompt = added_prompt, save_path = self.save_in)
		self.summary_assistant = Assistant.from_graph(summary_agent.runnable)
	
	def enable_search(self) -> None:
		self.search_assistant()
	
	def enable_summary(self) -> None:
		self.summary_assistant()

# ################################################## #
#  Research Assistant 								 #
# ################################################## #
class RAState(BaseState):
	"""
	state of the research assistant graph
	"""
	dialog_state: Annotated[List[Literal["search", "file", "summary", "rag"]],
						  update_dialog_state]

class ToSearch(BaseModel):
	"""
	transfers the work to the special assistant responsbile for search google
	scholar and arxiv and downloading technical documents
	"""
	search_query: str = Field(description = """The query to search for in google scholar and arxiv""")
	request: str = Field(description = """Any necessary follow up questions to update 
					  the list file agent should clarify before moving formward""")
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
	SEARCH_TOOLS = ["arxiv_search", "google_scholar_search", "pdf_download"]
	SEARCH_MESSAGE = [("system", """
						You are a specialized assistant for searching for technical documents on 
						google scholar and arxiv. The primary assistant delegates the task to you when they need to search for papers on arxiv
							and scholar. If user changes their mind, escalate the task back to the primary assistant. If none 
                            of your tools apply to complete the task, escalate the task back to the primary assistant
                            using 'CompleteOrEscalate' tool. Do not waste user's time. Do not make up tools or functions."""),
							  ("placeholder", "{message}")]
	FILE_MESSAGE = [("system", """
					You are a specialized assistant for listing the downloaded pdf files. 
				  The primary assistant delegates the task to you when they need to list the downloaded pdf files.
					If user changes their mind, escalate the task back to the primary assistant
						Do not waste user's time. Do not make up tools or functions. The path to the files
				  		is provided to your tool. do not make up the path"""),
							  ("placeholder", "{message}")]
	
	SUMMARY_MESSAGE = [("system", """
						You are a specialized assistant for summarizing pdf files. 
						The primary assistant delegates the task to you when they need to summarize pdf files.
						If user changes their mind, escalate the task back to the primary assistant
						Do not waste user's time. Do not make up tools or functions. The path to the files
						is provided to your tool. do not make up the path"""),
							  ("placeholder", "{message}")]

	RAG_MESSAGE = [("system", """
					You are a specialized assistant to answer user's question about pdf files. 
					The primary assistant delegates the task to you when they need to answer user's question about pdf files.
					If user changes their mind, escalate the task back to the primary assistant
					Do not waste user's time. Do not make up tools or functions."""),
							  ("placeholder", "{message}")]

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
					  primary_agent_llm: str = 'claude-3-sonnet-20240229',
					  summarizer_type: str = 'plain',
					   summary_chat_model: str = 'openai-gpt-4o-mini',
							checkpointer: Literal["memory"] = "memory") -> None:
		
		self.llm = models.configure_chat_model(special_agent_llm, temperature = 0)
		self.save_path = save_path  
		
		self.search_runnable = None
		self.search_tools = None 

		self.file_runnable = None 
		self.file_tools = None 

		self.summary_runnable = None 
		self.summary_tools = None 

		self.rag_runnable = None
		self.rag_tools = None 

		self.primary_assistant_runnable = None 
		# main runnable 
		self.runnable = None  

		self._configure_search_runnable(max_results, arxiv_page_size)
		self._configure_file_runnable()
		self._configure_summary_runnable(summarizer_type, summary_chat_model)
		self._configure_rag_runnable()
		self._configure_primary_assistant_runnable(primary_agent_llm)

		self.checkpointer = None  
		if checkpointer == "memory":
			self.checkpointer = MemorySaver()
		
	def _configure_search_runnable(self, max_results: int = None,
								 	 page_size: int = None) -> None:
		self.search_tools = [peruse_tools.get_tool(tool, tools_kwargs = {'save_path': self.save_path, 
						'max_results': max_results, 'page_size': page_size}) for tool in self.SEARCH_TOOLS]
		search_prompt = ChatPromptTemplate.from_messages(self.SEARCH_MESSAGE)
		self.search_runnable = search_prompt | self.llm.bind_tools(self.search_tools + [CompleteOrEscalate]) 

	def _configure_file_runnable(self) -> None:
		self.file_tools = [peruse_tools.get_tool("list_files", tools_kwargs = {'save_path': self.save_path, 'suffix': '.pdf'})]
		file_prompt = ChatPromptTemplate.from_messages(self.FILE_MESSAGE)
		self.file_runnable = file_prompt | self.llm.bind_tools(self.file_tools + [CompleteOrEscalate]) 

	def _configure_summary_runnable(self, summarizer_type: str, summary_chat_model: str) -> None:
		self.summary_tools = [peruse_tools.get_tool("summarize_pdfs", tools_kwargs = {'save_path': self.save_path, 'chat_model': summary_chat_model, 
														"summarizer_type": summarizer_type})]
		summary_prompt = ChatPromptTemplate.from_messages(self.SUMMARY_MESSAGE)
		self.summary_runnable = summary_prompt | self.llm.bind_tools(self.summary_tools + [CompleteOrEscalate]) 

	def _configure_rag_runnable(self) -> None:
		self.rag_tools = [peruse_tools.get_tool("rag", tools_kwargs = {'save_path': self.save_path})] 
		rag_prompt = ChatPromptTemplate.from_messages(self.RAG_MESSAGE)
		self.rag_runnable = rag_prompt | self.llm.bind_tools(self.rag_tools + [CompleteOrEscalate]) 

	def _configure_primary_assistant_runnable(self, primary_agent_llm: str) -> None:
		self.primary_llm = models.configure_chat_model(primary_agent_llm)
		primary_prompt = ChatPromptTemplate.from_messages(self.PRIMARY_MESSAGE)
		self.primary_assistant_runnable = primary_prompt | self.primary_llm.bind_tools([ToSearch, ToListFiles, ToSummarize, ToRAG])

	def build(self) -> None:
		workflow = StateGraph(RAState) 
		#primary assistant
		workflow.add_node("primary_assistant", Assistant(self.primary_assistant_runnable))
		workflow.add_edge(START, "primary_assistant")

		primary_assistant_router_options = ['enter_search', 'enter_list_files',
					'enter_summary', 'enter_rag']
		primary_assistant_router = Router([ToSearch, ToListFiles, ToSummarize, ToRAG],
									    primary_assistant_router_options)
		workflow.add_conditional_edges("primary_assistant", primary_assistant_router, 
								 primary_assistant_router_options + [END])
		
		# search entry 
		workflow.add_node("enter_search", create_entry_node("search assistant", "search"))
		workflow.add_node("search", Assistant(self.search_runnable))

		workflow.add_edge("enter_search", "search")
		workflow.add_node("search_tools", create_tool_node_with_fallback(self.search_tools))
		workflow.add_edge("search_tools", "search")
		
		search_router = RouterBinary("search_tools", cancel_message = "leave_skill")
		workflow.add_conditional_edges("search", search_router, ["leave_skill", 
								"search_tools", END])
		
		workflow.add_node("leave_skill", pop_dialog_state)
		workflow.add_edge("leave_skill", "primary_assistant")
		
		# adding the file agent 
		workflow.add_node("enter_list_files", create_entry_node("file assistant", "list_files"))
		workflow.add_node("list_files", Assistant(self.file_runnable))
		workflow.add_edge("enter_list_files", "list_files")
		workflow.add_node("list_files_tools", create_tool_node_with_fallback(self.file_tools))
		workflow.add_edge("list_files_tools", "list_files")

		file_router = RouterBinary("list_files_tools", cancel_message = "leave_skill")
		workflow.add_conditional_edges("list_files", file_router, ["leave_skill", "list_files_tools", END])

		
		# adding the summary agent 
		workflow.add_node("enter_summary", create_entry_node("summary assistant", "summarize_pdfs"))
		workflow.add_node("summarize_pdfs", Assistant(self.summary_runnable))
		workflow.add_edge("enter_summary", "summarize_pdfs")
		workflow.add_node("summarize_pdfs_tools", create_tool_node_with_fallback(self.summary_tools))
		workflow.add_edge("summarize_pdfs_tools", "summarize_pdfs")

		summary_router = RouterBinary("summarize_pdfs_tools", cancel_message = "leave_skill")
		workflow.add_conditional_edges("summarize_pdfs", summary_router, ["leave_skill", "summarize_pdfs_tools", END])

		
		# adding the rag agent 
		workflow.add_node("enter_rag", create_entry_node("rag assistant", "rag"))
		workflow.add_node("rag", Assistant(self.rag_runnable))
		workflow.add_edge("enter_rag", "rag")
		workflow.add_node("rag_tools", create_tool_node_with_fallback(self.rag_tools))
		workflow.add_edge("rag_tools", "rag")

		rag_router = RouterBinary("rag_tools", cancel_message = "leave_skill")
		workflow.add_conditional_edges("rag", rag_router, ["leave_skill", "rag_tools", END])
		self.runnable = workflow.compile(checkpointer = self.checkpointer)


	def run(self) -> None:
		pass



	



	

