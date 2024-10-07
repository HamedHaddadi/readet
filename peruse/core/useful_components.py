# ################################## #
# Agents built for specific purposes #
# ################################## #
from os import PathLike, path
from pathlib import Path  
from typing import Dict, Literal, List, Optional, overload, Sequence, Union
from xml.dom.minidom import Document    
import pandas as pd  
from collections.abc import Callable 
from langchain_core.pydantic_v1 import BaseModel, Field 
from langchain_core.prompts import PromptTemplate 
from langchain_community.callbacks import get_openai_callback 
from langchain_community.document_loaders import pdf 
from langchain_community.tools.tavily_search import TavilySearchResults 
from langgraph.prebuilt import ToolNode  
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import AIMessage 
from . import rags 
from .. utils import models, questions, prompts, docs  

# ######################################## #
#  Tavily search on extracted information  #
# This graph combined a RAG-Schema with an # 
# llm model binded with a Tavily search    #
# tool									   #
# ######################################## #
class RAGInput(BaseModel):
	question: str  

class QueryPDFAndSearch(Callable):
	"""
	queries a pdf document and stores the results in a schema.
	Then uses a ReAct Tavily search agent to search the internet for 
	more information about retrieved keywords
	"""
	def __init__(self, pdf_file: str, schemas: str = 'general',
					added_message: str = "",  chunk_size: int = 2000, 
					chunk_overlap: int = 150, chat_model: str = 'openai-chat', 
							embedding_model: str = 'openai-embedding', 
									rag_prompt: str = 'schema-rag', 
										max_search_results: int = 20, 
										temperature: int = 0):
		
		rag_chain = rags.RAGSinglePDF(chunk_size = chunk_size, chunk_overlap = chunk_overlap, 
							chat_model = chat_model, embedding_model = embedding_model, 
								prompt = rag_prompt, temperature = temperature)(pdf_file)
		schema_chain =  rags.extract_schema_plain(chat_model = chat_model, 
								temperature = temperature, schemas = schemas)
		self.rag_schema_chain = (rag_chain | schema_chain)
		#. build the search agent
		search = TavilySearchResults(max_results = max_search_results)
		llm = models.configure_chat_model(chat_model, temperature = temperature)
		self.llm_model = llm.bind_tools([search])
		self.message_to_model = added_message 
		self._tool_node = ToolNode([search])
		self._configure_graph()
	
	# methods called by graph nodes 
	def _call_rag(self, input: MessagesState) -> Dict[Literal["messages"], List[str]]:
		messages = input["messages"][0].content
		response = self.rag_schema_chain.invoke(messages)
		return {"messages": response.results}
	
	def _call_model(self, state: MessagesState) -> Dict[Literal["messages"], AIMessage]:
		messages = state["messages"]
		if len(messages) == 1:
			response = self.llm_model.invoke(f"search for more information about {','.join([res for res in messages[0]])} and make sure to {self.message_to_model}")
		else:
			response = self.llm_model.invoke(messages)
		return {"messages": response}
	
	def _should_continue(self, state: MessagesState) -> Literal["tools", "__end__"]:
		messages = state["messages"]
		last_message = messages[-1]
		if last_message.tool_calls:
			return "tools"
		return END

	def _configure_graph(self):
		flow = StateGraph(MessagesState)
		flow.add_node("rag", self._call_rag)
		flow.add_node("tools", self._tool_node)
		flow.add_node("model", self._call_model)

		flow.add_edge(START, "rag")
		flow.add_edge("rag", "model")
		flow.add_conditional_edges("model", self._should_continue)
		flow.add_edge("tools", "model")
		self.graph = flow.compile()
	
	def __call__(self, question: str):
		for chunk in self.graph.stream({"messages":question}):
			if "model" in chunk.keys():
				chunk["model"]["messages"].pretty_print()

# #### Extract Schema from PDF file #### #
class SchemaFromPDF(Callable):
	"""
	class to combine RAG and Extractor chains to extract structured information from 	
		texts. Structure is an schema 
	"""
	@overload
	def __init__(self, schemas: str,
		 		rag: str = 'single-pdf', extractor: str = 'plain',
					 chunk_size: int = 6000, chunk_overlap: int = 300,
					  	chat_model: str = 'openai-chat', 
							embedding_model: str = 'openai-embedding', temperature: int = 0, 
								replacements: Optional[Dict[str, str]] = None, 
									save_path: Optional[str] = None) -> None:
		...

	def __init__(self, schemas: str, rag: str = 'single-pdf',
	 				extractor: str = 'plain', chunk_size: int = 6000, 
					 	chunk_overlap: int = 300, chat_model: str = 'openai-chat', 
						 	embedding_model: str = 'openai-embedding', temperature: int = 0, 
							 		remplacements: Optional[Dict[str, str]] = None, 
									 	save_path: Optional[str] = None) -> None:
		
		schemas = schemas.split(',')
		rag = rags.RAGS[rag]
		Extract = rags.EXTRACTORS[extractor]
		extract_keys = self._get_arg_keys(Extract)

		rag_inputs = {key:value for key,value in locals().items() if key in rag.init_keys}
		ext_inputs = {key:value for key,value in locals().items() if key in extract_keys}
		
		self.rag_object = rag(**rag_inputs)
		self.extract_chains = Extract(**ext_inputs)
		self.questions = schemas 
		self.save_path = save_path 
	
	def __call__(self, pdf_file: str) -> None:
		"""
		asks questions from the extraction chain. Takes the answer and appends the __dict__ to
			the csv file if csv_files dictionary is not None. 
			Else it uses save_path to save the dictionary into a new csv file with the name of the schema.
			For example if the schema name is suspensions, __call__ will create suspensions.csv
		schemas are ['suspensions'] 
		:param pdf_file: pdf file to be queries for schemas 
		:type pdf_file : str
		"""
		rag_chain = self.rag_object(pdf_file)
		for schema,ext_chain in self.extract_chains.items():
			extract_chain = (rag_chain | ext_chain)
			with get_openai_callback() as cb:
				info = extract_chain.invoke(questions.QUESTIONS[schema])
				print(cb)
			info_df = pd.DataFrame.from_dict([info.__dict__])
			info_df.index.name = 'id'
			csv_file = path.join(self.save_path, schema + '.csv')
			if path.exists(csv_file):
				mode = 'a' 
				header = False  
			else:
				mode = 'w'
				header = True  
			info_df.to_csv(csv_file, header = header, index = True, sep = ',', mode = mode, na_rep = 'NULL')
			
	@staticmethod
	def _get_arg_keys(callable: Callable) -> list:
		args = callable.__code__.co_varnames[:callable.__code__.co_argcount]
		return [arg for arg in args if arg not in ['self', 'cls']]

# ### Runs a tool and saves the results in structured format ### #
class ToStructured:
	"""
	a chain to structure a text into a schema
	schema: a BaseModel pydantic model 
	self.run return a dictionary 
	"""
	def __init__(self, schema: BaseModel, model: str = 'openai-chat'):
		llm = models.configure_chat_model(model, temperature = 0)
		template = prompts.TEMPLATES['to-structured']
		prompt = PromptTemplate.from_template(template)
		self.runnable = (prompt | llm.with_structured_output(schema))
	
	def run(self, text: str) -> Dict:
		outputs = self.runnable.invoke(text)
		return outputs.dict()

