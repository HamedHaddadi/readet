# ########################################################## #
# All agents and LangGraph based agents for summary creation #
# ########################################################## #
import operator 
from typing import  List, Literal, Dict, TypedDict, Annotated  
from collections.abc import Callable 
from langchain.chains.combine_documents.reduce import (collapse_docs, split_list_of_docs)
from langchain_core.documents import Document 	
from langchain_core.prompts import PromptTemplate 
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from langgraph.constants import Send
from .. utils import models, docs

# ####################################### #
#  		text summary tools		          #
# ####################################### #
# plain text summarizer
PLAIN_SUMMARY_PROMPT = """
	you will be provided with a text. Write a concise summary and include the main points.
		{text} """
class PlainSummarizer(Callable):
	"""
	uses a simple (prompt | llm) chain to generate a summary of a text
	parameters:
		chat_model: chat model to be used
		temperature: temperature to be used
	"""
	def __init__(self, chat_model: str = 'openai-gpt-4o-mini', temperature: int = 0):
		llm = models.configure_chat_model(chat_model, temperature = temperature)
		template = PLAIN_SUMMARY_PROMPT 
		prompt = PromptTemplate.from_template(template)
		self.chain = (prompt | llm)

	def __call__(self, pdf_file: str | List[str], document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
					splitter: Literal['recursive', 'token'] | None = 'recursive', 
						chunk_size: int = 2000, chunk_overlap: int = 200) -> str:
		"""
		no streaming is supported for this summarizer
		input parameters:
			pdf_file: pdf file to be summarized
			document_loader: document loader to be used
			splitter: splitter to be used 
			chunk_size: chunk size to be used
			chunk_overlap: chunk overlap to be used
		"""
		document = docs.doc_from_pdf_files(pdf_file, document_loader = document_loader, splitter = splitter, 
									 	chunk_size = chunk_size, chunk_overlap = chunk_overlap)
		if document is not None:
			return self.chain.invoke(document).content 
		else:
			return ""
	
# refine summarizer
def refine_pdf_summary(pdf_file: str, chat_model: str = 'openai-gpt-4o-mini', 
				temperature = 0) -> str:
	"""
	uses predefined load_summarize_chain of LangChain to summarize a pdf file
	"""
	documents = docs.doc_from_pdf_files(pdf_file)
	if documents is not None:
		llm = models.configure_chat_model(chat_model, temperature = temperature)
		chain = load_summarize_chain(llm, chain_type = 'refine')
		summary = chain.invoke(documents)
		return summary["output_text"]
	else:
		return ""

# MapReduce Summarizer 
MAP_PROMPT = """write a concise summary of the following text: {context}"""
REDUCE_PROMPT = """ The following is a set of summaries: {docs}. Take them and 
		distill it into a final comsolidated summary """

class GraphState(TypedDict):
	"""
	The state of the map reduce graph
	"""
	contents: List[str]
	summaries: Annotated[List, operator.add]
	collapsed_summaries: List[Document]
	final_summary: str

class SummaryState(TypedDict):
	content: str

class MapReduceSummarizer(Callable):
	"""
	uses a MapReduce approach to generate a summary of the input text
	can be instantiated from pdf files or other texts
	"""
	def __init__(self, chat_model: str = 'openai-gpt-4o-mini', temperature: int = 0, 
			  	max_token: int = 1000):
		self.llm = models.configure_chat_model(chat_model, temperature = temperature)
		self.map_chain = (PromptTemplate.from_template(MAP_PROMPT) | self.llm | StrOutputParser())
		self.reduce_chain = (PromptTemplate.from_template(REDUCE_PROMPT) | self.llm | StrOutputParser())
		self.max_token = max_token
		self.runnable = None 
		self.built = False 
	
	def length_function(self, documents: List[Document]) -> int:
		return sum([self.llm.get_num_tokens(doc.page_content) for doc in documents])
	
	def generate_summary(self, state: SummaryState) -> Dict[str, List]:
		response = self.map_chain.invoke(state["content"])
		return {"summaries": [response]}
	
	def map_summaries(self, state: GraphState) -> List:
		return [Send("generate_summary", {"content": content}) for content in state["contents"]]
	
	def collect_summaries(self, state: GraphState) -> Dict:
		return {"collapsed_summaries": [Document(summary) for summary in state["summaries"]]}
	
	def collapse_summaries(self, state: GraphState) -> Dict:
		doc_lists = split_list_of_docs(state["collapsed_summaries"], self.length_function, self.max_token)
		results = []
		for doc_list in doc_lists:
			results.append(collapse_docs(doc_list, self.reduce_chain.invoke))
		return {"collapsed_summaries": results}
	
	def should_collapse(self, state: GraphState) -> Literal["collapse_summaries", "generate_final_summary"]:
		num_tokens = self.length_function(state["collapsed_summaries"])
		if num_tokens > self.max_token:
			return "collapse_summaries"
		else:
			return "generate_final_summary"
	
	def generate_final_summary(self, state: GraphState) -> Dict:
		summary = self.reduce_chain.invoke(state["collapsed_summaries"])
		return {"final_summary": summary}
	
	def build(self) -> None:
		graph = StateGraph(GraphState)
		graph.add_node("generate_summary", self.generate_summary)
		graph.add_node("collect_summaries", self.collect_summaries)
		graph.add_node("collapse_summaries", self.collapse_summaries)
		graph.add_node("generate_final_summary", self.generate_final_summary)

		graph.add_conditional_edges(START, self.map_summaries, ["generate_summary"])
		graph.add_edge("generate_summary", "collect_summaries")
		graph.add_conditional_edges("collect_summaries", self.should_collapse)
		graph.add_conditional_edges("collapse_summaries", self.should_collapse)
		graph.add_edge("generate_final_summary", END)
		self.runnable = graph.compile()
		self.built = True 
	
	def __call__(self, pdf_file: str | List[str], document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
					splitter: Literal['recursive', 'token'] | None = 'recursive', 
						chunk_size: int = 2000, chunk_overlap: int = 200) -> str:
		"""
		no streaming is supported for this summarizer
		"""
		if not self.built:
			self.build()
		documents = docs.doc_from_pdf_files(pdf_file, document_loader = document_loader, 
									 	splitter = splitter, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
		contents = [doc.page_content for doc in documents]
		return self.runnable.invoke({"contents": contents})["final_summary"]
			

SUMMARIZERS = {"plain": PlainSummarizer, 
					"map-reduce": MapReduceSummarizer}


