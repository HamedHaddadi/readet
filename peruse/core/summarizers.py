# ########################################################## #
# All agents and LangGraph based agents for summary creation #
# ########################################################## #
from typing import  List, TypeVar, Literal, Dict, Any
from collections.abc import Callable 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_core.prompts import PromptTemplate 
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain 
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import pdf 
from langchain.chains.llm import LLMChain 
from langchain.chains.combine_documents.stuff import StuffDocumentsChain 
from .. utils import prompts, models, docs

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
	"""
	def __init__(self, chat_model: str = 'openai-gpt-4o-mini', temperature: int = 0):
		llm = models.configure_chat_model(chat_model, temperature = temperature)
		template = PLAIN_SUMMARY_PROMPT 
		prompt = PromptTemplate.from_template(template)
		self.chain = (prompt | llm)

	def __call__(self, pdf_file: str | List[str], document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
					splitter: Literal['recursive', 'token'] | None = 'recursive', 
						splitter_kwargs: Dict[str, Any] = {}) -> str:
		document = docs.doc_from_pdf_files(pdf_file, document_loader = document_loader, 
									 	splitter = splitter, splitter_kwargs = splitter_kwargs)
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
	documents = docs.load_and_split_pdf(pdf_file)
	if documents is not None:
		llm = models.configure_chat_model(chat_model, temperature = temperature)
		chain = load_summarize_chain(llm, chain_type = 'refine')
		summary = chain.invoke(documents)
		return summary["output_text"]
	else:
		return ""

# MapReduce Summarizer 
class MapReduceSummary(Callable):
	"""
	uses a MapReduce approach to generate a summary of the input text
	can be instantiated from pdf files or other texts
	"""
	def __init__(self, chat_model: str = 'openai-gpt-4o-mini',
				 temperature: int = 0, reduce_max_tokens: int = 4000):
		self.llm = models.configure_chat_model(chat_model, temperature = temperature)
		self.map_reduce_chain = None 
		self._configure_chains(reduce_max_tokens = reduce_max_tokens)
	
	def _configure_chains(self, reduce_max_tokens: int = 4000) -> None:
		map_template, reduce_template = prompts.TEMPLATES['map-reduce']
		map_prompt = PromptTemplate.from_template(map_template)
		reduce_prompt = PromptTemplate.from_template(reduce_template)
		map_chain = LLMChain(llm = self.llm, prompt = map_prompt)
		reduce_chain = LLMChain(llm = self.llm, prompt = reduce_prompt)
		combine_documents_chain = StuffDocumentsChain(llm_chain = reduce_chain, 
					document_variable_name = 'docs')
		reduce_documents_chain = ReduceDocumentsChain(combine_documents_chain = combine_documents_chain, 
        			collapse_documents_chain = combine_documents_chain, 
            				token_max = reduce_max_tokens)
		self.map_reduce_chain = MapReduceDocumentsChain(llm_chain = map_chain, 
    					reduce_documents_chain = reduce_documents_chain, 
    						document_variable_name = 'docs', 
    							return_intermediate_steps = False)
	
	def __call__(self, pdf_file: str | List[str], document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
					splitter: Literal['recursive', 'token'] | None = 'recursive', 
						splitter_kwargs: Dict[str, Any] = {}) -> str:
		documents = docs.doc_from_pdf_files(pdf_file, document_loader = document_loader, 
									 	splitter = splitter, splitter_kwargs = splitter_kwargs)
		return self.map_reduce_chain.invoke(documents)["output_text"] if documents is not None else ""


SUMMARIZERS = {"plain": PlainSummarizer, 
					"map-reduce": MapReduceSummary}


