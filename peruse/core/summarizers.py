# ########################################################## #
# All agents and LangGraph based agents for summary creation #
# ########################################################## #
from typing import Optional, List, TypeVar, Union
from collections.abc import Callable 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_core.prompts import PromptTemplate 
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain 
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain 
from langchain.chains.combine_documents.stuff import StuffDocumentsChain 
from .. utils import prompts, models, docs

# ####################################### #
#  		text summary tools		          #
# ####################################### #
# plain text summarizer
PS = TypeVar('PS', bound = 'PlainSummarizer')
class PlainSummarizer(Callable):
	"""
	generates summary of a text using a simple (prompt | llm) chain
	it can be instantiated from a pdf file
	"""
	def __init__(self, document: Optional[List] = None, chat_model: str = 'openai-chat', temperature: int = 0):
		llm = models.configure_chat_model(chat_model, temperature = temperature)
		template = prompts.PLAIN_SUMMARY 
		prompt = PromptTemplate.from_template(template)
		self.chain = (prompt | llm)
		self.document = document 
	
	def __call__(self, pdf: Union[str, None]) -> str:
		if self.document is None and pdf is not None:
			self.document = docs.load_and_split_pdf(pdf)
		if self.document is not None:
			return self.chain.invoke(self.document).content 
		else:
			return ""
	
	@classmethod
	def from_pdf(cls, pdf: str, chat_model: str = 'oprnai-chat', temperature: int = 0) -> PS:
		document = docs.load_and_split_pdf(pdf)
		if document is not None:
			return cls(document, chat_model = chat_model, temperature = temperature)
		else:
			return None 

# refine summarizer
def refine_pdf_summary(pdf_file: str, chat_model: str = 'openai-chat', 
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
MR = TypeVar('MR', bound = 'MapReduceSummary')
class MapReduceSummary(Callable):
	"""
	uses a MapReduce approach to generate a summary of the input text
	can be instantiated from pdf files or other texts
	"""
	def __init__(self, split_documents: List, chat_model: str = 'openai-chat',
				 temperature: int = 0, reduce_max_tokens: int = 4000):
		self.llm = models.configure_chat_model(chat_model, temperature = temperature)
		self.map_reduce_chain = None 
		self.split_documents = split_documents 
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
	
	def __call__(self) -> str:
		summary = self.map_reduce_chain.invoke(self.split_docs)
		return summary['output_text']
	
	
	@classmethod 
	def from_pdf(cls, pdf_file: str, chat_model: str = 'openai-chat', temperature: int = 0, 
						chunk_size: int = 1000, chunk_overlap: int = 0, 
									reduce_max_tokens: int = 4000) -> MR:
		documents = docs.load_and_split_pdf(pdf_file, split = False)
		if documents is not None:
			text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size = chunk_size, 
							chunk_overlap = chunk_overlap)
			split_docs = text_splitter.split_documents(documents)
			return cls(split_docs, chat_model = chat_model,
					 temperature = temperature, reduce_max_tokens = reduce_max_tokens)
		else:
			return None 

SUMMARIZERS = {"plain": PlainSummarizer, 
					"map-reduce": MapReduceSummary}


