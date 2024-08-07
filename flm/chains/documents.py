# ############################################################ #
# module includes chains useful for interaction with documents #
# ############################################################ #

from typing import Optional, Dict, List, TypeVar  
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter 
from langchain_chroma import Chroma 
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate  
from langchain_core.runnables.base import RunnableSequence 
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser 
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain 
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain 
from langchain.chains.combine_documents.stuff import StuffDocumentsChain 
from collections.abc import Callable 
from .. utils import prompts, models, docs
from .. utils.schemas import SCHEMAS 


# ################################### #
# 			      RAGs				  #
# ################################### #
class RAGSinglePDF:
	"""
	vanilla RAG for single PDF files
	returns a runnbale chain of ({retriever, question} | prompt | llm | parser)
	chain will be invoked by the caller to get the output text. Question will be input by the caller
	"""
	init_keys = ['pdf_file', 'chunk_size', 'chunk_overlap', 'chat_model', 
						'embedding_model', 'prompt_template', 'temperature', 'replacements']

	def __init__(self, pdf_file: str, chunk_size: int = 2000,
	 				chunk_overlap: int = 150,
					 	 chat_model: str = 'openai-chat', 
						  	embedding_model: str = 'openai-embedding',
						  	prompt_template: str = 'suspensions', 
							  	temperature: int = 0, 
								  	replacements: Optional[Dict[str, str]] = None):
		self.retriever = None 
		self.prompt_template = prompts.TEMPLATES[prompt_template]
		self.prompt = None 
		self.llm = models.configure_chat_model(chat_model, temperature = temperature)
		self.parser = StrOutputParser()
		self._configure(pdf_file, chunk_size, chunk_overlap, embedding_model, replacements)
	
	def _configure(self, pdf_file: str, chunk_size: int,
				 			chunk_overlap: int,
							 	 embedding_model: str, 
								  	replacements: Optional[Dict[str, str]] = None) -> None:
		
		loader = PyPDFLoader(pdf_file)
		doc = loader.load()
		if replacements is not None:
			doc = docs.format_doc_object(doc, replacements)
		splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
						 chunk_overlap = chunk_overlap, add_start_index = True, 
						 		separators = ["\n\n", "\n", "(?<=\. )", " ", ""])
		splits = splitter.split_documents(doc)
		embedding = models.configure_embedding_model(embedding_model)
		store = Chroma.from_documents(documents = splits, embedding = embedding)
		self.retriever = store.as_retriever()
		self.prompt = PromptTemplate.from_template(self.prompt_template)
	
	def __call__(self) -> RunnableSequence:
		return ({'context': self.retriever, 'question': RunnablePassthrough()} | self.prompt | self.llm | self.parser)
	

RAGS = {'single-pdf': RAGSinglePDF}

# #################################### #
# 	Structured Outputs				   #
# Extracts information from text into  # 
# 			Schemas					   #	
# #################################### #
DEFAULT_PROMPT = """You are an expert extraction algorithm. 
     Only extract relevant information from the text. 
        If you do not find the attribute you are asked to extract, 
            return null."""	

# plain extactor chain
def plain(chat_model:str = 'openai-chat',
			 temperature: int = 0, schemas: List[str] = None) -> Dict[str, RunnableSequence]:
	prompt = ChatPromptTemplate.from_messages([("system", DEFAULT_PROMPT), ("human", "{text}")])
	llm = models.configure_chat_model(chat_model, temperature = temperature)
	return {schema: (prompt | llm.with_structured_output(schema = SCHEMAS[schema])) for schema in schemas}

EXTRACTORS = {'plain': plain}

# ####################################### #
#  		text summary tools		          #
# ####################################### #
# refine summarizer
def refine_pdf_summary(pdf_file: str, chat_model: str = 'openai-chat', 
				temperature = 0) -> str:
	"""
	uses predefined load_summarize_chain of LangChain to summarize a pdf file
	"""
	docs = PyPDFLoader(pdf_file).load_and_split()
	llm = models.configure_chat_model(chat_model, temperature = temperature)
	chain = load_summarize_chain(llm, chain_type = 'refine')
	summary = chain.invoke(docs)
	return summary["output_text"]

# MapReduce Summarizer 
MR = TypeVar('MR', bound = 'MapReduceSummary')
class MapReduceSummary(Callable):
	"""
	uses a MapReduce approach to generate a summary of the input text
	can be instantiated from pdf files or other texts
	"""
	def __init__(self, docs: List, chat_model: str = 'openai-chat',
				 temperature: int = 0, reduce_max_tokens: int = 4000):
		self.llm = models.configure_chat_model(chat_model, temperature = temperature)
		self.map_reduce_chain = None 
		self.split_docs = docs 
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
								split_method: str = 'tiktoken', 
									reduce_max_tokens: int = 4000) -> MR:
		docs = PyPDFLoader(pdf_file).load()
		if split_method == 'tiktoken':
			text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size = chunk_size, 
							chunk_overlap = chunk_overlap)
			split_docs = text_splitter.split_documents(docs)
		elif split_method == 'hugginface':
			raise NotImplementedError("huggingface splitter not implemented")
		
		return cls(split_docs, chat_model = chat_model,
					 temperature = temperature, reduce_max_tokens = reduce_max_tokens)
		
		
		