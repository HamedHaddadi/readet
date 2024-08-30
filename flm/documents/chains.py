# ############################################################ #
# module includes chains useful for interaction with documents #
# ############################################################ #

from typing import Optional, Dict, List, TypeVar, Union, Any 
from collections.abc import Callable 
from os import path, listdir, PathLike 
from pathlib import Path 
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
from langchain_experimental.graph_transformers import LLMGraphTransformer 
from langchain_core.documents import Document 
from langchain_community.graphs import Neo4jGraph 
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
	init_keys = ['chunk_size', 'chunk_overlap', 'chat_model', 
						'embedding_model', 'schema', 'temperature', 'replacements']

	def __init__(self, chunk_size: int = 2000,
	 				chunk_overlap: int = 150,
					 	 chat_model: str = 'openai-chat', 
						  	embedding_model: str = 'openai-embedding',
						  	prompt: str = 'suspensions', 
							  	temperature: int = 0, 
								  	replacements: Optional[Dict[str, str]] = None):
		self.retriever = None 
		self.prompt_template = prompts.TEMPLATES[prompt]
		self.prompt = None 
		self.llm = models.configure_chat_model(chat_model, temperature = temperature)
		self.parser = StrOutputParser()
		self.replacements = replacements 
		self._configure(chunk_size, chunk_overlap, embedding_model)
	
	
	def _configure(self, chunk_size: int, chunk_overlap: int, embedding_model: str) -> None:
		self.splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
						 chunk_overlap = chunk_overlap, add_start_index = True, 
						 		separators = ["\n\n", "\n", "(?<=\. )", " ", ""])
		self.embedding = models.configure_embedding_model(embedding_model)
		self.prompt = PromptTemplate.from_template(self.prompt_template)
	
	def __call__(self, pdf_file: str) -> RunnableSequence:
		loader = PyPDFLoader(pdf_file)
		doc = loader.load()
		if self.replacements is not None:
			doc = docs.format_doc_object(doc, self.replacements)
		splits = self.splitter.split_documents(doc)
		store = Chroma.from_documents(documents = splits, embedding = self.embedding)
		retriever = store.as_retriever()
		return ({'context': retriever, 'question': RunnablePassthrough()} | self.prompt | self.llm | self.parser)
	

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
def extract_schema_plain(schemas: Union[List[str], str], 
				chat_model:str = 'openai-chat', 
			 			temperature: int = 0) -> Union[Dict[str, RunnableSequence], RunnableSequence]:
	prompt = ChatPromptTemplate.from_messages([("system", DEFAULT_PROMPT), ("human", "{text}")])
	llm = models.configure_chat_model(chat_model, temperature = temperature)
	if isinstance(schemas, list):
		return {schema: (prompt | llm.with_structured_output(schema = SCHEMAS[schema])) for schema in schemas}
	elif isinstance(schemas, str):
		return (prompt | llm.with_structured_output(schema = SCHEMAS[schemas]))

EXTRACTORS = {'plain': extract_schema_plain}

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
	def __init__(self, document: List, chat_model: str = 'openai-chat', temperature: int = 0):
		llm = models.configure_chat_model(chat_model, temperature = temperature)
		template = prompts.PLAIN_SUMMARY 
		prompt = PromptTemplate.from_template(template)
		self.document = document 
		self.chain = (prompt | llm)
	
	def __call__(self) -> str:
		return self.chain.invoke(self.document).content 
	
	@classmethod
	def from_pdf(cls, pdf: str, chat_model: str = 'oprnai-chat', temperature: int = 0) -> PS:
		loader = PyPDFLoader(pdf)
		document = loader.load_and_split()
		return cls(document, chat_model = chat_model, temperature = temperature)


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

# ################################### #
# Knowledge Graph Builder			  #
# ################################### #

KG = TypeVar('KG', bound = 'KnowledgeGraph')
class KnowledgeGraph(Callable):
	"""
	Builds a knowledge graph from a text
	KnowledgeGraph builder is a chain.
	"""
	def __init__(self, summaries: str, store_graph: bool = True, 
				allowed_nodes: Optional[List[str]] = None,
					 allowed_relations: Optional[List[str]] = None):
		self.summaries = summaries 
		self.store_graph = store_graph 
		self.allowed_nodes = allowed_nodes 
		self.allowed_relations = allowed_relations 
		self.graph = Neo4jGraph()
		self.graph_doc = None 
	
	def _build(self):
		llm = models.configure_chat_model(model = 'openai-chat', temperature = 0)
		llm_transformer = LLMGraphTransformer(llm = llm, allowed_nodes = self.allowed_nodes, 
					allowed_relationships = self.allowed_relations)
		documents = [Document(self.summaries)]
		self.graph_doc = llm_transformer.convert_to_graph_documents(documents)
	
	@property 
	def nodes(self) -> List:
		if self.graph_doc is None:
			self._build()
		return self.graph_doc[0].nodes 
	
	@property
	def relations(self) -> List:
		if self.graph_doc is None:
			self._build()
		return self.graph_doc[0].relationships 
	
	def __call__(self) -> None:
		self._build()
		if self.store_graph:
			self.graph.add_graph_documents(self.graph_doc)

	@classmethod 
	def from_pdf(cls, pdf: Union[str, PathLike, List], chat_model: str = 'openai-chat', temperature: int = 0,
				 store_graph: bool = True,  summarizer: str = 'plain', allowed_nodes: Optional[List[str]] = None, 
						allowed_relations: Optional[List[str]] = None, **summarizer_kw: Any) -> KG:

		if not isinstance(pdf, (list, tuple)):
			p = Path(pdf)
			if p.is_dir():
				pdf = [path.join(p, pdf_file) for pdf_file in listdir(p) if '.pdf' in pdf_file]
			elif p.is_file:
				pdf = [p]
		
		print('pdf is: ', pdf)
		
		summarize_method = {'plain': PlainSummarizer}[summarizer]
		summaries = '\n'.join([summarize_method.from_pdf(pdf_file, chat_model = chat_model, 
								temperature = temperature, **summarizer_kw)() for pdf_file in pdf])
		return cls(summaries, store_graph = store_graph, allowed_nodes = allowed_nodes, 
										allowed_relations = allowed_relations)
		
		
		