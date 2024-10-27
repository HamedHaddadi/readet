# ################################# #
# Retrievers						#
# collection of retrievers with one #
# level more abstraction than		#
#  LangChain various retrievers		#
# ################################# #
from typing import List, Union, Literal, Dict, Any, TypeVar 
from abc import ABCMeta, abstractmethod 
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_loaders import PyPDFLoader, pdf  
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, TokenTextSplitter
from langchain.storage import InMemoryStore 
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI
from .. utils import models 

# ######################################### #
# helper functions						    #
# ######################################### #

def format_documents(documents: List[Document]) -> str:
	return f"\n {'-' * 100} \n".join([f"Document {i + 1}: \n\n" + 
								   	d.page_content for i, d in enumerate(documents)])

def doc_from_pdf_files(pdf_files: Union[str, List[str]], 
						document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
						splitter: Literal['recursive', 'token'] | None = 'recursive',
						splitter_kwargs: Dict[str, Any] = {}) -> List[Document]:
	
	loader_obj = {'pypdf': PyPDFLoader, 'pymupdf': pdf}[document_loader]		
	if splitter == 'recursive':
		chunk_size = splitter_kwargs.get('chunk_size', 2000)
		chunk_overlap = splitter_kwargs.get('chunk_overlap', 200) 
		splitter = RecursiveCharacterTextSplitter(separators = None, 
						chunk_size = chunk_size, 
								chunk_overlap = chunk_overlap, add_start_index = True)
	elif splitter == 'token':
		splitter = TokenTextSplitter()

	documents = []
	if not isinstance(pdf_files, (list, tuple)):
		pdf_files = [pdf_files]
	if splitter is not None:
		for pdf_file in pdf_files:
			documents.extend(loader_obj(pdf_file, extract_images = True).load_and_split(splitter))
	else:
		for pdf_file in pdf_files:
			documents.extend(loader_obj(pdf_file, extract_images = True).load())
	return documents

# ######################################### #
# Retrievers base class						#
# ######################################### #	
class Retriever(metaclass = ABCMeta):
	def __init__(self):
		self.runnable = None 
		self._built = False 
	
	@property
	def built(self) -> bool:
		return self._built
	
	@built.setter
	def built(self, value: bool) -> None:
		if self._built is False and value is True:
			self._built = True 
	
	@abstractmethod
	def build(self):
		...
	
	def run(self, query: str) -> str:
		if self.built is False:
			_ = self.build()
		docs = self.runnable.invoke(query)
		return format_documents(docs)
	
	def __call__(self, query: str) -> str:
		return self.run(query)
	
# ######################################### #
# PlainRetriever							#
# ######################################### #
PR = TypeVar('PR', bound = 'PlainRetriever')
class PlainRetriever(Retriever):
	"""
	Plain retriver class
	Class that abstracts the retrieval of documents from a vector store
	Accepts a list of Document objects and an embedding model name
	"""
	def __init__(self, documents: List[Document], 
				embeddings: str = 'openai-text-embedding-3-large'):
		super().__init__()
		self.vector_store = Chroma.from_documents(documents,
				embedding = models.configure_embedding_model(embeddings))
		self.runnable = None 
	
	def build(self, search_type: Literal["mmr", "similarity"] = "similarity", 
				k: int = 5, lambda_mult: float = 0.5, fetch_k: int = 20) -> VectorStoreRetriever:
		self.runnable = self.vector_store.as_retriever(search_type = search_type, k = k, 
												  lambda_mult = lambda_mult, fetch_k = fetch_k)
		return self.runnable
	
	@classmethod
	def from_pdf(cls, pdf_files: Union[str, List[str]], 
				embeddings: str = 'openai-text-embedding-3-large',
				document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
				splitter: Literal['recursive', 'token'] = 'recursive',
				splitter_kwargs: Dict[str, Any] = {}) -> PR:
		documents = doc_from_pdf_files(pdf_files, document_loader, splitter, splitter_kwargs)
		return cls(documents, embeddings)
		
# ######################################### #
# ContextualCompressionRetriever			#
# ######################################### #
CC = TypeVar('CC', bound = 'ContextualCompression')
class ContextualCompression(Retriever):
	"""
	Class that abstracts the ContextualCompressionPipeLine
	This class builds a ContextualCompressionRetriever and accepts a list of documents
	"""
	def __init__(self, documents: List[Document], llm: str = 'openai-gpt-4o-mini', 
				embeddings: str = 'openai-text-embedding-3-large', 
					search_type: Literal["mmr", "similarity"] = "similarity",
					k: int = 5, lambda_mult: float = 0.5, fetch_k: int = 20):
		super().__init__()
		self.documents = documents
		self.llm = OpenAI(temperature = 0)
		self.base_retriever = PlainRetriever(documents, embeddings).build(search_type, k, lambda_mult, fetch_k)
		self.runnable = None 
	
	def build(self) -> ContextualCompressionRetriever:
		"""
		returns the runnable for use in a chain
		"""
		compressor = LLMChainExtractor.from_llm(self.llm)
		self.runnable = ContextualCompressionRetriever(base_compressor = compressor, 
												 base_retriever = self.base_retriever) 
		return self.runnable 
	
	@classmethod
	def from_pdf(cls, pdf_files: Union[str, List[str]],
			   splitter: Literal['recursive', 'token'] = 'recursive', 
			   	document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
				llm: str = 'openai-gpt-4o-mini', 
				embeddings: str = 'openai-text-embedding-3-large',
				search_type: Literal["mmr", "similarity"] = "similarity",
				k: int = 5, lambda_mult: float = 0.5, fetch_k: int = 20,
			   	splitter_kwargs: Dict[str, Any] = {}) -> CC:
		
		documents = doc_from_pdf_files(pdf_files, document_loader, splitter, splitter_kwargs)
		return cls(documents, llm = llm, embeddings = embeddings, search_type = search_type,
					k = k, lambda_mult = lambda_mult, fetch_k = fetch_k)

# ######################################### #
# 	Parent Document Retriever				#
# ######################################### #
PD = TypeVar('PD', bound = 'ParentDocument')
class ParentDocument(Retriever):
	"""
	Class that abstracts the Parent Document Retriever
	"""
	def __init__(self, documents: List[Document], 
				embeddings: str = 'openai-text-embedding-3-large',
				store: str = "memory",
				parent_splitter: Literal['recursive', 'token'] = 'token', 
			  	child_splitter: Literal['recursive', 'token'] = 'recursive', 
					splitter_kwargs: Dict[str, Any] = {}):
		
		super().__init__()
		self.parent_splitter = None 
		self.child_splitter = None 

		if parent_splitter == 'recursive':
			self.parent_splitter = RecursiveCharacterTextSplitter(separators = None, 
				chunk_size = splitter_kwargs.get('parent_chunk_size', 2000), 
				chunk_overlap = splitter_kwargs.get('parent_chunk_overlap', 200), add_start_index = True)
		elif parent_splitter == 'token':
			self.parent_splitter = TokenTextSplitter()
		
		if child_splitter == 'recursive':
			self.child_splitter = RecursiveCharacterTextSplitter(separators = None, 
				chunk_size = splitter_kwargs.get('child_chunk_size', 2000), 
				chunk_overlap = splitter_kwargs.get('child_chunk_overlap', 100), add_start_index = True)
		elif child_splitter == 'token':
			self.child_splitter = TokenTextSplitter()
				
		self.documents = documents
		self.vector_store = Chroma(collection_name = "parent_document_retriever", 
								embedding_function = models.configure_embedding_model(embeddings))
		self.store = InMemoryStore() 
	
	def build(self) -> ParentDocumentRetriever:
		self.runnable = ParentDocumentRetriever(vectorstore = self.vector_store, 
												docstore = self.store, 
												parent_splitter = self.parent_splitter, 
												child_splitter = self.child_splitter, id_key = "doc_id")
		self.runnable.add_documents(self.documents)
		return self.runnable
	
	@classmethod
	def from_pdf(cls, pdf_files: Union[str, List[str]],
		embeddings: str = 'openai-text-embedding-3-large',
				store = "memory",
				document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
					parent_splitter: Literal['recursive', 'token'] = 'recursive', 
						child_splitter: Literal['recursive', 'token'] = 'recursive',
							splitter_kwargs: Dict[str, Any] = {}) -> PD:
		
		documents = doc_from_pdf_files(pdf_files,
			document_loader = document_loader, splitter = None)
		return cls(documents, embeddings = embeddings, store = store, 
					parent_splitter = parent_splitter, child_splitter = child_splitter, 
						splitter_kwargs = splitter_kwargs)


		
