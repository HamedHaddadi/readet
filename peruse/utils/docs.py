from langchain_community.document_loaders import PyPDFLoader, pdf
from langchain_core.documents.base import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from typing import Dict, List, Union, Literal, Any
from os import PathLike
from pathlib import Path    

# ################################ #
# utilities to work with documents #
# ################################ #
def format_doc_object(doc: List, replacements: Dict[str, str]) -> List:
	"""
	takes a doc object from PyPDFLoader and replace a str with another
	Returns: formatted doc
	"""
	for page in doc:
		for find,replace in replacements.items():
			page.page_content = page.page_content.replace(find, replace)
	return doc 

# pdf loading
def page_content_not_empty(docs: List[Document]) -> bool:
	if len(docs) == 0:
		return False 
	len_page_contnet = sum([len(doc.page_content) for doc in docs])
	if len_page_contnet == len(docs):
		return False 
	elif len_page_contnet > len(docs):
		return True 

def load_and_split_pdf(pdf_file: str, split = True) -> Union[List, None]:
	loader = PyPDFLoader(pdf_file)
	if split:
		docs = loader.load_and_split()
	else:
		docs = loader.load()
	if page_content_not_empty(docs):
		return docs 
	else:
		try:
			loader = pdf.PyMuPDFLoader(pdf_file, extract_images = True)
			docs = loader.load_and_split()
			if page_content_not_empty(docs):
				return docs 
			else:
				return None 
		except:
			return None


def doc_from_pdf_files(pdf_files: Union[str, List[str]], 
						document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
						splitter: Literal['recursive', 'token'] | None = 'recursive',
						splitter_kwargs: Dict[str, Any] = {}) -> List[Document]:
	
	loader_obj = {'pypdf': PyPDFLoader, 'pymupdf': pdf.PyMuPDFLoader}[document_loader]		
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


def text_from_pdf(document: Union[str, PathLike]) -> Union[str, None]:
	doc_path = Path(document)
	if doc_path.exists() and doc_path.is_file() and '.pdf' in document:
		pages = pdf.PyMuPDFLoader(document, extract_images = True)
		if pages is not None:
			pages = pages.load_and_split()
			text = '\n'.join([doc.page_content for doc in pages])
			return text 
		else:
			return None 	
