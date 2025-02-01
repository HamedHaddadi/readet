from langchain_community.document_loaders import PyPDFLoader, pdf
from langchain_core.documents.base import Document, Blob 
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from typing import List, Union, Literal, Any 
from os import path, PathLike, listdir
from pathlib import Path   
from tqdm import tqdm  

# ################################ #
# utilities to work with documents #
# ################################ #

def get_pdf_files(pdf_files: Union[str, List[str]]) -> List[PathLike]:
	if isinstance(pdf_files, str):
		return [path.join(pdf_files, file_name) for file_name in listdir(pdf_files) if file_name.endswith('.pdf')]
	elif isinstance(pdf_files, (list, tuple)):
		return pdf_files


def doc_from_pdf_files(pdf_files: Union[str, List[str]], 
						document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
						splitter: Literal['recursive', 'token'] | None = 'recursive',
						chunk_size: int = 2000, chunk_overlap: int = 200) -> List[Document]:
	
	loader_obj = {'pypdf': PyPDFLoader, 'pymupdf': pdf.PyMuPDFLoader}[document_loader]		
	if splitter == 'recursive':
		splitter = RecursiveCharacterTextSplitter(separators = None, 
						chunk_size = chunk_size, 
								chunk_overlap = chunk_overlap, add_start_index = True)
	elif splitter == 'token':
		splitter = TokenTextSplitter()

	if not isinstance(pdf_files, (list, tuple)):
		pdf_files = [pdf_files]

	
	def get_docs(loader: Any) -> List[Document]:
		try:
			if splitter is not None:
				docs = loader.load_and_split(splitter)
			else:
				docs = loader.load()
			return docs
		except:
			return None 

	documents = []
	for pdf_file in tqdm(pdf_files):
		loader = loader_obj(pdf_file, extract_images = True)
		docs = get_docs(loader)
		if docs is not None:
			documents.extend(docs)
		
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

# ### generates Document objects from an encoded file ### #
# useful for web apps that need pdf upload
class DocumentFromEncodedFile:
	"""
	Takes an encoded file and uses a Blob and PyPDFParser to generate a list of Document objects.
	"""
	def __init__(self, decoded: bytes, extract_images = True, source = 'docs'):
		self.blob = Blob.from_data(decoded, metadata = {'source': source})
		self.parser = PyPDFParser(extract_images = extract_images)
		self.docs = self.parser.parse(self.blob)
	
	def __call__(self, split = True, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[Document]:
		if split:
			splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
			return splitter.split_documents(self.docs)
		else:
			return self.docs

	def __len__(self):
		return len(self.docs)

