# ########################################## #
# LLM based utilities to work with PDF files #
# ########################################## #
import pandas as pd 
from os import path 
from typing import Optional, Dict, overload, Callable   
from langchain_community.callbacks import get_openai_callback 
from . chains import RAGS, EXTRACTORS
from .. utils.questions import QUESTIONS 
from collections.abc import Callable 


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
		rag = RAGS[rag]
		Extract = EXTRACTORS[extractor]
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
				info = extract_chain.invoke(QUESTIONS[schema])
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

