# ########################################## #
# LLM based utilities to work with PDF files #
# ########################################## #
import pandas as pd 
from os import path 
from typing import Optional, Any, List, Dict, overload, Callable   
from langchain_community.callbacks import get_openai_callback 
from .. chains import rags, extractors
from .. utils.questions import QUESTIONS 


class SchemaFromPDF:
	"""
	class to combine RAG and Extractor chains to extract structured information from 	
		texts. Structure is an schema 
	"""
	@overload
	def __init__(self, rag: str, extractor: str, pdf_file: str, chunk_size: Optional[int] = None,
				chunk_overlap: Optional[int] = None, chat_model: Optional[str] = None, 
					embedding_model: Optional[str] = None, prompt_template: Optional[str] = None, 
						temperature: Optional[int] = None, 
							schemas: Optional[List[str]] = None, 
								replacements: Optional[Dict[str, str]] = None) -> None:
		...

	def __init__(self, rag: str = 'single-pdf',
	 				extractor: str = 'plain', **kw) -> None:
		
		kw['schemas'] = kw['schemas'].split(',')
		rag = rags.RAGS[rag]
		Extract = extractors.EXTRACTORS[extractor]
		extract_keys = self._get_arg_keys(Extract)

		rag_inputs = {key:value for key,value in kw.items() if key in rag.init_keys}
		ext_inputs = {key:value for key,value in kw.items() if key in extract_keys}
		
		self.rag_chain = rag(**rag_inputs)()
		self.extract_chains = Extract(**ext_inputs)
		self.questions = kw['schemas']
	
	def ask(self, save_path: Optional[str] = None) -> None:
		"""
		asks questions from the extraction chain. Takes the answer and appends the __dict__ to
			the csv file if csv_files dictionary is not None. 
			Else it uses save_path to save the dictionary into a new csv file
		schemas are ['suspensions'] 
		"""
		for schema,ext_chain in self.extract_chains.items():
			extract_chain = (self.rag_chain | ext_chain)
			with get_openai_callback() as cb:
				info = extract_chain.invoke(QUESTIONS[schema])
				print(cb)
			print('the answer to this question is: ', info)
			info_df = pd.DataFrame.from_dict([info.__dict__])
			info_df.index.name = 'id'
			csv_file = path.join(save_path, schema + '.csv')
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

