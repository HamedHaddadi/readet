# ################################################################### #
# different tools to assist agents extract information from documents #
# ##################################################################  #
from langchain.tools import BaseTool 
from typing import Optional, Type, List, Dict 
from os import path, listdir 
from .. chains import documents 


class Summarize(BaseTool):
	name = "summarize"
	description =""" useful when you are given a path in the file system 
		and you are asked to prepare summaries of individual pdf files. You will use a 
			LangChain chain to perform this task """
	
	def _run(self, path_to_files: str) -> List[str]:
		summaries = []
		if path.exists(path_to_files):
			pdf_files = [path.join(path_to_files, pdf_file) for pdf_file in
				 listdir(path_to_files) if '.pdf' in pdf_file]
			for pdf in pdf_files:
				summaries.append(documents.MapReduceSummary.from_pdf(pdf)())
		else:
			raise FileExistsError("no pdf file is found in the directory")
		return summaries 
	
	async def _arun(self):
		raise NotImplementedError("the async version of run is not implemented ")


	

			

