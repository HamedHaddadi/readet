# ########################################### #
# custom tools for interacting with documents #
# ########################################### #
import os
import sys 
import plotly 
from time import time
import pandas as pd  
from pathlib import Path 
from os import path, makedirs, listdir, PathLike 
from functools import wraps 
import plotly.express as px 
from tqdm import tqdm 
from arxiv import Search as ArSearch 
from arxiv import Client as ArClient    
from serpapi import Client 
from semanticscholar import SemanticScholar 
from pydantic import BaseModel, Field, root_validator
from langchain_core.tools import BaseTool, tool  
from langchain_core.retrievers import BaseRetriever 
from langchain_core.prompts import BasePromptTemplate, PromptTemplate, format_document  
from langchain_community.document_loaders import pdf 
from typing import Literal, Optional, Any, Dict, Union, List, Sequence   
from urllib.request import urlretrieve  
from . summarizers import SUMMARIZERS
from . rags import RAGS 
from .. utils import models, prompts, docs 

# ## Google Scholar Search Tool ## #
# API Wrapper
class GoogleScholarSearch(BaseModel):
	"""
	Wrapper for Serp API Google Scholar Search
	Attributes:
		top_k_results: number of results to return from google-scholar query search
			by defualt it returns 20 results.
		serp_api_key: key for the serapi API call. It must be provided as a keyword argument or be available 
			in the environment
		scholar_search_engine: serpapi GoogleScholarSearch class
	"""
	top_k_results: int = 20
	sepr_api_key: Optional[str] = None 
	scholar_search_engine: Any 

	@root_validator(pre = True)
	def validate_environment_and_key(cls, values: Dict) -> Dict:
		serp_api_key = values.get('serp_api_key')
		if serp_api_key is None:
			serp_api_key = os.environ["SERP_API_KEY"]
		client = Client(api_key = serp_api_key) 
		values["scholar_search_engine"] = client 
		return values 
	
	@staticmethod 
	def _get_results(results: Optional[Dict] = None) -> Dict[str, str]:
		parsed_results = {key: None for key in
				 ["Title", "Authors", "Venue", "Citation Count", "PDF Link"]}
		parsed_results["Title"] = results.get("title")
		summary = results["publication_info"]["summary"].split('-')
		parsed_results["Authors"] = summary[0]
		parsed_results["Venue"] = summary[1]
		parsed_results["Citation Count"] = results["inline_links"]["cited_by"]["total"]
		resources = results.get("resources", None)
		if resources is not None:
			if resources[0]["file_format"] == "PDF":
				parsed_results["PDF Link"] = resources[0]["link"]
		else:
			parsed_results["PDF Link"] = "None"
		return parsed_results 
	
	def run(self, query: str) -> str:
		page = 0
		all_results = []
		while page < max((self.top_k_results - 20), 1):
			organic_results = (self.scholar_search_engine.search({"engine": "google_scholar"
						,"q": query, "start": page, "hl": "en",
                        "num": min( self.top_k_results, 20), "lr": "lang_en"}).get("organic_results", []))
			for result in organic_results:
				fields = self._get_results(result)
				all_results.append(fields)
			
			if not organic_results:  
				break
			page += 20
		if (self.top_k_results % 20 != 0 and page > 20 and all_results):  # From the last page we would only need top_k_results%20 results
			organic_results = (self.search_scholar_engine.search({"engine":"google_scholar",
							 "q": query,"start": page,
					"num": self.top_k_results % 20,
					 "hl": "en", "lr": "lang_en"}).get("organic_results", []))
			for result in organic_results:
				fields = self._get_results(result)
				all_results.append(fields)
		if not all_results:
			return "No good Google Scholar Result was found"

		docs = ["******************* \n"
            f"Title: {result.get('Title','')}\n"
            f"Authors: {result.get('Authors')}\n"  # noqa: E501
            f"Citation Count: {result.get('Citation Count')}\n"
            f"PDF Link: {result.get('PDF Link')}"  # noqa: E501
            for result in all_results
        ]
		return "\n\n".join(docs)

# Google scholar tool
class GoogleScholarTool(BaseTool):
	"""
	Tool that requires scholar search API
	"""
	name: str = "google_scholar_tool"
	description: str = """A wrapper around Google Scholar Search.
        Useful for when you need to get information about
        research papers from Google Scholar
        Input should be a search query."""
	api_wrapper: GoogleScholarSearch 

	def _run(self, query:str) -> str:
		"""
		Use the tool
		"""
		return self.api_wrapper.run(query)

# ############################## #
# ArXiv Tool					 #
# ############################## #
class ArxivSearch(BaseModel):
	"""
	Wrapper for Arxiv search tool. 
	Attributes:
		page_size: maximum number of results fetched in a single API request
	max_results:
		max_results: maximum number of results to be returned in a search execution
	"""
	page_size: int = 10
	max_results: int = 20 

	def run(self, query: str) -> str:
		client = ArClient(page_size = self.page_size, delay_seconds = 3, num_retries = 3)
		results = client.results(ArSearch(query, max_results = self.max_results))
		fetched_data = []
		for result in results:
			data = {"title": result.title, 
						"journal_ref": result.journal_ref, 
							"DOI": result.doi, 
								"authors": ",".join([author.name for author in result.authors]), 
									"pdf_url": result.pdf_url}
			fetched_data.append(data)
		
		papers = [f"""Title: {data.get("title", "None")}\n
			PDF: {data.get("pdf_url", "None")}\n
			Authors: {data.get("authors", "None")}\n
			Journal Reference: {data.get("journal_ref", "None")}\n
			DOI: {data.get("doi", "None")}\n"""
			for data in fetched_data]

		return " ************ \n".join(papers)

class ArxivTool(BaseTool):
	"""
	Tool that requires ArxivSearch. 
	"""
	name: str ="arxiv_tool"
	description: str = """ A wrapper around ArxivSearch.
        Useful for when you need to search Arxiv database for manuscripts. 
		Input should be a string query
	"""
	api_wrapper: ArxivSearch 

	def _run(self, query: str) -> str:
		"""
		Use the tool
		"""
		return self.api_wrapper.run(query)


# ## Google Patent Search Tool ## #
# API call					   ## #	 
# ##						   ## # 
class PatentSearch(BaseModel):
	"""
		Wrapper for Serp API Google Scholar Search
	Attributes:
		max_number_of_pages: maximum number of pages to peruse
			by defualt it searches 10 pages.
		serp_api_key: key for the serapi API call. It must be provided as a keyword argument or be available 
			in the environment
		patent_search_engine: serpapi Client object with engine defined as google_patents 
	"""
	max_number_of_pages: int = 10 
	serp_api_key: Optional[str] = None 
	patent_search_engine: Any 

	@root_validator(pre = True)
	def validate_environment_and_key(cls, values: Dict) -> Dict:
		serp_api_key = values.get('serp_api_key')
		if serp_api_key is None:
			serp_api_key = os.environ["SERP_API_KEY"]
		client = Client(api_key = serp_api_key)
		values["patent_search_engine"] = client 
		return values 
	
	def run(self, query:str) -> str:
		patent_data = []
		search_inputs = {"engine": "google_patents",
                            "q": query,
                             "page": 1, 
                                "country": "US"}

		for _ in range(self.max_number_of_pages):
			organic_results = self.patent_search_engine.search(search_inputs)["organic_results"]
			for result in organic_results:
				data = {"title": result.get("title"), "patent_id": result.get("patent_id"), 
                        "pdf": result.get("pdf"),
                "priority_date": result.get("priority_date"), "filing_date": result.get("filing_date"), 
                "grant_date": result.get("grant_date"), "publication_date": result.get("publication_date"), 
                    "inventor": result.get("inventor"), "assignee": result.get("assignee")}
            
			search_inputs["page"] += 1
			patent_data.append(data)
		
		patents = [f"""Title: {data.get("title")} \n
                    PDF: {data.get("pdf")} \n
                    Patent ID: {data.get("patent_id")} \n
                        Priority Date: {data.get("priority_date")} \n 
                            Filing Date: {data.get("filing date")} \n
                            Grant Date: {data.get("grant_date")} \n
                            Publication Date: {data.get("publication_date")} \n
                            Inventor: {data.get("inventor")} \n
                            Assignee: {data.get("assignee")} \n
        """ for data in patent_data]
		return "\n\n".join(patents)

# Google Patent Tool 
class GooglePatentTool(BaseTool):
	"""
	Tool that requires Google patent search API
	"""
	name: str = "google_patent_tool"
	description: str = """A wrapper around Google Patent Search.
        Useful for when you need to get information about
        patents from Google Patents
        Input should be a search query."""
	api_wrapper: PatentSearch 

	def _run(self, query:str) -> str:
		"""
		Use the tool
		"""
		return self.api_wrapper.run(query)

# ####### Semantic Scholar Custom Tool ####### #
class SemanticSearch(BaseModel):
	"""
	Wrapper for Semantic Scholar Search with custom outputs. 
	Attributes:
		limit: total number of papers to download 
		timeout: the number of seconds to wait for retrieving a fields from Paper class after semantic search
		semantic_search_engine: An sinstance of SemanticScholar class 
		fields: List of fields to retrieve by semantic search 
	"""
	limit: int = 20 
	timeout: int = 10 
	semantic_search_engine: Any 
	fields: List = ["title", "abstract", "venue", "year", 
						"citationCount", "openAccessPdf", 
							"authors", "externalIds"]

	@root_validator(pre = True)
	def validate_engine(cls, values:Dict) -> Dict:
		try:
			engine = SemanticScholar(timeout = 20)
		except:
			raise Exception("can not instantiate SemanticScholarClass")
		values["semantic_search_engine"] = engine 
		return values 
	
	def timelimit(func):
		start_time = time.time()
		def wrapper(self, *args, **kwargs):
			if time.time() - start_time < self.timeout:
				return func(*args, **kwargs)
			else:
				print("Timeout exceeds for this loop")
		return wrapper 
	
	def retrieve_results(self, results: Any) -> Dict[str, str]:
		retrieved_items = {key:[] if key != "externalIds" else "DOI" for key in self.fields}
		print("retrieved items are ", retrieved_items)
		print(f"the length of results are {len(results)}")
		try:
			for item in results:
				for field in self.fields:
					if field == 'authors':
						retrieved_items["authors"].append([auth.get("name") for auth in item["authors"]])
					elif field == "externalIds":
						retrieved_items["DOI"].append(item["externalIds"].get("DOI", "Not Available"))
					else:
						retrieved_items[field].append(item[field])
		except:
			pass 
		return retrieved_items 
				
	def run(self, query: str) -> str:
		results = self.semantic_search_engine.search_paper(query = query, limit = self.limit, fields = self.fields)
		retrieved_items = self.retrieve_results(results)
		return "**********\n\n".join([f"{key} : {value} \n" for key,value in retrieved_items.items()])


# #### utilities for downloading pdf files #### #
class PDFDownload(BaseModel):
	"""
	PDF download class for downloading and saving a pdf file 
	Attributes:
		save_path: path to saving the pdf files.
	"""
	save_path: str = Field(description = "path to the storing directory")

	@root_validator(pre = True)
	def validate_save_path(cls, values: Dict) -> Dict:
		save_path = values.get('save_path')
		if not path.exists(save_path):
			makedirs(save_path)
		return values 

	def run(self, url: str, name:str) -> Union[None, str]:
		if '.pdf' not in name:
			name = '_'.join(name.split(' ')) + '.pdf'
		try:
			urlretrieve(url, path.join(self.save_path, name))
		except:
			return str(name)

class PDFDownloadTool(BaseTool):
	"""
	Tool that downloads pdf file using a url and stores 
		the file in a designated path
	"""
	name: str = "pdf_download_tool"
	description: str = """
    	this tool is a wrapper around PDFDownload. Useful when you are asked to download
			and store the pdf file.
	"""
	downloader: PDFDownload 

	def _run(self, url:str, name:str):
		self.downloader.run(url, name)

# ### summarizer BaseModel and BaseTools ### #
class PDFSummary(BaseModel):
	summarizer: Any 
	summarizer_type: Literal['plain'] = 'plain'
	chat_model: Literal['openai-gpt-4o', 'openai-chat'] = 'openai-chat'

	@root_validator(pre = True)
	def setup_summarizer(cls, values: Dict) -> Dict:
		values['summarizer'] = SUMMARIZERS[values.get('summarizer_type', 'plain')](chat_model = values.get('chat_model', 'openai-chat'))
		return values 
	
	def run(self, pdf_file: str) -> str:
		return self.summarizer(pdf_file)

class PDFSummaryTool(BaseTool):
	"""
	Tool that generates a summary of all pdf files stored in a directory (save_path)
		and writes all summaries in a *.txt files
	"""
	name: str = "pdf_summary_tool"
	description: str = """
		this tool is a summary builder. Useful when you are asked to prepare a summary of files
			and appending the summaries to a .txt file
	"""
	save_path: str = Field(description = "path to the storing directory")
	to_file: bool = Field(description = "if True, the summary is written to a file", default = True)
	summarizer: PDFSummary = Field(description = "the summarizer model")

	def _write_to_file(self, summary: str) -> None:
		with open(path.join(self.save_path, 'summaries.txt'), 'a') as f:
			f.write(summary)
			f.write("\n")
			f.write('******* . ******* \n')

	def _run(self) -> Union[str, None]:
		pdf_files =[path.join(self.save_path, pdf_file) for pdf_file in listdir(self.save_path) if '.pdf' in pdf_file]
		summaries = []
		if len(pdf_files) > 0:
			for pdf_file in tqdm(pdf_files):
				summary = self.summarizer.run(pdf_file)
				if summary != "" and self.to_file:
					self._write_to_file(summary)
				else:
					summaries.append(summary)
			if len(summaries) > 0:
				return '******\n'.join(summaries)
			else:
				return None 
		else:
			return None 

class ListFilesTool(BaseTool):
	"""
	This tool lists all files stored in a directory or folder. Files must have 
		a certain format or suffix
	"""
	name: str = "file_list_tool"
	description: str = """
		this tool lists files that are stored in a directory or folder. Useful when you 
			are asked to list files that are stored in a filder or directory
	"""
	save_path: str = Field(description = "path to the storing directory")
	suffix: str = Field(default = '.pdf', description ="suffix of the file")

	def _run(self) -> List[Union[str, PathLike]]:
		files = [path.join(self.save_path, file_name) for file_name in listdir(self.save_path)
						if self.suffix in file_name]
		return files 

# ######### keyword extraction tool ######### #
class Keywords(BaseModel):
	keywords_list: Sequence[str] = Field(default = [], description = 'list of extracted keywords')

class ExtractKeywords(BaseModel):
	runnable: Any 

	@root_validator(pre =True)
	def generate_runnable_chain(cls, values: Dict) -> Dict:
		llm = models.configure_chat_model(model = 'openai-chat', temperature = 0)
		template = prompts.TEMPLATES["extract-keywords"]
		prompt = PromptTemplate.from_template(template)
		values['runnable'] = (prompt | llm.with_structured_output(Keywords))
		return values 
	
	def run(self, document: str) -> str:
		if document.lower().endswith('.pdf'):
			text = docs.text_from_pdf(document)
		outputs = self.runnable.invoke(text)
		return ','.join(outputs.keywords_list)

class ExtractKeywordsTool(BaseTool):
	"""
	This tool extract keywords from a list of pdf files that are stored in a folder. 
	It can either return 
	"""
	name: str = "extract_keywords_to_file_tool"
	description: str = """
		this tool extracts keywords from a list of pdf files stored in a folder
			and either writes the keywords in a .txt file or returns them.
			Use this tool when you are asked to extract keywords from pdf files that are
				stored in a path.
			_run first constructs a list of pdf files  """
	extractor: ExtractKeywords 
	files_path: Optional[str] = None 
	save_to_file: bool = True 

	def _write_to_file(self, keywords: str, title: str) -> None:
		with open(path.join(self.files_path, 'extracted_keywords.txt'), 'a+') as f:
			f.write(f">>>> {title} <<<< \n")
			f.write(keywords)
			f.write(">>>> <<<< \n")

	def _run(self) -> Union[str, None]:
		pdf_files = [path.join(self.files_path, pdf_file) for pdf_file in listdir(self.files_path) if '.pdf' in pdf_file]
		for pdf_file in tqdm(pdf_files):
			title = Path(pdf_file).name.split('.pdf')[0]
			keywords = self.extractor.run(pdf_file)
			if self.save_to_file:
				self._write_to_file(keywords, title)
			else:
				return keywords 

# ################### #
# A Plain RAG tool	  #
# ##################  #
class RAGTool(BaseTool):
	"""
	Tool that uses Retrieval Augmented Generation to query a pdf file
	"""
	name: str = "rag_tool"
	description: str = """
		A tool to query a pdf file. Useful when you are asked to query a
			pdf document
	"""
	rag: Any = Field(description = "the RAG type")  
	pdf_file: str = Field(description = "the pdf file to query")

	@root_validator(pre = True)
	def generate_rag(cls, values: Dict) -> Dict:
		pdf_file = values.get("pdf_file")
		values["rag"] = RAGS["agentic-rag-pdf"](pdf_file)
		return values 

	def _run(self, query: str) -> str:
		return self.rag.run(query)

# ############################################### 	#
# A tool to extract keywords from a pdf file and  	#
# extract keywords and query the pdf using keywords #
# ##############################################    #
class QueryKeywordsTool(BaseTool):
	"""
	Tool that uses Keyword extractor and retrieval augmented generation
		to query a pdf file using the keywords that are extracted 
		from the pdf.
	"""
	name: str = "query_keywords_tool"
	description: str = """
		A tool to query pdf files using keywords that are extracted from the pdf file 
	"""
	extractor: ExtractKeywords 
	
	def _run(self, pdf_file: str) -> Union[str, None]:
		rag = RAGS["agentic-rag-pdf"](pdf_file)
		rag.build()
		keywords = self.extractor.run(pdf_file)
		results = {key: None for key in keywords.split(',')}
		for keyword in keywords.split(','):
			query = f"what does this article say about {keyword}"
			results[keyword] = rag.run(query)
		return '===== \n'.join([f"***{key} ==> {result} \n" for key, result in results.items()])

# ############################################### 	#
# A tool to extract keywords from a pdf file and  	#
# extract keywords and query the pdf using keywords #
# ##############################################    #
class GistToFileTool(BaseTool):
	"""
	This tool uses keyword extractor, summarizer and retrieval augmented generation
		to summarize and query pdf files and stores the results in a text file. 
	It first lists all pdf files that are stored in a directory and then it 
		queries them one by one. 
	"""
	name: str =  "query_by_keywords_store_to_file"
	description: str =  """
		A tool to query pdf files using keywords that are extracted from the pdf file
			and writing the results to a text file. Use this tool when you are asked to 
			query pdf files using the keywords. 
	"""
	extractor: ExtractKeywords 
	files_path: str = Field(description = "path to all pdf files")
	rag_type: str = Field(description = "the RAG model", default = 'agentic-rag-pdf')
	summarizer_type: str = Field(description = "the summarizer model", default = 'plain')
	pdf_files: List[str]
	
	@root_validator(pre = True)
	def generate_files(cls, values: Dict) -> Dict:
		files_path = values.get('files_path')
		files = [path.join(files_path, file_name) for file_name in listdir(files_path)
					 if file_name.lower().endswith('.pdf')]
		values['pdf_files'] = files 
		return values 

	def _write_to_file(self, title: str, summary: str, query_result: Dict[str, str]) -> None:
		with open(path.join(self.files_path, 'gist.txt'), 'a+') as f:
			f.write(f"*** TITLE ***\n")
			f.write(title.upper() + '\n')
			f.write(f"*** SUMMARY ***\n")
			f.write(f'*** {summary} *** \n')
			f.write('\n')
			f.write(f"*** KEY INFORMATION ***\n")
			f.write('\n'.join([f"{key} : {query_results} \n" 
				for key, query_results in query_result.items() if query_results is not None]))
			f.write(f'*** no information could be obtained about the following keywords ***: \n')
			f.write(','.join([key for key in query_result.keys() if key is None]))
			f.write('**** >>> <<<   ****')
	
	get_title = staticmethod(lambda file_name: Path(file_name).name.split('.pdf')[0])

	def _run(self) -> str:
		try:
			summarizer = SUMMARIZERS[self.summarizer_type]()
			for pdf_file in tqdm(self.pdf_files):
				title = self.get_title(pdf_file)
				rag = RAGS[self.rag_type](pdf_file)
				rag.build()
				summary = summarizer(pdf_file)
				keywords = self.extractor.run(pdf_file)
				query_results = {key: None for key in keywords.split(',')}
				for keyword in query_results.keys():
					query = f"what does the manuscript say about {keyword} ?"
					results = rag.run(query)
					query_results[keyword] = results
				self._write_to_file(title, summary, query_results)
			return "keywords extraction, summary building and output to text file suucessful"
		except Exception as e:
			pass 


# #### Charts and Plot Tools #### #
class BarChart(BaseModel):
	"""
	takes a pandas dataframe and plots a barchart
	"""
	def run(self, frame: pd.DataFrame, x_label: str, y_label: str, color: str,
				hover_name: str, hover_data: str) -> None:
		hover_data = hover_data.split(',')
		fig = px.bar(frame, y = y_label, x = x_label, orientation = 'h', 
					color = color, color_continuous_scale = 'Turbo',
						hover_name = hover_name, hover_data = hover_data,
                 height = len(frame.index)*30, template = 'seaborn')
		fig.layout.font.family = 'Gill Sans'
		fig.layout.font.size = 15
		fig.layout.xaxis.gridcolor = 'black'
		fig.layout.yaxis.gridcolor = 'black'
		fig.layout.xaxis.titlefont.family = 'Gill Sans'
		fig.layout.xaxis.titlefont.size = 15
		fig.layout.xaxis.tickfont.size = 15
		plotly.offline.plot(fig, filename='search_results.html')		
	



























