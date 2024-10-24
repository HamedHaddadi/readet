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
	save_path: Optional[str] = Field(description = "path to the storing directory", default = None)

	def _run(self, query:str) -> str:
		"""
		Use the tool
		"""
		search_results = self.api_wrapper.run(query)
		if self.save_path is not None:
			with open(path.join(self.save_path, 'scholar_analytics_results.txt'), 'a') as f:
				f.write(search_results)
		return search_results

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
	save_path: Optional[str] = Field(description = "path to the storing directory", default = None)

	def _run(self, query:str) -> str:
		"""
		Use the tool
		"""
		search_results = self.api_wrapper.run(query)
		if self.save_path is not None:
			with open(path.join(self.save_path, 'patents_analytics_results.txt'), 'a') as f:
				f.write(search_results)
		return search_results

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
		all_files = [file for file in listdir(self.save_path) if '.pdf' in file]
		if name not in all_files:
			try:
				urlretrieve(url, path.join(self.save_path, name))
			except:
				return str(name)
		else:
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
	chat_model: str = Field(description = "the chat model", default = 'openai-gpt-4o-mini')

	@root_validator(pre = True)
	def setup_summarizer(cls, values: Dict) -> Dict:
		values['summarizer'] = SUMMARIZERS[values.get('summarizer_type', 'plain')](chat_model = values.get('chat_model', 'openai-gpt-4o-mini'))
		return values 
	
	def run(self, pdf_file: str) -> str:
		return self.summarizer(pdf_file)

class PDFSummaryTool(BaseTool):
	"""
	Tool that generates a summary of all pdf files stored in a directory (path_to_files)
		and writes all summaries in a *.txt file
	"""
	name: str = "summary_tool"
	description: str = """
		this tool is a summary builder. Useful when you are asked to prepare a summary of files
			and appending the summaries to a .txt file
	"""
	path_to_files: str = Field(description = "path to the directory/folder containing pdf fioles")
	to_file: bool = Field(description = "if True, the summaries are written to a file", default = True)
	path_to_summaries: Optional[str] = Field(description = "path to the directory/folder to store the summaries", default = None)
	summarizer: PDFSummary = Field(description = "the summarizer model")

	@root_validator(pre = True)
	def validate_path_to_summaries(cls, values: Dict) -> Dict:
		path_to_summaries = path.join(values.get('path_to_files'), 'summaries')
		if not path.exists(path_to_summaries):
			makedirs(path_to_summaries)
		values['path_to_summaries'] = path_to_summaries
		return values 
	
	def _write_to_file(self, summary: str, title: str, all: Literal[True, False] = True) -> None:
		if all:
			file_name = 'all_summaries.txt'
		else:
			file_name = f'summary of {title}.txt'
		with open(path.join(self.path_to_summaries, file_name), 'a') as f:
			f.write(f">>>> {title} <<<< \n")
			f.write(summary)
			f.write("\n")
			f.write('******* . ******* \n')
	
	def _summarize_all(self) -> str:
		pdf_files = [path.join(self.path_to_files, pdf_file) for pdf_file in listdir(self.path_to_files) if '.pdf' in pdf_file]
		for pdf_file in tqdm(pdf_files):
			title = Path(pdf_file).name.split('.pdf')[0]
			summary = self.summarizer.run(pdf_file)
			self._write_to_file(summary, title, True)
		return "all summaries are written to the file"
	
	def _summarize_one(self, pdf_file: str) -> str:
		pdf_file = path.join(self.path_to_files, pdf_file)
		summary = self.summarizer.run(pdf_file)
		title = Path(pdf_file).name.split('.pdf')[0]
		self._write_to_file(summary, title, False)
		return summary 

	def _run(self, pdf_file: str) -> str:
		if "all" in pdf_file.lower() and not pdf_file.endswith('.pdf'):
			return self._summarize_all()
		else:
			return self._summarize_one(pdf_file)
		
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
	path_to_files: str = Field(description = "path to the storing directory")
	suffix: str = Field(default = '.pdf', description ="suffix of the file")

	def _run(self) -> List[Union[str, PathLike]]:
		files = [path.join(self.path_to_files, file_name) for file_name in listdir(self.path_to_files)
						if self.suffix in file_name]
		return files  

# ######### keyword extraction tool ######### #
class Keywords(BaseModel):
	keywords_list: Sequence[str] = Field(default = [], description = 'list of extracted keywords')

class ExtractKeywords(BaseModel):
	runnable: Any 
	chat_model: str = Field(description = "the chat model", default = 'openai-gpt-4o-mini')

	@root_validator(pre =True)
	def generate_runnable_chain(cls, values: Dict) -> Dict:
		chat_model = values.get('chat_model', 'openai-gpt-4o-mini')
		llm = models.configure_chat_model(model = chat_model, temperature = 0)
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
	extractor: ExtractKeywords = Field(description = "the keyword extractor model")
	path_to_files: Optional[str] = None 
	save_to_file: bool = True 

	def _write_to_file(self, keywords: str, title: str) -> None:
		with open(path.join(self.path_to_files, 'extracted_keywords.txt'), 'a+') as f:
			f.write(f">>>> {title} <<<< \n")
			f.write(keywords)
			f.write(">>>> <<<< \n")

	def _run(self) -> Union[str, None]:
		pdf_files = [path.join(self.path_to_files, pdf_file) for pdf_file in listdir(self.path_to_files) if '.pdf' in pdf_file]
		for pdf_file in tqdm(pdf_files):
			title = Path(pdf_file).name.split('.pdf')[0]
			keywords = self.extractor.run(pdf_file)
			if self.save_to_file:
				self._write_to_file(keywords, title)
			else:
				return keywords 


# ############################################### 	#
# A tool to extract keywords from a pdf file and  	#
# extract keywords and query the pdf using keywords #
# ##############################################    #
class QueryByKeywords(BaseTool):
	"""
	This tool uses keyword extractor and retrieval augmented generation
		to extract keywords from a pdf file and stores the results in a text file
	"""
	name: str =  "query_by_keywords_store_to_file"
	description: str =  """
		A tool to ask questions about a document using keywords that are extracted from the text 
			It also writes the results to a text file. Use this tool when you are asked to 
			ask questions about a document using keywords. 
	"""
	extractor: ExtractKeywords = Field(description = "the keyword extractor model")
	rag_chat_model: str = Field(description = "the chat model for the RAG", default = 'openai-gpt-4o-mini')
	rag_embedding_model: str = Field(description = "the embedding model for the RAG", default = 'openai-text-embedding-3-large')
	path_to_files: str = Field(description = "path to all pdf files")
	rag_type: str = Field(description = "the RAG model", default = 'agentic-rag-pdf')

	get_title = staticmethod(lambda file_name: Path(file_name).name.split('.pdf')[0])

	def _run(self, pdf_file: str) -> str:
		pdf_file = path.join(self.path_to_files, pdf_file)
		title = self.get_title(pdf_file)
		rag = RAGS[self.rag_type](pdf_file, chat_model = self.rag_chat_model, 
					embedding_model = self.rag_embedding_model)
		rag.build()
		keywords = self.extractor.run(pdf_file)
		query_results = {key: None for key in keywords.split(',')}
		f = open(path.join(self.path_to_files, f'gist_of_{title}.txt'), 'a+')
		f.write(f'***** TITLE: {title.upper()} ***** \n')
		for keyword in query_results.keys():
			query = f"what does the manuscript say about {keyword} ?"
			results = rag.run(query)
			print(f'>>> {keyword}: {results} \n\n')
			f.write(f'>>> {keyword}: {results} \n\n')
			query_results[keyword] = results
		f.close()
		return '===== \n'.join([f"***{key} ==> {result} \n" for key, result in query_results.items() if result is not None])

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


# helper function to get tools using strings 
def get_tool(tool_name: str, tools_kwargs: Dict) -> BaseTool:
	if tool_name == 'arxiv_search':
		page_size = tools_kwargs.get('page_size', 10)
		max_results = tools_kwargs.get('max_results', 20)
		return ArxivTool(api_wrapper = ArxivSearch(page_size = page_size, max_results = max_results))
	
	elif tool_name == 'google_scholar_search':
		return GoogleScholarTool(api_wrapper = GoogleScholarSearch(top_k_results = tools_kwargs.get('max_results', 20)), 
				save_path = tools_kwargs.get('save_path', path.join(os.getcwd(), 'pdfs')))
	
	elif tool_name == 'google_patent_search':
		return GooglePatentTool(api_wrapper = PatentSearch(max_number_of_pages = tools_kwargs.get('max_results', 20), 
				save_path = tools_kwargs.get('save_path', path.join(os.getcwd(), 'pdfs'))))

	elif tool_name == 'pdf_download':
		save_path = tools_kwargs.get('save_path', path.join(os.getcwd(), 'pdfs'))
		return PDFDownloadTool(downloader = PDFDownload(save_path = save_path))
	
	elif tool_name == "summarize_pdfs":
		summarizer_type = tools_kwargs.get('summarizer_type', 'plain')
		chat_model = tools_kwargs.get('chat_model', 'openai-gpt-4o-mini')
		return PDFSummaryTool(path_to_files = tools_kwargs.get('save_path', path.join(os.getcwd(), 'pdfs')),
				summarizer = PDFSummary(summarizer_type = summarizer_type, chat_model = chat_model))
	
	elif tool_name == "list_files":
		suffix = tools_kwargs.get('suffix', '.pdf')
		return ListFilesTool(path_to_files = tools_kwargs.get('save_path', path.join(os.getcwd(), 'pdfs')),
								suffix = suffix)
	
	elif tool_name == "query_by_keywords":
		return QueryByKeywords(extractor = ExtractKeywords(chat_model = tools_kwargs.get('chat_model', 'openai-gpt-4o-mini')),
					path_to_files = tools_kwargs.get('save_path', path.join(os.getcwd(), 'pdfs')),
					rag_type = tools_kwargs.get('rag_type', 'agentic-rag-pdf'),
					rag_chat_model = tools_kwargs.get('rag_chat_model', 'openai-gpt-4o-mini'),
					rag_embedding_model = tools_kwargs.get('rag_embedding_model', 'openai-text-embedding-3-large'))
	
	else:
		raise ValueError(f"tool {tool_name} is not found") 
	
	



























