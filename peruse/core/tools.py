# ########################################### #
# custom tools for interacting with documents #
# ########################################### #
import os
from time import time 
from os import path, makedirs, listdir
from functools import wraps 
from tqdm import tqdm   
from serpapi import Client 
from semanticscholar import SemanticScholar 
from langchain.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.tools import BaseTool, tool  
from langchain_core.retrievers import BaseRetriever 
from langchain_core.prompts import BasePromptTemplate, PromptTemplate, format_document  
from typing import Optional, Any, Dict, Union, List   
from urllib.request import urlretrieve  
from . summarizers import PlainSummarizer

# ## Google Scholar Search Tool ## #
# API Wrapper
class ScholarSearch(BaseModel):
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
	
	def run(self, query: str) -> str:
		page = 0
		all_results = []
		while page < max((self.top_k_results - 20), 1):
			results = (self.scholar_search_engine({"engine": "google_scholar"
						,"q": query, "start": page, "hl": "en",
                        "num": min( self.top_k_results, 20), "lr": "lang_en"}).get_dict().get("organic_results", []))
			all_results.extend(results)
			if not results:  
				break
			page += 20
		if (self.top_k_results % 20 != 0 and page > 20 and all_results):  # From the last page we would only need top_k_results%20 results
			results = (self.search_scholar_engine({"engine":"google_scholar",
							 "q": query,"start": page,
					"num": self.top_k_results % 20,
					 "hl": "en", "lr": "lang_en"}).get_dict().get("organic_results", []))
			all_results.extend(results)
		if not all_results:
			return "No good Google Scholar Result was found"
		docs = [
            f"Title: {result.get('title','')}\n"
            f"Authors: {','.join([author.get('name') for author in result.get('publication_info',{}).get('authors',[])])}\n"  # noqa: E501
            f"Summary: {result.get('publication_info',{}).get('summary','')}\n"
            f"Total-Citations: {result.get('inline_links',{}).get('cited_by',{}).get('total','')}"  # noqa: E501
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
	api_wrapper: ScholarSearch 

	def _run(self, query:str) -> str:
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

class PDFSummaryTool(BaseTool):
	"""
	Tool that generates a summary of all pdf files stored in a directory (save_path)
		and writes all summaries in a *.txt files
	"""
	name: str = "pdf_summary_tool"
	description: str = """
		this tool is summary builder. Useful when you are asked to prepare a summary of pdf files
			stored in a directory and appending the summaries to a .txt file
	"""
	save_path: str = Field(description = "path to the storing directory")

	def _run(self) -> None:
		pdf_files =[path.join(self.save_path, pdf_file) for pdf_file in listdir(self.save_path) if '.pdf' in pdf_file]
		sum_chain = PlainSummarizer(chat_model = 'openai-chat')
		if len(pdf_files) > 0:
			for pdf_file in tqdm(pdf_files):
				summary = sum_chain(pdf_file)
				with open(path.join(self.save_path, 'summaries.txt'), 'a') as f:
					f.write(summary)
					f.write("\n")
					f.write('******* . ******* \n')

# ########### Retriever Tool ############## #
class RetrieverRunnable(BaseModel):
	"""
	Takes a retriever as an attribute and invokes a query on the retriever
	"""
	retriever: BaseRetriever
	prompt: BasePromptTemplate = PromptTemplate.from_template("{page_content}")
	def run(self, query: str) -> str:
		docs = self.retriever.invoke(query)
		return "\n\n".join(format_document(doc, self.prompt) for doc in docs)

class RetrieverTool(BaseTool):
	name: str = "retriever_tool"
	description: str = """retriever tool which is used to retriever documents from a 
				vector store. """
	retriever: RetrieverRunnable 

	def _run(self, query: str) -> str:
		return self.retriever.run(query)

	
# #### The following tools are useful when Args need not to be passed an instantiation #### #
@tool 
def download_file(url: str, save_path: str, name: str) -> str:
    """
    this function takes url to the pdf file, a path in which the pdf is stored and a name.
	Then it downloads and saves the pdf file. 
    it saves the pdf file under the name name.pdf! name can be a string of numbers or a string and is one of the inputs to this function. 
    Use this tool when you are asked to download a pdf. if you could not download the pdf file, simply pass 
    """
    try:
        urlretrieve(url, os.path.join(save_path, str(name) + '.pdf'))
    except:
        return str(name) 

@tool 
def summarizer_tool(save_path: str) -> Union[None,str]:
	"""
    this function prepares a summary of all pdf files that are stored in the save_path and
        appends all summaries to a .txt file called summary.txt. use this tool when you are asked to 
            prepare a summary of all pdf files
    """
	pdf_files = [path.join(save_path, pdf_file) for pdf_file in listdir(save_path) if '.pdf' in pdf_file]
	sum_chain = PlainSummarizer(chat_model = 'openai-chat')
	print('pdf file is ', pdf_files)
	if len(pdf_files) > 0:
		for pdf_file in tqdm(pdf_files):
			summary = sum_chain(pdf_file)
			with open(path.join(save_path, 'summaries.txt'), 'a') as f:
				f.write(summary)
				f.write("\n")
	else:
		return "was not able to summarize"
























