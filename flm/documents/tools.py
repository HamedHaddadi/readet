# ########################################### #
# custom tools for interacting with documents #
# ########################################### #
import os 
from serpapi import GoogleScholarSearch 
from langchain.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.tools import BaseTool  
from typing import Optional, Any, Dict, Type   

# ## Google Scholar Search Tool ## #
# API Wrapper
class ScholarSearch(BaseModel):
	"""
	Wrapper for Serp API Google Scholar Search
	Attributes:
		top_k_results: number of results to return from goog-scholar query search
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
		GoogleScholarSearch.SERP_API_KEY = serp_api_key 
		values["scholar_search_engine"] = GoogleScholarSearch 
		return values 
	
	def run(self, query: str) -> str:
		page = 0
		all_results = []
		while page < max((self.top_k_results - 20), 1):
			results = (self.scholar_search_engine({"q": query, "start": page, "hl": "en",
                        "num": min( self.top_k_results, 20), "lr": "lang_en"}).get_dict().get("organic_results", []))
			all_results.extend(results)
			if not results:  
				break
			page += 20
		if (self.top_k_results % 20 != 0 and page > 20 and all_results):  # From the last page we would only need top_k_results%20 results
			results = (self.search_scholar_engine({"q": query,"start": page,
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



