
# ######################################### #
# Agents to interact with documents			#
# Includes:									#
#	1) GoogleScholar search 				#
# ######################################### #
import pandas as pd 
import plotly.express as px 
from collections.abc import Callable 
from langchain.agents import AgentExecutor, create_tool_calling_agent 
from langchain_openai import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools.google_scholar import GoogleScholarQueryRun 
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper  
from typing import Union 

class ScholarSearch(Callable):
	"""
	callable class for searching google scholar and returning 
		articles and citations 
	"""
	def __init__(self, num_results: int = 50, model: str = 'gpt-3.5-turbo'):
		search_tool = GoogleScholarQueryRun(api_wrapper = GoogleScholarAPIWrapper(
				top_k_results = num_results, hl = 'en', lr = 'lang_en'))
		llm = ChatOpenAI(model = model)
		prompt = ChatPromptTemplate.from_messages(
			[
			("system", 
				"""you are a helpful assistant for finding articles.  
						Make sure you use GoogleScholarQueryRun too for information. make sure to
							report number of citations in your final output.
								Make sure you report the full citation in your report including journal 
									or book title, volume, year and page. 
								report all articles that you find.  """
			),
			("placeholder", "{chat_history}"),
			("human", "{input}"),
			("placeholder", "{agent_scratchpad}"),
			]
		)
		tools = [search_tool]
		self.executor = AgentExecutor(agent = create_tool_calling_agent(llm, [search_tool], prompt),
				 tools = tools, verbose = True, 
				 			return_intermediate_steps = True)
	
	@staticmethod 
	def _parse_to_df(response: str) -> pd.DataFrame:
		results = {'Title': [],
		 			'Authors': [],
						  	'Reference': [],
						   		'Number of Citations':[]}
		for line in response.splitlines():
			if 'Title' in line:
				results['Title'].append(line.split(':')[1].lstrip())
			elif 'Summary' in line:
				summary = line.split(':')[1]
				results['Authors'].append(summary.split('-')[0])
				results['Reference'].append('-'.join(summary.split('-')[1:]))
			elif 'Total-Citations' in line:
				results['Number of Citations'].append(int(line.split(':')[1].lstrip()))			
		return pd.DataFrame.from_dict(results)
				
	def __call__(self, query: str, output: str = 'txt', 
						output_path: str = 'txt') -> pd.DataFrame:
		response = self.executor.invoke({"input": query}, 
					return_only_outputs = True)
		results_df = self._parse_to_df(response["intermediate_steps"][0][1])
		return results_df 
		
		
	
	
