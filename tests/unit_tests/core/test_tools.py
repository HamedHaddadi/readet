import pytest 
import unittest
from shutil import rmtree 
from os import path, getcwd, makedirs  
from peruse.core import tools 


@pytest.mark.skip(reason = "tested and to avoid exhausting API calls")
def test_google_scholar_search_for_top_k_result_values():
	""" tests the output of search for various top_k_results """
	for k in range(20, 100, 20):
		google_scholar_search = tools.GoogleScholarSearch(top_k_results = k)
		results = google_scholar_search.run('finite inertia suspensions')
		assert len(results.split("\n\n")) == k

@pytest.mark.skip(reason = "tested and to avoid exhausting API calls")
def test_google_scholar_search_for_pdf_link_for_various_topics(random_topic):

	google_scholar_search = tools.GoogleScholarSearch(top_k_results = 20)
	results = google_scholar_search.run(random_topic)
	results = results.split("\n\n")
	sum = 0
	for result in results:
		result_lines = result.split("\n")
		for line in result_lines:
			if "PDF Link: https://" in line:
				sum += 1
	assert sum > 0

@pytest.mark.skip(reason = "tested and to avoid exhausting API calls")
def test_google_partent_search_for_max_number_of_patents():
	""" tests the number of patents collected in each search """
	for k in range(20, 100, 20):
		google_patent_search = tools.PatentSearch(max_number_of_patents = k)
		results = google_patent_search.run('direct air capture')
		assert len(results.split("\n\n")) > k

@pytest.mark.skip(reason = "tested and to avoid exhausting API calls")
class TestGoogleTools(unittest.TestCase):
	@classmethod 
	def setUpClass(cls):
		test_save_path = path.join(getcwd(), 'test_results')
		if not path.exists(test_save_path):
			makedirs(test_save_path)
		cls.save_path = test_save_path 
		cls.scholar_api = tools.GoogleScholarSearch(top_k_results = 20)
		cls.patent_api = tools.PatentSearch(max_number_of_patents = 20)
	
	@classmethod
	def tearDownClass(cls):
		rmtree(cls.save_path)

	def test_google_scholar_tool_saves_search_analytics(self):
		scholar_tool = tools.GoogleScholarTool(api_wrapper = self.scholar_api, 
								   		save_path = self.save_path)
		scholar_tool.invoke("finite inertial suspensions")
		self.assertTrue(path.exists(path.join(self.save_path, 'scholar_analytics_results.txt'))) 

	def test_google_patent_tool_saves_search_analytics(self):
		patent_tool = tools.GooglePatentTool(api_wrapper = self.patent_api, 
								   		save_path = self.save_path)
		patent_tool.invoke("direct air capture")
		self.assertTrue(path.exists(path.join(self.save_path, 'patents_analytics_results.txt'))) 

def test_get_tool_works_for_all_inputs():
	test_save_path = './test_results'
	if not path.exists(test_save_path):
		makedirs(test_save_path)
	tool_list = ['google_scholar_search', 'google_patent_search', 'arxiv_search',
			   'pdf_download', 'summarize_pdfs', 'list_files', 'query_by_keywords', 'rag']
	for tool_name in tool_list:
		tool = tools.get_tool(tool_name, {'save_path': test_save_path, 
										'max_results': 20, 'page_size': 10, 
											'chat_model': 'openai-gpt-4o-mini'})
		assert isinstance(tool, tools.BaseTool)
	rmtree(test_save_path)

 
