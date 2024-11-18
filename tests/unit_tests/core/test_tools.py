import unittest
from shutil import rmtree 
from os import path, getcwd, makedirs  
from peruse.core.tools import GoogleScholarSearch, GoogleScholarTool, PatentSearch, GooglePatentTool


def test_google_scholar_search_for_top_k_result_values():
	""" tests the output of search for various top_k_results """
	for k in range(20, 100, 20):
		google_scholar_search = GoogleScholarSearch(top_k_results = k)
		results = google_scholar_search.run('finite inertia suspensions')
		assert len(results.split("\n\n")) == k

def test_google_partent_search_for_max_number_of_patents():
	""" tests the number of patents collected in each search """
	for k in range(20, 100, 20):
		google_patent_search = PatentSearch(max_number_of_patents = k)
		results = google_patent_search.run('direct air capture')
		assert len(results.split("\n\n")) > k

class TestGoogleTools(unittest.TestCase):
	@classmethod 
	def setUpClass(cls):
		save_path = path.join(getcwd(), 'test_results')
		if not path.exists(save_path):
			makedirs(save_path)
		cls.save_path = save_path 
		cls.scholar_api = GoogleScholarSearch(top_k_results = 20)
		cls.patent_api = PatentSearch(max_number_of_patents = 20)
	@classmethod
	def tearDownClass(cls):
		rmtree(cls.save_path)

	def test_google_scholar_tool_saves_search_analytics(self):
		scholar_tool = GoogleScholarTool(api_wrapper = self.scholar_api, 
								   		save_path = self.save_path)
		scholar_tool.invoke("finite inertial suspensions")
		self.assertTrue(path.exists(path.join(self.save_path, 'scholar_analytics_results.txt'))) 

	def test_google_patent_tool_saves_search_analytics(self):
		patent_tool = GooglePatentTool(api_wrapper = self.patent_api, 
								   		save_path = self.save_path)
		patent_tool.invoke("direct air capture")
		self.assertTrue(path.exists(path.join(self.save_path, 'patents_analytics_results.txt'))) 



 
