import unittest
from os import path, listdir, getcwd, makedirs 
from shutil import rmtree
from readet.bots.agents import ReAct


class TestReactAgent(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		test_save_path = path.join(getcwd(), 'test_results')
		if not path.exists(test_save_path):
			makedirs(test_save_path)
		cls.test_save_path = test_save_path

	@classmethod
	def tearDownClass(cls):
		rmtree(cls.test_save_path)

	def test_react_runs_with_scholar_and_download_tools(self):
		agent = ReAct(tools = ['google_scholar_search', 'pdf_download'], 
					save_path = self.test_save_path, max_results = 10)
		agent.run("find and download all papers on finite inertial suspensions")
		files = [f for f in listdir(self.test_save_path) if f.endswith('.pdf')]
		assert len(files) > 0 

