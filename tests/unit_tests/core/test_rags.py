from os import path, getcwd, makedirs, listdir
from shutil import rmtree
from unittest import TestCase
from langchain_core.stores import BaseStore
from peruse.core.retrievers import ParentDocument
from peruse.core.rags import RAGWithCitations, PlainRAG, AVAILABLE_RETRIEVERS, AgenticRAG

def test_plain_rag_for_different_retrievers(single_pdf_file):
	for retriever_type in AVAILABLE_RETRIEVERS:
		plain_rag = PlainRAG(single_pdf_file, retriever = retriever_type)
		plain_rag.build()
		response = plain_rag.run(query = 'What is the main idea of the document?')
		assert isinstance(response, str)

def test_rag_with_citations_includes_citations(single_pdf_file):
	rag_cite = RAGWithCitations(single_pdf_file, retriever = 'parent-document')
	rag_cite.build()
	_, citations = rag_cite.run(query = 'who are the authors of the document?', parse = True)
	assert set(['Haddadi', 'Morris', 'Kulkarni']).difference(set(citations)) != set()


# test agentic rag
class TestAgenticRAG(TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		docstore_path = path.join(getcwd(), 'test_results', 'docstore')
		if not path.exists(docstore_path):
			makedirs(docstore_path)
		cls.docstore_path = docstore_path
		pdf_path = path.join(getcwd(), 'tests', 'data')
		cls.pdf_files = [path.join(pdf_path, f) for f in listdir(pdf_path) if f.endswith('.pdf')]

	@classmethod
	def tearDownClass(cls) -> None:
		rmtree(cls.docstore_path)
	
	def test_initialization_with_store_path_and_version_number_none(self):
		agentic_rag = AgenticRAG(self.pdf_files[0], retriever = 'parent-document', store_path = self.docstore_path, load_version_number = None)
		assert isinstance(agentic_rag.retriever, ParentDocument)
		assert isinstance(agentic_rag.retriever.runnable.docstore.store, dict)
		assert len(agentic_rag.retriever.runnable.docstore.store.values()) > 0

	def test_initialization_with_store_path_and_version_number(self):
		agentic_rag = AgenticRAG(self.pdf_files[0], retriever = 'parent-document', store_path = self.docstore_path, load_version_number = 'last')
		assert isinstance(agentic_rag.retriever, ParentDocument)
		assert isinstance(agentic_rag.retriever.runnable.docstore.store, dict)
		assert len(agentic_rag.retriever.runnable.docstore.store.values()) > 0
	
	def test_run_without_building(self):
		agentic_rag = AgenticRAG(self.pdf_files[0], retriever = 'parent-document', store_path = self.docstore_path, load_version_number = None)
		response = agentic_rag.run(query = 'What is the main idea of the document?', stream = False)
		assert isinstance(response, str)


