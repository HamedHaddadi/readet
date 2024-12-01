import pytest 
from unittest import TestCase
from os import path, getcwd, makedirs, listdir
from shutil import rmtree
from peruse.core.retrievers import PlainRetriever, ParentDocument, get_retriever

@pytest.mark.skip(reason = "tested and to avoid exhausting API calls")
def test_plain_retriever_add_pdf(single_pdf_file, pdf_file_list):
	plain_retriever = PlainRetriever.from_pdf(single_pdf_file)
	num_docs = len(plain_retriever.vector_store.get()["ids"])
	plain_retriever.add_pdf(pdf_file_list)
	assert len(plain_retriever.vector_store.get()["ids"]) > num_docs


# parent document retriever tests
class TestParentDocument(TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		test_save_path = path.join(getcwd(), 'test_results')
		if not path.exists(test_save_path):
			makedirs(test_save_path)
		cls.save_path = test_save_path
		pdf_path = path.join(getcwd(), 'tests', 'data')
		cls.pdf_files = [path.join(pdf_path, f) for f in listdir(pdf_path) if f.endswith('.pdf')]
		cls.template_kw  = {'store_path': path.join(cls.save_path, 'docstore'), 
					'load_version_number': None,'embeddings': 'openai-text-embedding-3-large',
					'document_loader': 'pypdf', 'parent_splitter': 'token',
					'child_splitter': 'recursive', 'parent_chunk_size': 2000,
					'child_chunk_size': 2000, 'parent_chunk_overlap': 200, 'child_chunk_overlap': 100}
	
#	@classmethod
#	def tearDownClass(cls) -> None:
#		rmtree(cls.save_path)
	
	def test_docstore_persist_on_dist_init_from_pdf(self):
		store_path = path.join(self.save_path, 'docstore')
		if not path.exists(store_path):
			makedirs(store_path)
		parent_retriever = ParentDocument.from_pdf(self.pdf_files[0], store_path = store_path)
		parent_retriever.build()
		assert path.exists(path.join(store_path, 'parent_document_retriever_1.pkl'))
	
	def test_init_by_loading_from_disk_no_pdf(self):
		store_path = path.join(self.save_path, 'docstore')
		parent_retriever = ParentDocument.load_from_disk(store_path, pdf_files = None, 
																	version_number = 'last')
		assert isinstance(parent_retriever, ParentDocument)
	
	def test_docstore_length_before_and_after_loading_from_disk(self):
		store_path = path.join(self.save_path, 'docstore')
		if not path.exists(store_path):
			makedirs(store_path)
		parent_retriever = ParentDocument.from_pdf(self.pdf_files[0], store_path = store_path)
		parent_retriever.build()
		num_docs = len(parent_retriever.runnable.docstore.store.values())
		
		parent_retriever2 = ParentDocument.load_from_disk(store_path, pdf_files = None, 
																	version_number = 'last')
		parent_retriever2.build()
		num_docs2 = len(parent_retriever2.runnable.docstore.store.values())
		assert num_docs == num_docs2

	def test_get_retriever_for_parent_document_from_pdf(self):
		store_path = self.template_kw['store_path']
		parent_retriever = get_retriever(self.pdf_files[0], retriever_type = 'parent-document', **self.template_kw)
		assert isinstance(parent_retriever, ParentDocument)
		assert len(parent_retriever.runnable.docstore.store.values()) > 0
		assert path.exists(path.join(store_path, 'parent_document_retriever_1.pkl'))
	
	def test_get_retriever_for_parent_document_from_disk(self):
		parent_retriever = get_retriever(self.pdf_files[0], retriever_type = 'parent-document', **self.template_kw)
		new_kw = self.template_kw.copy()
		new_kw['load_version_number'] = 'last'
		parent_retriever2 = get_retriever(self.pdf_files[0], retriever_type = 'parent-document', **new_kw)
		assert 2*len(parent_retriever.runnable.docstore.store.values()) == len(parent_retriever2.runnable.docstore.store.values())
		
		









