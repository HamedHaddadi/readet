# test docs module
import pytest 
from peruse.utils import docs
from langchain_core.documents import Document

	 
def test_doc_from_pdf_files_for_pdf_arguments(pdf_file_list, single_pdf_file):
	""" tests the doc_from_pdf_files function for different values of pdf_files arg """
	for pdf_input in [pdf_file_list, single_pdf_file]:
		documents = docs.doc_from_pdf_files(pdf_input)
		assert all([isinstance(doc, Document) for doc in documents])
