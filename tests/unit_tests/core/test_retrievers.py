from peruse.core.retrievers import PlainRetriever  

def test_plain_retriever_add_pdf(single_pdf_file, pdf_file_list):
	plain_retriever = PlainRetriever.from_pdf(single_pdf_file)
	num_docs = len(plain_retriever.vector_store.get()["ids"])
	plain_retriever.add_pdf(pdf_file_list)
	assert len(plain_retriever.vector_store.get()["ids"]) > num_docs 


