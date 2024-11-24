from peruse.core.rags import RAGWithCitations, PlainRAG

def test_plain_rag_for_different_retrievers(single_pdf_file):
	retriever_types = ['plain', 'contextual-compression', 'parent-document']
	for retriever_type in retriever_types:
		plain_rag = PlainRAG(single_pdf_file, retriever = retriever_type)
		plain_rag.build()
		response = plain_rag.run(query = 'What is the main idea of the document?')
		assert isinstance(response, str)
