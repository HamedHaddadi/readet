from peruse.core.rags import RAGWithCitations, PlainRAG

def test_plain_rag_for_different_retrievers(single_pdf_file):
	retriever_types = ['plain', 'contextual-compression', 'parent-document']
	for retriever_type in retriever_types:
		plain_rag = PlainRAG(single_pdf_file, retriever = retriever_type)
		plain_rag.build()
		response = plain_rag.run(query = 'What is the main idea of the document?')
		assert isinstance(response, str)

def test_rag_with_citations_includes_citations(single_pdf_file):
	rag_cite = RAGWithCitations(single_pdf_file)
	rag_cite.build()
	_, citations = rag_cite.run(query = 'who are the authors of the document?', parse = True)
	assert set(['Haddadi', 'Morris', 'Kulkarni']).difference(set(citations)) != set()


