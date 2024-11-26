# ############################################################
# evaluations of the RAG and retrievers
# ############################################################

from dotenv import load_dotenv
from os import path
from peruse.core.rags import RAGWithCitations, PlainRAG
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

load_dotenv('./peruse/configs/keys.env')
PDF_PATH = './tests/data'

# evaluation cases #
def test_plain_rag_relevancy_score():
	query = "what is the effect of inertia on viscosity?"
	pdf_file = path.join(PDF_PATH, 'HaddadiMorrisJFM2014.pdf')
	rag = PlainRAG(pdf_file)
	actual_output = rag.run(query)
	docs = rag.retriever.runnable.invoke(query)
	retrieved_context = ['\n\n'.join(docs.page_content) for docs in docs]
	test_case = LLMTestCase(
		input = query,
		actual_output = actual_output,
		retrieval_context = retrieved_context
	)
	metric = AnswerRelevancyMetric()
	metric.measure(test_case)
	print('plain RAG relevancy score is ', metric.score)

def test_RAG_with_citations_relevancy_score():
	query = "what is the effect of inertia on viscosity?"
	pdf_file = path.join(PDF_PATH, 'HaddadiMorrisJFM2014.pdf')
	rag = RAGWithCitations(pdf_file)
	actual_output = rag.run(query)
	docs = rag.retriever.runnable.invoke(query)
	retrieved_context = ['\n\n'.join(docs.page_content) for docs in docs]	
	test_case = LLMTestCase(
		input = query,
		actual_output = actual_output,
		retrieval_context = retrieved_context
	)
	metric = AnswerRelevancyMetric()
	metric.measure(test_case)
	print('RAG with citations relevancy score is ', metric.score)



if __name__ == '__main__':
	test_plain_rag_relevancy_score()
	test_RAG_with_citations_relevancy_score()

