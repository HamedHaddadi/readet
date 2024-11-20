
import pytest 
from peruse.core.summarizers import PlainSummarizer, MapReduceSummary

def test_plain_summarizer_for_pdf_inputs(single_pdf_file, pdf_file_list):
	plain_summarizer = PlainSummarizer()
	for pdf_file in pdf_file_list:
		summary = plain_summarizer(pdf_file)
		assert isinstance(summary, str)
	summary = plain_summarizer(single_pdf_file)
	assert isinstance(summary, str)


@pytest.mark.skip(reason = "MapReduceSummary is being transfered to LangGraph version")
def test_map_reduce_for_pdf_inputs(single_pdf_file, pdf_file_list):	
	map_reduce_summarizer = MapReduceSummary()
	for pdf_file in pdf_file_list:
		summary = map_reduce_summarizer(pdf_file)
		assert isinstance(summary, str)
	summary = map_reduce_summarizer(single_pdf_file)
	assert isinstance(summary, str)
