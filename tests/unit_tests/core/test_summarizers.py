import pytest 
from peruse.core.summarizers import PlainSummarizer, MapReduceSummary

@pytest.mark.skip(reason = "tested and passed")
def test_plain_summarizer_for_pdf_inputs(single_pdf_file, pdf_file_list):
	plain_summarizer = PlainSummarizer()
	for pdf_file in pdf_file_list:
		summary = plain_summarizer(pdf_file)
		assert isinstance(summary, str)
	summary = plain_summarizer(single_pdf_file)
	assert isinstance(summary, str)


def test_map_reduce_build():
	map_reduce_summarizer = MapReduceSummary()
	map_reduce_summarizer.build()
	assert map_reduce_summarizer.runnable is not None 

@pytest.mark.skip(reason = "tested and passed")
def test_map_reduce_single_pdf_input(single_pdf_file):
	map_reduce_summarizer = MapReduceSummary()
	map_reduce_summarizer.build()
	summary = map_reduce_summarizer(single_pdf_file)
	assert isinstance(summary, str)

@pytest.mark.skip(reason = "tested and passed")
def test_map_reduce_multiple_pdf_inputs(pdf_file_list):
	map_reduce_summarizer = MapReduceSummary()
	map_reduce_summarizer.build()
	summary = map_reduce_summarizer(pdf_file_list)
	assert isinstance(summary, str)