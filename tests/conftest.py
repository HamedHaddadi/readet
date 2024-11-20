import pytest 
from os import path, listdir, getcwd 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv('./peruse/configs/keys.env')

@pytest.fixture
def pdf_file_list():
	pdf_path = path.join(getcwd(), 'tests', 'data')
	return [path.join(pdf_path, f) for f in listdir(pdf_path) if f.endswith('.pdf')]

@pytest.fixture
def single_pdf_file(pdf_file_list):
	return pdf_file_list[0]

@pytest.fixture
def random_topic():
	prompt = ChatPromptTemplate.from_template(
		"propose a scientific topic. limit your response to five words")
	llm = ChatOpenAI(model_name = 'gpt-4o-mini', temperature = 0)
	response = prompt | llm
	return response.invoke({"topic": None}).content



