#!/usr/bin/env python3
from os import path, makedirs, getcwd, listdir 
from datetime import datetime  
import argparse
from pathlib import Path 
from dotenv import load_dotenv 
from peruse.core.prebuilt_components import SchemaFromPDF
from peruse.core.prebuilt_components import QueryPDFAndSearch
from peruse.core.rags import SelfRAGSinglePDF

KEYS = '/Users/hamedhaddadi/Documents/ComplexFluidInformatics/builder/fluidllm/peruse/configs/keys.env'
load_dotenv(KEYS)

def make_dir(module_name):
	dirname = path.join(getcwd(), module_name + '_on_' + datetime.now().strftime('%m-%d-%H-%M'))
	if not path.exists(dirname):
		makedirs(dirname)
	return dirname 

# this function does not work due to an error in langchain_core.pydantic_v1 
def run_scholar_search() -> None:
	pass 

def run_schema_from_pdf() -> None:
	schemas = input("Enter the Schema name; it must already be available in utils/schemas: ")
	output_dir = input(""" Enter directory of the existing
							 schema to append to existing csv file else pass: \n""")
	pdf = input("""pdf file or directory to query:
						 Enter a path to query all pdf files and append them to a single csv file: \n""")
	if len(output_dir) == 0:
		output_dir = make_dir('data_from_pdf')
	pdf = Path(pdf)
	if pdf.is_file():
		SchemaFromPDF(schemas, save_path = output_dir)(pdf)
	elif pdf.is_dir():
		pdf_files = [path.join(pdf, pdf_file) for pdf_file in listdir(pdf)]
		for pdf_file in pdf_files:
			SchemaFromPDF(schemas, save_path = output_dir)(pdf_file)

def run_pdf_query() -> None:
	"""
	query a pdf file using SelfRAG 
	"""
	pdf = input("Enter the full path and name of the pdf file: \n")
	self_rag = SelfRAGSinglePDF(pdf)
	go_on = True 
	while go_on:
		query = input("Enter your question regarding this document: \n")
		self_rag(query)
		print('**********')
		go = input("Do you have more questions? Y/N? \n")
		if go == 'Y':
			go_on = True
		elif go == 'N':
			print('Have a good day! ;)')
			go_on = False 
		else:
			print('Your response was not clear! Finish')
			go_on = False  

def run_query_pdf_and_search() -> None:
	"""
	function to call QueryPDFAndSearch
		this function first queries a pdf file using RAGs 
		then performs a Google search for more information about queried points. 
	Example:
		searching for raw materials used in an experimental research paper
	"""
	pdf = input("Enter the full path and name of the pdf file: \n")
	question = input("""Enter your question: \n
		Note that your question better be asked in one sentence and be as precise as possible! Example: What are the raw materials used in this document? \n
			""")
	added_message = input(""" Do you have any added message? \n
					for example: If you are search for raw materials in a document you can add 'give a brief description and find the name of manufacturers and vendors' \n""")
	query_search = QueryPDFAndSearch(pdf, added_message= added_message)
	query_search(question)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'reads command line of modules to run')
	parser.add_argument('appnames', nargs = '*', type = str, help = 'names of apps to run')
	apps = parser.parse_args().appnames 
	for app in apps:
		{'schema_from_pdf': run_schema_from_pdf, 
				'query_pdf': run_pdf_query, 
					'query_pdf_and_search': run_query_pdf_and_search}[app]()
		

