from os import path, makedirs, getcwd, listdir 
from datetime import datetime  
import argparse
from pathlib import Path 
from dotenv import load_dotenv 
from flm.agents.documents import ScholarSearch 
from flm.builders.documents import SchemaFromPDF 

KEYS = '/Users/hamedhaddadi/Documents/ComplexFluidInformatics/builder/fluidllm/flm/configs/keys.env'
load_dotenv(KEYS)

def make_dir(module_name):
	dirname = path.join(getcwd(), module_name + '_on_' + datetime.now().strftime('%m-%d-%H-%M'))
	if not path.exists(dirname):
		makedirs(dirname)
	return dirname 

def run_scholar_search() -> None:
	output_dir = make_dir('scholar_search')
	num_inputs = input("Enter the number of articles to search: ")
	scholar = ScholarSearch(num_results = num_inputs)
	query = input(""" enter your question:
	 				For example: search all articles that are helpful in understanding fluid flow : \n""")
	scholar(query, output_dir)

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
		print(pdf_files)
		for pdf_file in pdf_files:
			SchemaFromPDF(schemas, save_path = output_dir)(pdf_file)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'reads command line of modules to run')
	parser.add_argument('appnames', nargs = '*', type = str, help = 'names of apps to run')
	apps = parser.parse_args().appnames 
	for app in apps:
		{'search_scholar': run_scholar_search, 
			'schema_from_pdf': run_schema_from_pdf}[app]()

