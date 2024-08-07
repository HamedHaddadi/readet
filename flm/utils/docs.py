
from typing import Dict, List 

# ################################ #
# utilities to work with documents #
# ################################ #

def format_doc_object(doc: List, replacements: Dict[str, str]) -> List:
	"""
	takes a doc object from PyPDFLoader and replace a str with another
	Returns: formatted doc
	"""
	for page in doc:
		for find,replace in replacements.items():
			page.page_content = page.page_content.replace(find, replace)
	return doc 

