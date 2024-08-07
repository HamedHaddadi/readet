# ################################ #
# prompt templates 				   #
# ################################ #

SUSPENSIONS = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer 
    the question. If you don't know the answer, say that you 
    don't know. 
	If you are asked about the author, find the authors of the text and not the citations. Their name is repeated more than other names.
	If you are asked about the flow type, give a two word answer. 
	If you are asked about the study or characterization method, find the method that authors used with method name or a short description. 
	If you are asked about the volume fraction, search for the term volume fraction, solid fraction and concentration as well and report the range. 
	If you are asked about citation do not include article title.
	Report numeric values for solid fraction, volume fraction or concentration. 
	Report numeric values for particle size, particle density, fluid density and fluid viscosity. Otherwise report null
	If you find more than one value for a quantity, separate them with comma.
	{context}
    Question: {question}
    Helpful Answer:	
"""

# ###### Chain specific prompts ###### #
MAP = """
    the following contains a set of documents.
    {docs}
    Based on this list, identify the main themes. 
    Helpful Answer:
"""

REDUCE = """
    the following is a collection of summaries:
    {docs}
    take these and distill it into a final, consolidated summary of the main theme.
	If possible include details such as numbers and values. 
    Helpful Answer:
"""

TEMPLATES = {'suspensions': SUSPENSIONS, 
					'map-reduce': [MAP, REDUCE]}


