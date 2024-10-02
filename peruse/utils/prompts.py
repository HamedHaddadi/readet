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

# ###### 	Default RAG		  ####### #
SCHEMA_RAG = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer 
        the question. If you don't know the answer, say that you 
        don't know. 
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

# ###### Self-RAG Templates ###### #
RETRIEVAL_GRADER = """
You are a grader assessing relevance of a retrieved answer to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    	If the answer contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    	Give a binary score 'yes' or 'no' score to indicate whether the answer is relevant to the question.
"""
RAG = """
	You are an assistant for question-answering tasks. 
		Use the following pieces of retrieved context to answer the question.
		 If you don't know the answer, just say that you don't know. 
		 	Use three sentences maximum and keep the answer concise.
	Question: {question} 
	Context: {context} 
	Answer:
"""
HALLUCINATION_GRADER = """
		You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
			"""

ANSWER_GRADER = """
		You are a grader assessing whether an answer addresses / resolves a question \n 
     		Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.	
"""

QUESTION_REWRITER = """
		You a question re-writer that converts an input question to a better version that is optimized \n 
     		for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.	
"""

# ### Plain Summarizer ### #
PLAIN_SUMMARY = """
	you will be provided with a text. Write a concise summary and include the main points.
		{text}
"""

# ### Agentic RAG #### #
RELEVANCE_GRADER = """
        You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.	
"""

# ### Keyword extraction from documents ### #
EXTRACT_KEYWORDS = """
    you will be provided with a text. try to find keywords in this text. Avoid including name of the assignee, 
        corporation, or the name of authors or people. Extract keywords that are technically relevant
{text}	
"""

# ### structured outputs ### #
TO_STRUCTURED = """
	you are goven the following text. Try to structure this text using the keywords
		provided to you. 
			{text} """


TEMPLATES = {'suspensions': SUSPENSIONS, 
					'map-reduce': [MAP, REDUCE], 
						'schema-rag': SCHEMA_RAG, 
							'rag': RAG, 
							'plain-summary': PLAIN_SUMMARY, 
						'self-rag': {'retrieval-grader':RETRIEVAL_GRADER,
										'rag': RAG, 
											'hallucination-grader': HALLUCINATION_GRADER, 
												'answer-grader': ANSWER_GRADER, 
													'question-rewriter': QUESTION_REWRITER}, 
						'agentic-rag': {'relevance-grader': RELEVANCE_GRADER}, 
							'extract-keywords': EXTRACT_KEYWORDS, 
								'to-structured': TO_STRUCTURED}


