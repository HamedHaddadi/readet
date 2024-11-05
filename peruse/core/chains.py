# ################################################## #
# 					chains 							 #
# this module contains chains that are used in the   #
# research assistant 								 #
# for instance a title extraction chain 			 #
# ################################################## #
from .. utils import models, prompts 
from langchain_core.prompts import ChatPromptTemplate


# ################################################## #
# title extractor 								     #
# ################################################## #
class TitleExtractor:
	"""
	infers title from a given text
	"""
	def __init__(self, chat_model: str = 'openai-gpt-4o-mini', temperature: float = 0):
		llm = models.configure_chat_model(chat_model, temperature = temperature)
		template = prompts.TEMPLATES['title-extraction']
		prompt = ChatPromptTemplate.from_template(template)
		self.chain = (prompt | llm)
	
	def run(self, text: str) -> str:
		return self.chain.invoke({'text': text}).content 

	def __call__(self, text: str) -> str:
		return self.run(text)
