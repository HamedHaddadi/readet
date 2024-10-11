
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 

OPENAI_CHAT = {'openai-gpt-4o-mini': 'gpt-4o-mini', 
					'openai-gpt-4o': 'gpt-4o'}

OPENAI_EMBEDDING = 'text-embedding-3-large'

def configure_chat_model(model, **model_kw):
	if 'openai' in model:
		model = model.replace('openai-', '')
		temperature = model_kw.get("temperature", 0)
		del model_kw["temperature"]
		return ChatOpenAI(model = model, temperature = temperature, **model_kw)

def configure_embedding_model(model, **model_kw):
	if 'openai' in model:
		model = model.replace('openai-', '')
		return OpenAIEmbeddings(model = model, **model_kw)

