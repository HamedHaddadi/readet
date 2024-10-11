
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 

OPENAI_CHAT = {'openai-chat': 'gpt-4o-mini', 
					'openai-gpt-4o': 'gpt-4o'}

OPENAI_EMBEDDING = 'text-embedding-3-large'

def configure_chat_model(model, **model_kw):
	if 'openai' in model:
		temperature = model_kw.get("temperature", 0)
		del model_kw["temperature"]
		return ChatOpenAI(model = OPENAI_CHAT[model], temperature = temperature, **model_kw)

def configure_embedding_model(model, **model_kw):
	if model == 'openai-embedding':
		return OpenAIEmbeddings(model = OPENAI_EMBEDDING)

