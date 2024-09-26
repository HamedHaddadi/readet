
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 

OPENAI_CHAT = 'gpt-4o-mini'
OPENAI_EMBEDDING = 'text-embedding-3-large'

def configure_chat_model(model, **model_kw):
	if model == 'openai-chat':
		temperature = model_kw.get("temperature", 0)
		del model_kw["temperature"]
		return ChatOpenAI(model = OPENAI_CHAT, temperature = temperature, **model_kw)

def configure_embedding_model(model, **model_kw):
	if model == 'openai-embedding':
		return OpenAIEmbeddings(model = OPENAI_EMBEDDING)

