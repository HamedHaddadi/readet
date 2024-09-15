
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 

OPENAI_CHAT = 'gpt-4o-mini'
OPENAI_EMBEDDING = 'text-embedding-3-large'

def configure_chat_model(model, **model_kw):
	if model == 'openai-chat':
		return ChatOpenAI(model = OPENAI_CHAT, temperature = model_kw['temperature'])

def configure_embedding_model(model, **model_kw):
	if model == 'openai-embedding':
		return OpenAIEmbeddings(model = OPENAI_EMBEDDING)

