
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_anthropic import ChatAnthropic

OPENAI_CHAT = {'openai-gpt-4o-mini': 'gpt-4o-mini', 
					'openai-gpt-4o': 'gpt-4o'}

OPENAI_EMBEDDING = 'text-embedding-3-large'

def configure_chat_model(model, **model_kw):
	if 'openai' in model:
		model = model.replace('openai-', '')
		temperature = model_kw.get("temperature", 0)
		del model_kw["temperature"]
		return ChatOpenAI(model = model, temperature = temperature, **model_kw)
	if 'claude' in model:
		return ChatAnthropic(model = model, **model_kw)

def configure_llm(model, **model_kw):
	if 'openai' in model:
		model = model.replace('openai-', '')
		return OpenAI(model = model, **model_kw)


def configure_embedding_model(model, **model_kw):
	if 'openai' in model:
		model = model.replace('openai-', '')
		return OpenAIEmbeddings(model = model, **model_kw)

