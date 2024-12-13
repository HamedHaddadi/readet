# ################################# #
# All RAGS and query systems		#
# ################################# #
from typing import (Optional, Dict, List,Union, Any, TypedDict, Annotated, Sequence, Literal)
from langchain_core.documents import Document 
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough 
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser 
from pydantic import BaseModel, Field 
from langchain_core.tools import create_retriever_tool  
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition 
from langgraph.graph.message import add_messages 
from pprint import pprint 
from collections.abc import Callable
from . retrievers import get_retriever, Retriever 
from .. utils import models

AVAILABLE_RETRIEVERS = ['parent-document', 'plain']

# ################################### #
# Prompts    						  #
# ################################### #
PLAIN_RAG_PROMPT = """
	You are an assistant for question-answering tasks. 
		Use the following pieces of retrieved context to answer the question.
		 If you don't know the answer, just say that you don't know. 
		 	Use three sentences maximum and keep the answer concise.
	Question: {question} 
	Context: {context} 
	Answer:
"""
RETRIEVAL_GRADER_PROMPT = """
You are a grader assessing relevance of a retrieved answer to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    	If the answer contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    	Give a binary score 'yes' or 'no' score to indicate whether the answer is relevant to the question.
"""
HALLUCINATION_GRADER_PROMPT = """
		You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
			"""

ANSWER_GRADER_PROMPT = """
		You are a grader assessing whether an answer addresses / resolves a question \n 
     		Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.	
"""

QUESTION_REWRITER_PROMPT = """
		You a question re-writer that converts an input question to a better version that is optimized \n 
     		for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.	
"""

RELEVANCE_GRADER_PROMPT = """
        You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.	
"""

# ################################### #
# Plain RAG  						  #
# ################################### #
class PlainRAG(Callable):
	"""
	Plain RAG class; 
	__init__ parameters:
		documents: documents to be used for creating the retriever
		retriever: type of retriever to be used. Currently supporting parent-document retriever.
			Note that retriever can also be a Retriever object
		embeddings: embeddings model to be used
		chat_model: chat model to be used
		prompt: prompt to be used
		document_loader: document loader to be used
		splitter: splitter to be used 
		load_version_number: version number of the retriever to be loaded from disk; if None a brand new retriever is created and persisted on disk
		store_path: path to store the retriever on disk. if None, retriever is not persisted on disk
	Attributes:
		retriever: retriever object
		llm: language model object
		prompt: prompt object
		runnable: the RAG chain which can be invoked based on the query
	Main methods:
		build(self): builds the RAG chain
		run(self, query: str): invokes the RAG chain
		__call__(self, query: str): invokes the RAG chain
	"""
	def __init__(self, documents: Optional[Union[List[Document], List[str], str]] = None, retriever: Union[str, Retriever] = 'parent-document', 
				embeddings: str = 'openai-text-embedding-3-large', 
					store_path: Optional[str] = None, load_version_number: Optional[Literal['last'] | int] = None,
					chat_model: str = 'openai-gpt-4o-mini',
						prompt: Optional[str] = 'rag',
					document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf', 
						splitter: Literal['recursive', 'token'] = 'recursive',
							kwargs: Dict[str, Any] = {}):
		
		if isinstance(retriever, Retriever):
			self.retriever = retriever
		elif isinstance(retriever, str) and retriever.lower() in AVAILABLE_RETRIEVERS:
			self.retriever = get_retriever(documents = documents, retriever_type = retriever,
				embeddings = embeddings, document_loader = document_loader, splitter = splitter,
				 	store_path = store_path, load_version_number = load_version_number, **kwargs)
		else:
			raise ValueError(f"retriever must be a Retriever object or a string in ['parent-document']")
		
		self.llm = models.configure_chat_model(chat_model, temperature = 0) 
		self.prompt = prompt 
		self.runnable = None  
		self.built = False 
	
	def __setattr__(self, name: str, value: Any) -> None:
		if name == 'prompt':
			if value in ['rag']:
				template = PLAIN_RAG_PROMPT
				super(PlainRAG, self).__setattr__(name, ChatPromptTemplate.from_template(template))
			elif isinstance(value, str) and len(value.split(' ')) > 1:
				super(PlainRAG, self).__setattr__(name, value)
			elif value is None:
				raise ValueError('prompt cannot be None') 
		else:
			super(PlainRAG, self).__setattr__(name, value)
	
	def set_retriever(self, retriever: Retriever) -> None:
		"""
		sets the retriever after initialization
		to generate multi-RAG chains
		"""
		if isinstance(retriever, Retriever):
			self.retriever = retriever
		else:
			raise ValueError(f"setting retriever after initialization must be a Retriever object")
	
	def add_pdf(self, pdf_file: str) -> None:
		self.retriever.add_pdf(pdf_file)
		self.build()

	def build(self) -> None:
		self.runnable = ({'context': self.retriever.runnable, 'question': RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser())
		self.built = True 

	def run(self, query: str) -> str:
		if self.built is False:
			self.build()
		return self.runnable.invoke(query)

	def __call__(self, query: str) -> str:
		return self.run(query)

# ################################### #
# RAG with Citations 				  #
# ################################### #
class AnswerWithCitations(TypedDict):
	"""Answer to the question with citations"""
	answer: str 
	citations: Annotated[List[str], "List of citations (authors and year and publication info) used to answer the question."]

class RAGWithCitations(PlainRAG):
	"""
	RAG with citations class; 
	class inherits from PlainRAG class and accepts identical parameters
	input parameters:
		documents: documents to be used for creating the retriever
		retriever: type of retriever to be used. Currently supporting parent-document retriever.
			Note that retriever can also be a Retriever object
		embeddings: embeddings model to be used
		chat_model: chat model to be used
		prompt: prompt to be used
		document_loader: document loader to be used
		splitter: splitter to be used 
		load_version_number: version number of the retriever to be loaded from disk; if None a brand new retriever is created and persisted on disk
		store_path: path to store the retriever on disk. if None, retriever is not persisted on disk
	Attributes:
		retriever: retriever object
		llm: language model object
		prompt: prompt object
		runnable: the RAG chain which can be invoked based on the query
	Main methods:
		build(self): builds the RAG chain
			run(self, query: str): invokes the RAG chain
			__call__(self, query: str): invokes the RAG chain
	"""
	def __init__(self, documents: Optional[Union[List[Document], List[str], str]] = None, retriever: Union[str, Retriever] = 'parent-document', 
				embeddings: str = 'openai-text-embedding-3-large', 
					store_path: Optional[str] = None, load_version_number: Optional[Literal['last'] | int] = None,
					chat_model: str = 'openai-gpt-4o-mini',
						prompt: Optional[str] = 'rag',
					document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf', 
						splitter: Literal['recursive', 'token'] = 'recursive',
							kwargs: Dict[str, Any] = {}):
		super(RAGWithCitations, self).__init__(documents, retriever, embeddings, store_path,
					load_version_number, chat_model, prompt, document_loader, splitter, kwargs)

	@staticmethod
	def _format_docs(docs: List[Document]) -> str:
		return "\n\n".join(doc.page_content for doc in docs)

	def build(self) -> None:
		rag_chain = ({"question": lambda x: x["question"], 
						"context": lambda x: self._format_docs(x["context"])} | self.prompt | self.llm.with_structured_output(AnswerWithCitations))
		retrieve_context_chain = (lambda x: x["question"]) | self.retriever.runnable 
		self.runnable = RunnablePassthrough.assign(context = retrieve_context_chain).assign(answer = rag_chain)
		self.built = True 
	
	def run(self, query: str, parse = False) -> str:
		if self.built is False:
			self.build()
		response = self.runnable.invoke({"question": query})
		if parse:
			response =  response["answer"]["answer"], response["answer"]["citations"][0]
		return response 
	
	def __call__(self, query: str, parse = True) -> str:
		return self.run(query, parse)
	


# ####################### #
# Self-RAG 				  #
# ####################### #
class GradeRetrieval(BaseModel):
	"""
	Binary score for relevance check on retrieved answers
	"""
	binary_score: str = Field(description = "Retrieved answers are relevant to the question, 'yes' or 'no' ")

class GradeHallucinations(BaseModel):
	"""
	Binary score for hallucination present in the generated answer
	"""
	binary_score: str = Field(description = "Answer is grounded in the factsm 'yes' or 'no' ")

class GraderAnswer(BaseModel):
	"""
	Binary score to assess answers to addressed questions.
	"""
	binary_score: str = Field(description = "Answer addressed the question, 'yes' or 'no' ")

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
	Attributes:
        question: question
        generation: LLM generation
        answers: list of answers; answers are retrieved from a retrieval system
	"""
    question: str
    generation: str
    answers: List[str]

class SelfRAG:
	"""
	self-RAG graph with retrieve, grading and query corection nodes
	This class is used to query a single pdf file 
	Class can be used to query a pdf file using any question. It is also possible to use this class
		to extract structured information using schemas.
	There are three main methods:
		build(self): which builds the graph
		run(self): whicb runs the graph
	parameters 
		documents: documents to be used for creating the retriever
		retriever: type of retriever to be used. Currently supporting parent-document retriever.
			Note that retriever can also be a Retriever object
		embeddings: embeddings model to be used
		chat_model: chat model to be used
		prompt: prompt to be used
		document_loader: document loader to be used
		splitter: splitter to be used 
		load_version_number: version number of the retriever to be loaded from disk; if None a brand new retriever is created and persisted on disk
		store_path: path to store the retriever on disk. if None, retriever is not persisted on disk
	"""
	RECURSION_LIMIT = 40
	def __init__(self, documents: Optional[Union[List[Document],List[str],str]] = None,
		retriever: Union[Literal['parent-document', 'plain'], Retriever] = 'parent-document',
			store_path: Optional[str] = None, load_version_number: Optional[Literal['last'] | int] = None,
			splitter: Literal['recursive', 'token'] = 'recursive',
				document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
					 	 chat_model: str = 'openai-gpt-4o-mini', 
						  	embeddings: str = 'openai-text-embedding-3-large', 
									kwargs: Dict[str, Any] = {}):
		self.runnable = None 

		# note that self.retriever.runnable is the RunnableSequence method to call 
		if isinstance(retriever, Retriever):
			self.retriever = retriever
		elif isinstance(retriever, str) and retriever.lower() in AVAILABLE_RETRIEVERS:
			self.retriever = get_retriever(documents = documents, retriever_type = retriever,
				embeddings = embeddings, document_loader = document_loader, splitter = splitter,
					store_path = store_path, load_version_number = load_version_number, **kwargs)
		else:
			raise ValueError(f"retriever must be a Retriever object or a string in ['parent-document']")
		
		self.retrieval_grader = None 
		self.hallucination_grader = None 
		self.answer_grader = None 
		self.question_rewriter = None 
		self.rag_chain = None 
		self.chat_model = chat_model 

		self.configured = False 
		self.built = False 

	def set_retriever(self, retriever: Retriever) -> None:
		if isinstance(retriever, Retriever):
			self.retriever = retriever
		else:
			raise ValueError(f"setting retriever after initialization must be a Retriever object")
		
	def _configure_grader(self) -> None:
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		struct_llm_grader = llm.with_structured_output(GradeRetrieval)
		system = RETRIEVAL_GRADER_PROMPT
		grade_prompt = ChatPromptTemplate.from_messages(
			[
				("system", system), 
				("human", "Retrieved answer: \n\n {answer} \n\n User question: {question}")
			]
				)
		self.retrieval_grader = grade_prompt | struct_llm_grader 
	
	def _configure_rag_chain(self) -> None:
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		template = PLAIN_RAG_PROMPT
		prompt = ChatPromptTemplate.from_template(template)
		self.rag_chain = prompt | llm | StrOutputParser()
	
	def _configure_hallucination_grader(self) -> None:
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		struct_llm_grader = llm.with_structured_output(GradeHallucinations)
		system  = HALLUCINATION_GRADER_PROMPT
		hallucination_prompt = ChatPromptTemplate.from_messages(
			[
				("system", system), 
				("human", "Set of facts: \n\n {answers} \n\n LLM generation: {generation}")
			])
		self.hallucination_grader = hallucination_prompt | struct_llm_grader
	
	def _configure_answer_grader(self) -> None:
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		struct_llm_grader = llm.with_structured_output(GraderAnswer)
		system = ANSWER_GRADER_PROMPT
		answer_prompt = ChatPromptTemplate.from_messages(
			[
				("system", system), 
				("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
			]
		)
		self.answer_grader = answer_prompt | struct_llm_grader 
	
	def _configure_question_rewriter(self) -> None:
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		system = QUESTION_REWRITER_PROMPT
		rewrite_prompt = ChatPromptTemplate.from_messages(
			[
				("system", system), 
				("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
			]
		)
		self.question_rewriter = rewrite_prompt | llm | StrOutputParser()
	
	# ### Graph nodes ### #
	def retrieve(self, state: Dict[str, Union[str, None]]) -> Dict[str, Union[str, None]]:
		"""
		Retrieve Document objects by querying a retriever
    	Args:
        	state (dict): The current graph state; keys are 'question', 'generation', 'answers'
    	Returns:
        	state (dict): New key added to state, answers, that contains retrieved answers
		"""
		question = state["question"]
		answers = self.retriever.runnable.invoke(question)
		return {"answers": answers, "question": question}
	
	def generate(self, state: Dict[str, Union[str, None]]) -> Dict[str, Union[str, None]]:
		"""
		Generate answer 
		Args:
			state (dict): The current grapg state 
		Returns:
			state (dict): New key added to the state: 'generation', which contains LLM generation 
		"""
		question = state['question']
		answers = state['answers']

		generation = self.rag_chain.invoke({"context": answers, "question": question})
		return {"answers": answers, "question": question, "generation": generation}
	
	def grader_answers(self, state: Dict[str, Union[str, None]]) -> Dict[str, Union[str, Union[List, str]]]:
		"""
    	Determines whether the retrieved answers are relevant to the question.
    	Args:
        	state (dict): The current graph state
    	Returns:
        	state (dict): Updates answers key with only filtered relevant answers		
		"""
		question = state["question"]
		answers = state["answers"]

		filtered = []
		for a in answers:
			score = self.retrieval_grader.invoke(
				{"question": question, "answer": a.page_content})
			grade = score.binary_score 
			if grade == "yes":
				filtered.append(a)
			
		return {"answers": filtered, "question": question}
	
	def transform_query(self, state: Dict[str, Union[str, None]]) -> Dict[str, str]:
		"""
    	Transform the query to produce a better question.
    	Args:
    	    state (dict): The current graph state
    	Returns:
    	    state (dict): Updates question key with a re-phrased question		
		"""
		question = state["question"]
		answers = state["answers"]

		better_question = self.question_rewriter.invoke({"question": question})
		return {"answers": answers, "question": better_question}
	
	def generate_or_not(self, state: Dict[str, Union[str, None]]) -> str:
		"""
		Determines whether to generate an answer, or re-generate a question
		Args:
			state (dict): The current graph state 
		Returns:
			std: Binary decision for next node to call
		"""
		filtered_answers = state["answers"]

		if not filtered_answers:
			return "transform_query"
		else:
			return "generate"
	
	def grade_generation_v_answers_and_question(self, state: Dict[str, Union[str, None]]) -> str:
		"""
		Determines whether the generation is grounded in the answers and answer the question
		Args:
			state (dict): The current graph state 
		Returns:
			str: Decision for next node to call
		"""
		question = state["question"]
		answers = state["answers"]
		generation = state["generation"]

		score = self.hallucination_grader.invoke(
			{"answers": answers, "generation": generation})

		grade = score.binary_score 
		if grade == "yes":
			score = self.answer_grader.invoke({"question": question, "generation": generation})
			grade = score.binary_score
			if grade == "yes":
				return "useful"
			else:
				return "not useful"
		else:
			return "not supported"

	def configure(self) -> None:
		self._configure_grader()
		self._configure_rag_chain()
		self._configure_hallucination_grader()
		self._configure_answer_grader() 
		self._configure_question_rewriter() 
		self.configured = True 
	
	def build(self) -> None:
		if not self.configured:
			self.configure()
		flow = StateGraph(GraphState)
		flow.add_node("retrieve", self.retrieve)
		flow.add_node("grade_answers", self.grader_answers)
		flow.add_node("generate", self.generate)
		flow.add_node("transform_query", self.transform_query)

		flow.add_edge(START, "retrieve")
		flow.add_edge("retrieve", "grade_answers")
		flow.add_conditional_edges(
			"grade_answers", self.generate_or_not, 
				{"transform_query": "transform_query", 
						"generate": "generate",},)

		flow.add_edge("transform_query", "retrieve")
		flow.add_conditional_edges(
			"generate", self.grade_generation_v_answers_and_question, 
			{"not supported": "generate", "useful": END, 
				"not useful": "transform_query",},)
		self.runnable = flow.compile()
		self.built = True 
	
	# other helper methods 
	def _run(self, question: str) -> str:
		inputs = {"question": question}
		response = "did not find an answer to this question!!!"
		try:
			outputs = self.runnable.invoke(inputs)
			response = outputs["generation"]
		except:
			pass 
		return response 

	def _run_stream(self, question: str, stream_mode: Literal['updates', 'values'] = 'updates') -> None:
		inputs = {"question": question}
		response = "did not find an answer to this question!!!"
		try:
			for output in self.runnable.stream(inputs, {"recursion_limit": self.RECURSION_LIMIT}, stream_mode = stream_mode):
				for key,value in output.items():
					pprint(f"Node '{key}' : ")
				pprint("*****")
				response = value["generation"]
		except:
			pass 
		return response		
	
	def run(self, question: str, stream: bool = False, stream_mode: Literal['updates', 'values'] = 'updates') -> Union[str, None]:
		if not self.built:
			self.build()
		response = "did not find an answer to this question!!!"
		if stream:
			response = self._run_stream(question, stream_mode = stream_mode)
		else:
			response = self._run(question)
		return response 
	
	def add_pdf(self, pdf_file: str) -> None:
		self.configured = False 
		self.retriever.add_pdf(pdf_file)
		self.build()

	def __call__(self, question: str, stream: bool = False, stream_mode: Literal['updates', 'values'] = 'updates') -> Union[str, None]:
		return self.run(question, stream = stream, stream_mode = stream_mode)

# ##########################################  #
# Agentic RAG							      #
# an llm is used as an agent to decide		  # 
# between rewriting query or final generation #
#  ########################################### #

class AgentState(TypedDict):
	"""
	state of the graph which is a sequence of instances of BaseMessage types
	"""
	messages: Annotated[Sequence[BaseMessage], add_messages]

class RelevanceGrader(BaseModel):
	"""
	binary score for relevance check of retrieved document
	"""
	binary_score: str = Field(description = "Relevance score which can be either 'yes' or 'no' " )

class AgenticRAG:
	"""
	agentic RAG that runs a RAG on a single pdf file. 
	"""
	def __init__(self, documents: Optional[Union[List[Document],List[str],str]] = None,
			  	retriever: Union[Literal['parent-document', 'plain'], Retriever] = 'parent-document',
				  store_path: Optional[str] = None, load_version_number: Optional[Literal['last'] | int] = None,
				  splitter: Literal['recursive', 'token'] = 'recursive',
						document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
								chat_model: str = "openai-gpt-4o-mini", 
									embeddings: str = "openai-text-embedding-3-large", 
										kwargs: Dict[str, Any] = {}):
		self.runnable = None 
		
		if isinstance(retriever, Retriever):
			self.retriever = retriever
		elif isinstance(retriever, str) and retriever.lower() in AVAILABLE_RETRIEVERS:
			self.retriever = get_retriever(documents = documents, retriever_type = retriever,
				embeddings = embeddings, document_loader = document_loader, splitter = splitter,
					store_path = store_path, load_version_number = load_version_number, **kwargs)
		else:
			raise ValueError(f"retriever must be a Retriever object or a string in ['parent-document']")
		
		self.retriever_tool = create_retriever_tool(self.retriever.runnable, name = 'retriever', description = 'retrieve documents from the retriever')

		self.relevance_chain = None 
		self.generate_chain = None
		self.chat_model = chat_model 

		# runnable is a graph in this case
		self.configured = False
		self.built = False  
		
	def set_retriever(self, retriever: Retriever) -> None:
		if isinstance(retriever, Retriever):
			self.retriever = retriever
			self.retriever_tool = create_retriever_tool(self.retriever.runnable, name = 'retriever', description = 'retrieve documents from the retriever')
		else:
			raise ValueError(f"setting retriever after initialization must be a Retriever object")
	
	def _configure_relevance_check_chain(self) -> None:
		llm = models.configure_chat_model(self.chat_model, temperature = 0, streaming = True)
		model_with_struct = llm.with_structured_output(RelevanceGrader)
		template = RELEVANCE_GRADER_PROMPT
		prompt = PromptTemplate.from_template(template)
		self.relevance_chain = (prompt | model_with_struct)
	
	def _configure_generate_chain(self) -> None:
		template = PLAIN_RAG_PROMPT
		prompt = PromptTemplate.from_template(template)
		llm = models.configure_chat_model(self.chat_model, temperature = 0, streaming = True)
		self.generate_chain = prompt | llm | StrOutputParser()

	# methods that will be used as graph nodes 
	def _check_relevance(self, state: AgentState) -> Literal["generate", "rewrite"]:
		"""
    	determines whether the retrieved documents are relevant to the question
    	Args:
     	   state (messages): the current state 
    	Returns:
    	    std: A decision for whether the document is relevant (generate) or not (rewrite)		
		"""
		messages = state["messages"]
		last_message = messages[-1]
		question = messages[0].content 
		context = last_message.content 

		score_results = self.relevance_chain.invoke({"question": question, "context": context})
		score = score_results.binary_score 
		if score == "yes":
			return "generate"
		elif score == "no":
			return "rewrite"
	
	def _agent(self, state: AgentState) -> Dict[Literal["messages"], Sequence[BaseMessage]]:
		"""
    	Invokes the agent model to generate a response based on the current state. Given
    	the question, it will decide to retrieve using the retriever tool, or simply end.
    	Args:
        	state (messages): The current state
    	Returns:
        	dict: The updated state with the agent response appended to messages
		"""
		messages = state["messages"]
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		llm_with_tools = llm.bind_tools([self.retriever_tool])
		response = llm_with_tools.invoke(messages)
		return {"messages": [response]}
	
	def _rewrite(self, state: AgentState) -> Dict[Literal["messages"], Sequence[BaseMessage]]:
		"""
    	Transform the query to produce a better question.
    	Args:
    	    state (messages): The current state
    	Returns:
    	    dict: The updated state with re-phrased question	
		"""
		messages = state["messages"]
		question = messages[0].content 
		msg = [HumanMessage(content = f""" \n  Look at the input and try
				 to reason about the underlying semantic intent / meaning. \n 
                Here is the initial question:
                \n ------- \n
                {question} 
            \n ------- \n
            Formulate an improved question:""")]
		llm = models.configure_chat_model(self.chat_model, temperature = 0, streaming = True)
		response = llm.invoke(msg)
		return {"messages": [response]}
	
	def _generate(self, state: AgentState) -> Dict[Literal["messages"], Sequence[BaseMessage]]:
		"""
		Generate Answer:
			Args:
				state (messages): the current state 
			Returns:
				dict: The updated state with re-phrased question
		"""
		messages = state["messages"]
		question = messages[0].content 
		last_message = messages[-1]
		context = last_message.content 
		response = self.generate_chain.invoke({"context": context, "question": question})
		return {"messages": [response]}
	
	def configure(self) -> None:
		self._configure_relevance_check_chain()
		self._configure_generate_chain()
		self.configured = True 

	def build(self) -> None:

		if not self.configured:
			self.configure()

		flow = StateGraph(AgentState)
		flow.add_node("agent", self._agent)
		flow.add_node("rewrite", self._rewrite)
		flow.add_node("generate", self._generate)
		retriever_node = ToolNode([self.retriever_tool])
		flow.add_node("retrieve", retriever_node)

		flow.add_edge(START, "agent")
		flow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
		flow.add_conditional_edges("retrieve", self._check_relevance)
		flow.add_edge("generate", END)
		flow.add_edge("rewrite", "agent")
		self.runnable = flow.compile()
		self.built = True 
	
	# run methods 
	def _run_stream(self, query: str):
		inputs = {"messages": [("user", query)]}
		response = "did not find an answer to this question!!! make your question more specific"
		for output in self.runnable.stream(inputs, stream_mode = 'updates'):
			for key, value in output.items():
				if key == 'generate':
					response = value["messages"][-1]
		return response 
		
	
	def _run(self, query: str) -> str:
		inputs = {"messages": [(query),]}
		output = self.runnable.invoke(inputs, stream_mode= "updates")
		if 'generate' in output[-1].keys():
			return output[-1]["generate"]["messages"][-1]
		else:
			return "no response to this query" 
	
	def run(self, query: str, stream: bool = False) -> Union[str, None]:
		if not self.built:
			self.build()
		response = "did not find an answer to this question!!! make your question more specific"
		if not stream:
			response =  self._run(query)
		else:
			response = self._run_stream(query)
		return response 
	
	def add_pdf(self, pdf_file: str) -> None:
		self.configured = False 
		self.retriever.add_pdf(pdf_file)
		self.build()
	
	# to support calling the class as a function
	def __call__(self, query: str, stream: bool = False) -> Union[str, None]:
		return self.run(query, stream = stream)


# ########################################## #
# Rag ensembles						   		 #
# ########################################## #
class RAGEnsemble(Callable):
	"""
	An ensemble of RAGs that are used to answer a question
	parameters:
		documents: documents to be used for creating the retriever
		retriever: type of retriever to be used. Currently supporting parent-document retriever.
			Note that retriever can also be a Retriever object
		store_path: path to store the retriever on disk. if None, retriever is not persisted on disk
		load_version_number: version number of the retriever to be loaded from disk; if None a brand new retriever is created and persisted on disk
		splitter: splitter to be used 
		document_loader: document loader to be used
		chat_model: chat model to be used
		embeddings: embeddings model to be used
		kwargs: additional keyword arguments to be passed to the retriever
	"""
	RAG_TYPES = {'plain_rag': PlainRAG, 'agentic_rag': AgenticRAG, 'self-rag': SelfRAG}
	def __init__(self, documents: List[Document] | List[str] | str,
				retriever: Literal['parent-document'] = 'parent-document',
					store_path: Optional[str] = None, load_version_number: Optional[Literal['last'] | int] = None,
					splitter: Literal['recursive', 'token'] = 'recursive',
						document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
								chat_model: str = "openai-gpt-4o-mini", 
									embeddings: str = "openai-text-embedding-3-large", 
										kwargs: Dict[str, Any] = {}):
		
		self.retriever = get_retriever(documents = documents, retriever_type = retriever,
								 embeddings = embeddings, document_loader = document_loader, splitter = splitter,
								 store_path = store_path, load_version_number = load_version_number, **kwargs)
		self.rags = {}
		for rag_name, rag in self.RAG_TYPES.items():
			self.rags[rag_name] = rag(documents = None, retriever = self.retriever, chat_model = chat_model) 
	
	def run(self, query: str) -> str:
		results = {}
		for rag_name in self.rags.keys():
			results[rag_name] = self.rags[rag_name](query)
		return '\n'.join([f" >>> {rag}: {response}" for rag,response in results.items()])
	
	def __call__(self, query: str) -> str:
		return self.run(query)

# ########################################## #
# Adaptive RAG								 #
# ########################################## #

class AdaptiveRAGState(TypedDict):
	"""
	state of the adaptive RAG graph
	Attributes:
		question: question to be answered
		generation: LLM generation
		documents: list of retrieved documents
	"""
	question: str 
	generation: str 
	documents: List[Document]

class AdaptiveRAG:
	"""
	An adaptive RAG implementation. 
	Method performs a query screening; if the answer is relevant to the index, it is returned. 
	Otherwise, the query is redirected to a web search.
	Query is refined based on analysis of the RAG response.
	input parameters:
		documents: documents to be used for creating the retriever
		documents types: list of documents, list of strings, or a single string
		retriever: type of retriever to be used. Currently supporting parent-document retriever.
			Note that retriever can also be a Retriever object
		retriever types: string in ['parent-document'] or a Retriever object
		store_path: path to store the retriever on disk. if None, retriever is not persisted on disk
		store_path type: string
		load_version_number: version number of the retriever to be loaded from disk; if None a brand new retriever is created and persisted on disk
		load_version_number type: string in ['last'] or an integer
		splitter: splitter to be used 
		splitter type: string in ['recursive', 'token']
		document_loader: document loader to be used
		document_loader type: string in ['pypdf', 'pymupdf']
		chat_model: chat model to be used
		chat_model type: string
		embeddings: embeddings model to be used
		embeddings type: string
		kwargs: additional keyword arguments to be passed to the retriever
	"""
	def __init__(self, documents: List[Document] | List[str] | str,
				retriever: Literal['parent-document'] = 'parent-document',
					store_path: Optional[str] = None, load_version_number: Optional[Literal['last'] | int] = None,
					splitter: Literal['recursive', 'token'] = 'recursive',
						document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
								chat_model: str = "openai-gpt-4o-mini", 
									embeddings: str = "openai-text-embedding-3-large", 
										kwargs: Dict[str, Any] = {}):
		self.runnable = None 
		if isinstance(retriever, Retriever):
			self.retriever = retriever
		elif isinstance(retriever, str) and retriever.lower() in AVAILABLE_RETRIEVERS:
			self.retriever = get_retriever(documents = documents, retriever_type = retriever,
				embeddings = embeddings, document_loader = document_loader, splitter = splitter,
					store_path = store_path, load_version_number = load_version_number, **kwargs)
		else:
			raise ValueError(f"retriever must be a Retriever object or a string in ['parent-document']")
		
		self.chat_model = chat_model
		self.embeddings = embeddings

		# chains that are used in ths RAG
		self.query_router_chain = None 
		self.document_grader_chain = None 
		self.rag_chain = None 
		self.hallucination_grader_chain = None
		self.answer_grader_chain = None 
		self.query_rewriter_chain = None

		# tools
		self.web_search = TavilySearchResults(k = 5) 

		self.configured = False
		self.built = False 
	
	def _configure_query_router_chain(self) -> None:
		"""
		configure the query router chain
		self.query_router_chain is a chain that routes a user query to a vectorstore or web search
		it can be independently invoked and tested by 
			self.query_router_chain.invoke({"question": query})
		"""
		class RouteQuery(BaseModel):
			"""Routes a user query to the most relevant source of information"""
			source: Literal['vectorstore', 'web_search'] = Field(...,
												 description = "Fiven the user query, choose to route it to the vectorstore or the web search")
		
		llm_router = models.configure_chat_model(self.chat_model, temperature = 0)
		llm_with_struct = llm_router.with_structured_output(RouteQuery)
		
		MESSAGE = """ 
				You are an expert at routing a user question to a vectorstore or web search.
				The vectorstore contains documents related to fluid dynamics and physics of fluids.
				Use the vectorstore for questions on these topics. Otherwise, use web-search."""

		prompt = ChatPromptTemplate.from_messages([("system", MESSAGE), ("human", "{question}")])
		self.query_router_chain = prompt | llm_with_struct

	def _configure_document_grader_chain(self) -> None:
		"""
		configure the document grader chain
		self.document_grader_chain is a chain that grades the relevance of a retrieved document to a user question
		it can be independently invoked and tested by 
			self.document_grader_chain.invoke({"question": query, "context": context})
		context is the retrieved document that is obtaibed from the vectorstore:
			docs = self.retrever.runnable.invoke(question)
			context = docs[0].page_content
		"""
		class GradeDocument(BaseModel):
			""" Binary score for relevance check on retrieved documents """
			binary_score: str = Field(description = "documents are relevant to the question, 'yes' or 'no'")
		
		llm_grader = models.configure_chat_model(self.chat_model, temperature = 0)
		llm_with_struct = llm_grader.with_structured_output(GradeDocument)

		MESSAGE = """ 
				You are a grader assessing relevance of a retrieved document to a user question. 
    			If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
    			It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
    			Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
		prompt = ChatPromptTemplate.from_messages([("system", MESSAGE),
											  ("human", "Retrieved documents are: {context} \n and user question is: {question}")])
		self.document_grader_chain = prompt | llm_with_struct
	
	def _configure_rag_chain(self) -> None:
		"""
		configure the rag chain
		"""
		TEMPLATE = """ 
			You are an assistant for question-answering tasks.
			Use the following pieces of retrieved context to answer the question.
			If you don't know the answer, just say that you don't know.
			Use three sentences maximum and keep the answer concise.
			Question: {question} \nContext: {context} Answer:"""
		prompt = ChatPromptTemplate.from_template(TEMPLATE)
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		self.rag_chain = prompt | llm | StrOutputParser()
	
	def _configure_hallucination_grader_chain(self) -> None:
		"""
		configure the hallucination grader chain
		"""
		class GradeHallucination(BaseModel):
			""" Binary score for hallucination present in generated answers """
			binary_score: str = Field(description = "Answer is grounded in facts, 'yes' or 'no'")
		
		hal_llm = models.configure_chat_model(self.chat_model, temperature = 0)
		llm_with_struct = hal_llm.with_structured_output(GradeHallucination)
		MESSAGE = """ You are a grader assessing whether an LLM generation 
		is grounded in / supported by a set of retrieved facts. \n 
     	Give a binary score 'yes' or 'no'. 
		'Yes' means that the answer is grounded in / supported by the set of facts.
		'No' means that the answer is not grounded in / supported by the set of facts.
		"""
		prompt = ChatPromptTemplate.from_messages([("system", MESSAGE),
											  ("human", "Set of facts: {documents} \n LLM generation: {generation}")])
		self.hallucination_grader_chain = prompt | llm_with_struct


	def _configure_answer_grader_chain(self) -> None:
		"""
		configure the answer grader chain
		"""
		class GradeAnswer(BaseModel):
			""" Binary score to assess if answer addresses the question"""
			binary_score: str = Field(description = "Answer addresses the question, 'yes' or 'no'")
		
		llm_grader = models.configure_chat_model(self.chat_model, temperature = 0)
		llm_with_struct = llm_grader.with_structured_output(GradeAnswer)

		MESSAGE = """ You are a grader assessing whether an LLM generation 
		addresses/resolves a question. \n Give a binary score 'yes' or 'no'. 
		'Yes' means that the answer addresses/resolves the question.
		'No' means that the answer does not address/resolve the question.
		"""
		prompt = ChatPromptTemplate.from_messages([("system", MESSAGE),
											  ("human", "User question: {question} \n LLM generated answer is: {generation}")])
		self.answer_grader_chain = prompt | llm_with_struct

	def _configure_query_rewriter_chain(self) -> None:
		"""
		configure the query rewriter chain
		"""
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		SYSTEM = """
			You a question re-writer that converts an input question to a better version that is optimized \n 
     		for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
		"""
		prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "here is the question: {question} \n formulate a better question")])
		self.query_rewriter_chain = prompt | llm | StrOutputParser() 
	
	def configure(self) -> None:
		self._configure_query_router_chain()
		self._configure_document_grader_chain()
		self._configure_rag_chain()
		self._configure_hallucination_grader_chain()
		self._configure_answer_grader_chain()
		self._configure_query_rewriter_chain()
		self.configured = True
	
	# functions to be called on the graph are define here
	def retrieve(self, state: AdaptiveRAGState) -> Dict[str, Any]:
		question = state["question"]
		documents = self.retriever.runnable.invoke(question)
		return {"documents": documents, "question": question}
	
	def generate(self, state: AdaptiveRAGState) -> Dict[str, Any]:
		question = state["question"]
		documents = state["documents"]
		generation = self.rag_chain.invoke({"question": question, "context": documents})
		return {"generation": generation, "question": question, "documents": documents}
	
	def grade_document(self, state: AdaptiveRAGState) -> Dict[str, Any]:
		question = state["question"]
		documents = state["documents"]
		screened_documents = []
		for d in documents:
			score = self.document_grader_chain.invoke({"question": question, "context": d.page_content})
			grade = score.binary_score
			if grade == "yes":
				screened_documents.append(d)
		return {"screened_documents": screened_documents, "question": question}

	def transform_query(self, state: AdaptiveRAGState) -> Dict[str, Any]:
		question = state["question"]
		documents = state["documents"]
		transformed_question = self.query_rewriter_chain.invoke({"question": question})
		return {"question": transformed_question, "documents": documents}
	
	def web_search(self, state: AdaptiveRAGState) -> Dict[str, Any]:
		question = state["question"]
		search_results = self.web_search.invoke({"quesry": question})
		web_results = "\n".join([d["content"] for d in search_results])
		web_results = Document(page_content = web_results)
		return {"documents": [web_results], "question": question}
	
	def route_question(self, state: AdaptiveRAGState) -> Literal['vectorstore', 'web_search']:
		question = state["question"]
		source = self.query_router_chain.invoke({"question": question})
		if source.source == "vectorstore":
			return "vectorstore"
		elif source.source == "web_search":
			return "web_search"
	
	def decide_to_generate(self, state: AdaptiveRAGState) -> Literal['transform_query', 'generate']:
		decision = state["documents"]
		if not decision:
			return "transform_query"
		else:
			return "generate"
	
	def grade_generation(self, state: AdaptiveRAGState) -> Literal['useful', 'not useful', 'not supported']:
		question = state["question"]
		documents = state["documents"]
		generation = state["generation"]
		score = self.hallucination_grader_chain.invoke({"documents": documents, "generation": generation})
		grade = score.binary_score
		if grade == "yes":
			score = self.answer_grader_chain.invoke({"question": question, "generation": generation})
			grade = score.binary_score
			if grade == "yes":
				return "useful"
			else:
				return "not useful"
		else:
			return "not supported"
	
	def build(self) -> None:
		if not self.configured:
			self.configure()
		flow = StateGraph(AdaptiveRAGState)
		flow.add_node("web_search", self.web_search)
		flow.add_node("retrieve", self.retrieve)
		flow.add_node("grade_documents", self.grade_document)
		flow.add_node("generate", self.generate)
		flow.add_node("transform_query", self.transform_query)

		flow.add_conditional_edges(START, self.route_question, {"web_search":"web_search",
														   "vectorstore":"retrieve"})
		flow.add_edge("web_search", "generate")
		flow.add_edge("retrieve", "grade_documents")

		flow.add_conditional_edges("grade_documents", self.decide_to_generate, 
							 	{"transform_query":"transform_query", "generate":"generate"})
		
		flow.add_edge("transform_query", "retrieve")

		flow.add_conditional_edges("generate", self.grade_generation, 
							 	{"useful":END, "not useful":"transform_query",
							 		"not supported":"generate"})
		
		self.runnable = flow.compile()
		self.built = True
	
	# run the graph
	def _run_stream(self, query: str) -> str:
		"""
		use this for debugging only
		"""
		inputs = {"question": query}
		for output in self.runnable.stream(inputs, stream_mode = "updates"):
			for key,value in output.items():
				print(f"the key value is {key} and the value is {value}")
		return value["generation"]

	def _run(self, query: str) -> str:
		inputs = {"question": query}
		response = self.runnable.invoke(inputs)
		return response["generation"]

	
	def run(self, query: str, stream: bool = False) -> str:
		if not self.built:
			self.build()
		if stream:
			return self._run_stream(query)
		else:
			return self._run(query)
	
	def __call__(self, query: str, stream: bool = False) -> str:
		return self.run(query, stream)
	
	def add_pdf(self, pdf_file: str) -> None:
		self.configured = False 
		self.retriever.add_pdf(pdf_file)
		self.build()
	









