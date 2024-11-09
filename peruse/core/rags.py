# ################################# #
# All RAGS and query systems		#
# ################################# #
from typing import (Optional, Dict, List,Union, Any, TypedDict, Annotated, Sequence, Literal)
from langchain_community.document_loaders import PyPDFLoader, pdf  
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_core.documents import Document 
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import BasePromptTemplate, PromptTemplate, ChatPromptTemplate, format_document  
from langchain_core.runnables.base import RunnableSequence 
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser 
from pydantic import BaseModel, Field 
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool  
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition 
from langgraph.graph.message import add_messages 
from pprint import pprint 
from pathlib import Path 
from collections.abc import Callable
from . retrievers import get_retriever 
from .. utils import prompts, models, docs
from .. utils.schemas import SCHEMAS 

# ##### classes that are useful for all RAGS ##### #
class RetrieverRunnable(BaseModel):
	"""
	Takes a retriever as an attribute and invokes a query on the retriever
	"""
	retriever: BaseRetriever
	prompt: BasePromptTemplate = PromptTemplate.from_template("{page_content}")

	def run(self, query: str) -> str:
		docs = self.retriever.invoke(query)
		return "\n\n".join(format_document(doc, self.prompt) for doc in docs)

class RetrieverTool(BaseTool):
	name: str = "retriever_tool"
	description: str = """retriever tool which is used to retriever documents from a 
				vector store. """
	retriever: RetrieverRunnable 

	def _run(self, query: str) -> str:
		return self.retriever.run(query)

# ################################### #
# Plain RAG  						  #
# ################################### #
class PlainRAG(Callable):
	"""
	Plain RAG class; 
	__init__ parameters:
		documents: documents to be used for creating the retriever
		retriever: type of retriever to be used
		embeddings: embeddings model to be used
		chat_model: chat model to be used
		prompt: prompt to be used
		document_loader: document loader to be used
		splitter: splitter to be used 
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
	def __init__(self, documents: List[Document] | List[str] | str, retriever: Literal['plain', 'contextual-compression', 'parent-document'] = 'contextual-compression', 
				embeddings: str = 'openai-text-embedding-3-large', 
					chat_model: str = 'openai-gpt-4o-mini',
						prompt: Optional[str] = 'rag',
					document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf', 
						splitter: Literal['recursive', 'token'] = 'recursive',
							kwargs: Dict[str, Any] = {}):
		self.retriever = get_retriever(documents = documents, retriever_type = retriever,
				embeddings = embeddings, document_loader = document_loader, splitter = splitter, **kwargs)
		self.llm = models.configure_chat_model(chat_model, temperature = 0) 
		self.prompt = prompt 
		self.runnable = None  
		self._built = False 
	
	@property 
	def built(self):
		return self._built 

	@built.setter 
	def built(self, status: bool):
		if status is True and self._built is False:
			self._built = True 
		elif status is False and self._built is True:
			self._built = False 

	def __setattr__(self, name: str, value: Any) -> None:
		if name == 'prompt':
			if value in ['rag']:
				template = prompts.TEMPLATES['rag']
				super(PlainRAG, self).__setattr__(name, ChatPromptTemplate.from_template(template))
			elif isinstance(value, str) and len(value.split(' ')) > 1:
				super(PlainRAG, self).__setattr__(name, value)
			elif value is None:
				raise ValueError('prompt cannot be None') 
		else:
			super(PlainRAG, self).__setattr__(name, value)
	
	def add_pdf(self, pdf_file: str) -> None:
		self.retriever.add_pdf(pdf_file)

	def build(self) -> None:
		self.runnable = ({'context': self.retriever.runnable, 'question': RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser())
		self._built = True 

	def run(self, query: str) -> str:
		if self.built is False:
			self.built = True 
			self.build()
		return self.runnable.invoke(query)

	def __call__(self, query: str) -> str:
		return self.run(query)


# #################################### #
# 	Structured Outputs				   #
# Extracts information from text into  # 
# 			Schemas					   #	
# #################################### #
DEFAULT_PROMPT = """You are an expert extraction algorithm. 
     Only extract relevant information from the text. 
        If you do not find the attribute you are asked to extract, 
            return null."""	

# plain extactor chain
def extract_schema_plain(schemas: Union[List[str], str], 
				chat_model:str = 'openai-gpt-4o-mini', 
			 			temperature: int = 0) -> Union[Dict[str, RunnableSequence], RunnableSequence]:
	prompt = ChatPromptTemplate.from_messages([("system", DEFAULT_PROMPT), ("human", "{text}")])
	llm = models.configure_chat_model(chat_model, temperature = temperature)
	if isinstance(schemas, list):
		return {schema: (prompt | llm.with_structured_output(schema = SCHEMAS[schema])) for schema in schemas}
	elif isinstance(schemas, str):
		return (prompt | llm.with_structured_output(schema = SCHEMAS[schemas]))


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
	"""
	RECURSION_LIMIT = 40
	def __init__(self, documents: List[Document] | List[str] | str,
		retriever: Literal['plain', 'contextual-compression', 'parent-document'] = 'contextual-compression',
			splitter: Literal['recursive', 'token'] = 'recursive',
				document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf',
					 	 chat_model: str = 'openai-gpt-4o-mini', 
						  	embedding_model: str = 'openai-text-embedding-3-large', 
									kwargs: Dict[str, Any] = {}):
		self.runnable = None 

		# note that self.retriever.runnable is the RunnableSequence method to call 
		# retriver object is created here in case more pdf needs to be added in the RAG
		self.retriever = get_retriever(documents = documents, retriever_type = retriever,
				embeddings = embedding_model, document_loader = document_loader, splitter = splitter, **kwargs)
		
		self.retrieval_grader = None 
		self.hallucination_grader = None 
		self.answer_grader = None 
		self.question_rewriter = None 
		self.rag_chain = None 

		self.chat_model = chat_model 

		self._configured = False 
		self._built = False 

	@property 
	def configured(self):
		return self._configured 
	
	@configured.setter 
	def configured(self, status: bool):
		if self._configured is False and status is True:
			self._configured = True 

	@property 
	def built(self):
		return self._built 

	@built.setter
	def built(self, status: bool):
		if self._built is False and status is True:
			self._built = True  
	
	def _configure_grader(self) -> None:
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		struct_llm_grader = llm.with_structured_output(GradeRetrieval)
		system = prompts.TEMPLATES['self-rag']['retrieval-grader']
		grade_prompt = ChatPromptTemplate.from_messages(
			[
				("system", system), 
				("human", "Retrieved answer: \n\n {answer} \n\n User question: {question}")
			]
				)
		self.retrieval_grader = grade_prompt | struct_llm_grader 
	
	def _configure_rag_chain(self) -> None:
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		template = prompts.TEMPLATES['self-rag']['rag']
		prompt = ChatPromptTemplate.from_template(template)
		self.rag_chain = prompt | llm | StrOutputParser()
	
	def _configure_hallucination_grader(self) -> None:
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		struct_llm_grader = llm.with_structured_output(GradeHallucinations)
		system  = prompts.TEMPLATES['self-rag']['hallucination-grader']
		hallucination_prompt = ChatPromptTemplate.from_messages(
			[
				("system", system), 
				("human", "Set of facts: \n\n {answers} \n\n LLM generation: {generation}")
			])
		self.hallucination_grader = hallucination_prompt | struct_llm_grader
	
	def _configure_answer_grader(self) -> None:
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		struct_llm_grader = llm.with_structured_output(GraderAnswer)
		system = prompts.TEMPLATES['self-rag']['answer-grader']
		answer_prompt = ChatPromptTemplate.from_messages(
			[
				("system", system), 
				("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
			]
		)
		self.answer_grader = answer_prompt | struct_llm_grader 
	
	def _configure_question_rewriter(self) -> None:
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		system = prompts.TEMPLATES['self-rag']['question-rewriter']
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
		self._configured = True 
	
	def build(self) -> None:

		if not self._configured:
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
		self._built = True 
	
	# other helper methods 
	def _run(self, question: str) -> str:
		inputs = {"question": question}
		outputs = self.runnable.invoke(inputs)
		return outputs["generation"]

	def _run_stream(self, question: str, stream_mode: Literal['updates', 'values'] = 'updates') -> None:
		inputs = {"question": question}
		for output in self.runnable.stream(inputs, {"recursion_limit": self.RECURSION_LIMIT}, stream_mode = stream_mode):
			for key,value in output.items():
				pprint(f"Node '{key}' : ")
			pprint("*****")
		pprint(value["generation"])		
	
	def run(self, question: str, stream: bool = False, stream_mode: Literal['updates', 'values'] = 'updates') -> Union[str, None]:
		if stream:
			self._run_stream(question, stream_mode = stream_mode)
		else:
			return self._run(question)
	
	def add_pdf(self, pdf_file: str) -> None:
		self.retriever.add_pdf(pdf_file)

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
	def __init__(self, pdf_file: str, chunk_size: int = 2000, 
						chunk_overlap: int = 150, 
								chat_model: str = "openai-gpt-4o-mini", 
									embedding_model: str = "openai-text-embedding-3-large"):
		self.retriever = None 
		self.retriever_tool = None 
		self.relevance_chain = None 
		self.generate_chain = None

		# runnable is a graph in this case
		self.runnable = None 
		self._configured = False
		self._built = False  

		self.chat_model = chat_model 
		self.embedding_model = embedding_model 
		self.splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
						 chunk_overlap = chunk_overlap, add_start_index = True)

		self.split_docs = pdf_file 
	
	@property 
	def configured(self):
		return self._configured 
	
	@configured.setter
	def configured(self, status: bool):
		if self._configured is False and status is True:
			self._configured = True 
	
	@property
	def built(self):
		return self._built 
	
	@built.setter
	def built(self, status: bool):
		if self._built is False and status is True:
			self._built = True 
	
	def __setattr__(self, name: str, value: Any) -> None:
		if name == 'split_docs':
			value_path = Path(value)
			if value_path.exists() and value_path.is_file() and '.pdf' in value:
				pages = pdf.PyMuPDFLoader(value, extract_images = True)
				if pages is not None:
					pages = pages.load()
					pages = self.splitter.split_documents(pages)
				super(AgenticRAG, self).__setattr__(name, pages)
			else:
				raise FileExistsError(f"pdf file {value} does not exist")
		else:
			super(AgenticRAG, self).__setattr__(name, value)
	
	def _configure_retriever(self) -> None:
		vectorstore = Chroma.from_documents(documents = self.split_docs, collection_name = "rag-chroma", 
								embedding = models.configure_embedding_model(self.embedding_model)) 
		self.retriever = vectorstore.as_retriever()
	
	def _configure_retriever_tool(self) -> None:
		self.retriever_tool = RetrieverTool(retriever = RetrieverRunnable(retriever = self.retriever))

	def _configure_relevance_check_chain(self) -> None:
		llm = models.configure_chat_model(self.chat_model, temperature = 0)
		model_with_struct = llm.with_structured_output(RelevanceGrader)
		template = prompts.TEMPLATES['agentic-rag']['relevance-grader']
		prompt = PromptTemplate.from_template(template)
		self.relevance_chain = (prompt | model_with_struct)
	
	def _configure_generate_chain(self) -> None:
		template = prompts.TEMPLATES["rag"]
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

		score = self.relevance_chain.invoke({"question": question, "context": context}).binary_score 
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
		self._configure_retriever()
		self._configure_retriever_tool()
		self._configure_relevance_check_chain()
		self._configure_generate_chain()
		self._configured = True 

	def build(self) -> None:

		if not self._configured:
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
		self._built = True 
	
	# run methods 
	def _run_stream(self, query: str):
		inputs = {"messages": [("user", query)]}
		for output in self.runnable.stream(inputs, stream_mode = 'updates'):
			for key, value in output.items():
				pprint(f"Output from node '{key}':")
				pprint(" >>> <<<")
				pprint(value, indent=2, width=80, depth=None)
				pprint("\n---\n")
	
	def _run(self, query: str) -> str:
		inputs = {"messages": [(query),]}
		output = self.runnable.invoke(inputs, stream_mode= "values")
		return output["messages"][-1].content 
	
	def run(self, query: str, stream: bool = False) -> Union[str, None]:
		if not self._built:
			self.build()

		if not stream:
			return self._run(query)
		else:
			self._run_stream(query)
	
	# to support calling the class as a function
	def __call__(self, query: str, stream: bool = False) -> Union[str, None]:
		return self.run(query, stream = stream)

# ##################################### #
RAGS = {'self-single-pdf': SelfRAG, 
				'agentic-rag-pdf': AgenticRAG}

EXTRACTORS = {'plain': extract_schema_plain}








