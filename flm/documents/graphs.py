# ########################################### #
# collection of graphs to work with documents #
# ########################################### #
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.pydantic_v1 import BaseModel, Field 
from langgraph.graph import END, START, StateGraph 
from pprint import pprint 
from collections.abc import Callable 
from typing import Optional, Dict, Any, List, Union
from typing_extensions import TypedDict   
from pathlib import Path 
from .. utils import models, prompts 

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


class SelfRAGSinglePDF(Callable):
	"""
	self-RAG graph with retrieve, grading and query corection nodes
	This class is used to query a single pdf file 
	Class can be used to query a pdf file using any question. It is also possible to use this class
		to extract structured information using schemas 
	"""
	RECURSION_LIMIT = 40
	def __init__(self, pdf_file: str, chunk_size: int = 4000,
	 				chunk_overlap: int = 150,
					 	 chat_model: str = 'openai-chat', 
						  	embedding_model: str = 'openai-embedding',
								  	replacements: Optional[Dict[str, str]] = None):

		self.retrieval_grader = None 
		self.hallucination_grader = None 
		self.answer_grader = None 
		self.question_rewriter = None 
		self.rag_chain = None 
		self.retriever = None 
		self.graph = None 

		self.splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
						 chunk_overlap = chunk_overlap, add_start_index = True, 
						 		separators = ["\n\n", "\n", "(?<=\. )", " ", ""])
		self.split_docs = pdf_file 

		self._configure_retrieval_grader(chat_model = chat_model, embedding_model = embedding_model)
		self._configure_rag_chain(chat_model = chat_model)
		self._configure_hallucination_grader(chat_model = chat_model)
		self._configure_answer_grader(chat_model = chat_model)
		self._configure_question_rewriter(chat_model = chat_model)
		self._configure_graph()
	

	def __setattr__(self, name: str, value: Any) -> None:
		if name == 'split_docs':
			if Path(value).exists() and Path(value).is_file() and '.pdf' in value:
				loader = PyPDFLoader(value)
				docs = self.splitter.split_documents(loader.load())
				super(SelfRAGSinglePDF, self).__setattr__(name, docs)
		else:
			super(SelfRAGSinglePDF, self).__setattr__(name, value)	

	
	def _configure_retrieval_grader(self, chat_model: str = 'openai-chat',
		 					embedding_model: str = 'openai-embedding') -> None:
		vectorstore = Chroma.from_documents(documents = self.split_docs, collection_name = "rag-chroma", 
							embedding = models.configure_embedding_model(embedding_model))
		self.retriever = vectorstore.as_retriever()
		llm = models.configure_chat_model(chat_model, temperature = 0)
		struct_llm_grader = llm.with_structured_output(GradeRetrieval)
		system = prompts.TEMPLATES['self-rag']['retrieval-grader']
		grade_prompt = ChatPromptTemplate.from_messages(
			[
				("system", system), 
				("human", "Retrieved answer: \n\n {answer} \n\n User question: {question}")
			]
				)
		self.retrieval_grader = grade_prompt | struct_llm_grader 
	
	def _configure_rag_chain(self, chat_model: str = 'openai-chat') -> None:
		llm = models.configure_chat_model(chat_model, temperature = 0)
		template = prompts.TEMPLATES['self-rag']['rag']
		prompt = ChatPromptTemplate.from_template(template)
		self.rag_chain = prompt | llm | StrOutputParser()
	
	def _configure_hallucination_grader(self, chat_model: str = 'openai-chat') -> None:
		llm = models.configure_chat_model(chat_model, temperature = 0)
		struct_llm_grader = llm.with_structured_output(GradeHallucinations)
		system  = prompts.TEMPLATES['self-rag']['hallucination-grader']
		hallucination_prompt = ChatPromptTemplate.from_messages(
			[
				("system", system), 
				("human", "Set of facts: \n\n {answers} \n\n LLM generation: {generation}")
			])
		self.hallucination_grader = hallucination_prompt | struct_llm_grader
	
	def _configure_answer_grader(self, chat_model: str = 'openai-chat') -> None:
		llm = models.configure_chat_model(chat_model, temperature = 0)
		struct_llm_grader = llm.with_structured_output(GraderAnswer)
		system = prompts.TEMPLATES['self-rag']['answer-grader']
		answer_prompt = ChatPromptTemplate.from_messages(
			[
				("system", system), 
				("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
			]
		)
		self.answer_grader = answer_prompt | struct_llm_grader 
	
	def _configure_question_rewriter(self, chat_model: str = 'chat-openai') -> None:
		llm = models.configure_chat_model(chat_model, temperature = 0)
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
		print(">>> RETRIEVE <<<")
		question = state["question"]
		answers = self.retriever.invoke(question)
		return {"answers": answers, "question": question}
	
	def generate(self, state: Dict[str, Union[str, None]]) -> Dict[str, Union[str, None]]:
		"""
		Generate answer 
		Args:
			state (dict): The current grapg state 
		Returns:
			state (dict): New key added to the state: 'generation', which contains LLM generation 
		"""
		print(">>> GENERATE <<<")
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
		print(">>> CHECK ANSWER RELEVANCE TO QUESTION <<<")
		question = state["question"]
		answers = state["answers"]

		filtered = []
		for a in answers:
			score = self.retrieval_grader.invoke(
				{"question": question, "answer": a.page_content})
			grade = score.binary_score 
			if grade == "yes":
				print("**GRADE: ANSWER RELEVANT!**")
				filtered.append(a)
			else:
				print("!!GRADE: ANSWER NOT RELEVANT!!")
		return {"answers": filtered, "question": question}
	
	def transform_query(self, state: Dict[str, Union[str, None]]) -> Dict[str, str]:
		"""
    	Transform the query to produce a better question.
    	Args:
    	    state (dict): The current graph state
    	Returns:
    	    state (dict): Updates question key with a re-phrased question		
		"""
		print(">> TRANSFORM QUERY <<")
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
		print(">> ASSESS GRADED ANSWERS <<")
		filtered_answers = state["answers"]

		if not filtered_answers:
			print("!! DECISION: ALL NASWERS ARE NOT RELEVANT TO QUESTION. TRANSFORM QUERY ---")
			return "transform_query"
		else:
			print("** DECISION: GENERATE >>>")
			return "generate"
	
	def grade_generation_v_answers_and_question(self, state: Dict[str, Union[str, None]]) -> str:
		"""
		Determines whether the generation is grounded in the answers and answer the question
		Args:
			state (dict): The current graph state 
		Returns:
			str: Decision for next node to call
		"""
		print(">> CHECK HALLUCINATION <<")
		question = state["question"]
		answers = state["answers"]
		generation = state["generation"]

		score = self.hallucination_grader.invoke(
			{"answers": answers, "generation": generation})

		grade = score.binary_score 
		if grade == "yes":
			print("*** DECISION: GENERATION IS GROUNDED IN ANSWERS ***")
			print("?? GRADE GENERATION VS QUESTION ??")
			score = self.answer_grader.invoke({"question": question, "generation": generation})
			grade = score.binary_score
			if grade == "yes":
				print("DECISION ==> GENERATION ADDRESSES THE QUESTION!! ")
				return "useful"
			else:
				print("DECISION ==> GENERATION DOES NOT ADDRESS THE QUESTION !!")
				return "not useful"
		else:
			print("DECISION: GENERATION IS NOT GROUNDED IN ANSWERS, RETRY! ")
			return "not supported" 
	
	def _configure_graph(self) -> None:
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
		self.graph = flow.compile()
	
	def __call__(self, question: str) -> str:
		inputs = {"question": question}
		for output in self.graph.stream(inputs, {"recursion_limit": self.RECURSION_LIMIT}):
			for key,value in output.items():
				pprint(f"Node '{key}' : ")
			pprint("*****")
		pprint(value["generation"])

