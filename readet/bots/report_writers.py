# ################################################## #
# include agentic systems for report writing 
# ################################################## #
import operator 
from tqdm import tqdm 
from typing import Union, List, Literal, Dict, Any, Annotated, TypedDict 
from pydantic import BaseModel, Field 
from os import path, makedirs, listdir 
from .. utils import models 
from .. core.rags import CitationRAG
from .. core.retrievers import Retriever
from . agents import ReAct 
# langchain and langgraph 
from langchain_core.messages import HumanMessage, SystemMessage 
from langgraph.graph import StateGraph, START, END 
from langgraph.constants import Send 


# ################################################## #
# Write with Scholar Search
# builds the RAG based on queries that it generates
# ################################################## #
# instructions for the report structure
# ################################################## #
REPORT_STRUCTURE = """
This report focuses on describing a concept applicable to this topic.
{topic}
The report structure should include:
1.Abstract
    - Brief summary of what is covered in the main report

1. Introduction 
   - Gives context to the reader about what is covered in the report
   - requires research and using references

2. Main Body Sections:
   - Does not need to have a separate preamble such as an introduction
   - Each section should examine in detail:
     - Core concepts
     - example use cases
   
3. Conclusion 
   - distill the article and give a gist of results
   - include recommendations for further study """

# ################################################## #
INITIAL_QUERY_GENERATION_INSTRUCTIONS = """
I want a collection of good queries in order to search
in Google Scholar and Arxiv to write a review article about the following topic:
{topic}
Here are main points that describe what I am interested in covering in this report:
{main_points}
start your questions with the word 'Question'
"""

# ################################################## #
REPORT_PLANNER_INSTRUCTIONS = """
You are an expert technical writer helping to create an outline for a scientific review article.
Your goal is to generate the outline of this report. 
Plan a structure for a comprehensive review article about the following topic:
{topic}

The report structure must follow the following structure:
{report_structure}

But you can decide about the number and title of sections based on the following context 
{context}

- Name - Name for this section of the report.
- Description - Brief overview of the main topics and concepts to be covered in this section.
- Research - Whether to ask for more context for this section of the report.
- Content - The content of the section, which you will leave blank for now.

Consider which sections require more context and research.  
For example, abstract and conclusion will not require more context because 
    it distills information from other parts of the report.

"""
# ################################################## #
QUERY_GENERATION_INSTRUCTIONS = """
I want a collection of good questions to gather comprehensive information for writing a section of a technical report. 
    here is the title of this section
    {title}
    and here is a brief description of this section 
    {description}
    Limit the number of questions to {number_of_queries} and start your questions with the word 'Question' 

    Questions should:
    1. cover different aspects of the topic
    2. include specific technical terms
    3. be specific enough to avoid generic results
    4. technical enough to capture detailed information
"""
# ################################################## #
SECTION_WRITER_INSTRUCTIONS = """
I want to write a section for a technical report. 
Here are some information about this section. 
The title is 
{title}
and description is 
{description}. 

The following context is also available for writing the section
{context}.

Guidelines:
1. Make full use of the context and use all the sources in line. For example: 
Many complex fluids exhibit a yield stress, meaning they do not flow until a certain stress threshold is exceeded.
This is a critical property in applications involving suspensions and emulsions, 
where the material may behave like a solid until sufficient stress is applied (Eow & Ghadiri, 2002)

2.No preamble or intro before writing the section.Strictly focus on this section.

2. Length and style:
 Technical focus
 Maximum 2000 words for each section
 simple and clear language

3. Structure and formatting:
- Use ## for section title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
- put references in paranthesis
"""
# ################################################## #
FINAL_SECTION_WRITER_INSTRUCTIONS = """
    I want to write a section for a review article based on the content of other sections of this report.
    Here are some information and instructions:
        section topic is
		{topic}
        and section description is
		{description}

    And here are sections that have been written which form a context for you to synthesize a new section:
    {sections_from_research}

Section specific guidelines:
For Intruduction:
- Use # for report title (Markdown format)
- 800-1000 words 
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Include NO structural elements (no lists or tables)
- mention sources in line with the text and inside paranthesis.

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 400-500 word limit
- End with specific next steps or implications
- No sources section needed

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point

4. Quality Checks:
- Markdown format
- Do not include word count or any preamble in your response
"""


class Section(BaseModel):
	title: str = Field(description = "The title of the section")
	description: str = Field(description = "brief overview of the main topics and concepts in this report")
	research: bool = Field(description = "whether this section requires research for more contextor not")
	content: str = Field(description = "the content of the section")

class Sections(BaseModel):
	sections: List[Section] = Field(description = "list of sections in the report")


# graph states #
class ReportState(TypedDict):
	""" State of the document generation graph"""
	topic: str
	main_points: str 
	max_number_of_queries: int 
	sections: List[Section]
	completed_sections: Annotated[List, operator.add]
	report_sections_from_research: str 
	final_report: str 

class ReportInputState(TypedDict):
	topic: str 
	main_points: str

class ReportOutputState(TypedDict):
	final_report: str 

class SectionState(TypedDict):
	section: Section 
	search_queries: List[str]
	context: str 
	report_sections_from_research: str 
	completed_sections: List[Section]

class SectionOutputState(TypedDict):
	completed_sections: List[Section]

	
# main class
class WriterWithScholarSearch:
	"""
	Writer that uses scholar search to generate context for the report
	max_results: maximum number of results to return from scholar search per query 
	"""
	def __init__(self, save_path: str, max_search: int = 10, max_questions: int = 10,
			  main_llm: str = "openai-o1-mini", helper_llm: str = "openai-gpt-4o",
			    retriever: Union[str, Retriever] = 'parent-document', 
					embeddings: str = 'openai-text-embedding-3-large',
						chat_model: str = 'openai-gpt-4o-mini',
								document_loader: Literal['pypdf', 'pymupdf'] = 'pymupdf', 
									splitter: Literal['recursive', 'token'] = 'recursive',
										kwargs: Dict[str, Any] = {}) -> None:
		
		self.rag_arguments = {key:value for key,value in locals().items() 
						if key not in ['self', 'max_search', 'question_llm', 'save_path']} 
		if not path.exists(save_path):
			makedirs(save_path)
		self.save_path = save_path

		rag_store_path = path.join(save_path, "rag")
		if not path.exists(rag_store_path):
			makedirs(rag_store_path)
		self.rag_arguments["store_path"] = rag_store_path

		self.max_search = max_search
		self.max_questions = max_questions

		self.main_llm = models.configure_chat_model(main_llm)
		self.helper_llm = models.configure_chat_model(helper_llm)
		self.search_agent = None 
		self.rag = None 

		# graphs
		self.section_writer = None 
		self.report_writer = None 
		self.built = False 
	
	# Helper functions #
	@staticmethod
	def _parse_questions(message: str) -> List[str]:
		lines = message.splitlines()
		return [line for line in lines if 'Question' and '?' in line]
	
	# Main functions #
	def plan_report(self, state: ReportState) -> List[Section]:
		topic = state["topic"]
		f_report_structure = REPORT_STRUCTURE.format(topic = topic)
		main_points = state["main_points"]
		f_query_generation_instructions = INITIAL_QUERY_GENERATION_INSTRUCTIONS.format(topic = topic, main_points = main_points)

		message = self.main_llm.invoke([HumanMessage(content = f_query_generation_instructions)])	
		questions = self._parse_questions(message.content)

		self.search_agent = ReAct(tools = ["google_scholar_search", "arxiv_search", "pdf_download"], 
				added_prompt = "You are an AI assistant; search and download all papers that are related to the user query",
				  save_path = self.save_path, max_results = self.max_search)
		
		print(f"Searching for {len(questions)} questions")
		for q in tqdm(questions[:1]):
			self.search_agent(f"search and download all papers related to this question {q}")
		
		pdf_files = [path.join(self.save_path, f) for f in listdir(self.save_path) if f.endswith('.pdf')]
		print(f"Building the vector space")
		self.rag = CitationRAG(pdf_files, retriever = self.rag_arguments["retriever"], 
								embeddings = self.rag_arguments["embeddings"], 
								store_path = self.rag_arguments["store_path"], 
								chat_model = self.rag_arguments["chat_model"])
		context = "\n".join([self.rag(q) for q in questions])
		plan_llm = self.helper_llm.with_structured_output(Sections)

		f_report_planner_instructions = REPORT_PLANNER_INSTRUCTIONS.format(topic = topic, 
																	 	report_structure = f_report_structure, 
																		context = context)
		report_sections = plan_llm.invoke([SystemMessage(content = f_report_planner_instructions), 
											HumanMessage(content = """Generate the sections of this report.
                                          Your response must include a 'sections' field containing a list of sections. 
                                          Each section must have: title, description, research and content fields""")])
		return {"sections": report_sections.sections}

	def generate_section_queries(self, state: SectionState):
		section = state["section"]
		f_query_generation_instructions = QUERY_GENERATION_INSTRUCTIONS.format(title = section.title, 
																				description = section.description, 
																				number_of_queries = self.max_questions)
		message = self.main_llm.invoke([HumanMessage(content = f_query_generation_instructions)])
		search_queries = self._parse_questions(message.content)
		return {"search_queries": search_queries}
	
	def prepare_context(self, state: SectionState):
		search_queries = state["search_queries"]
		context = "\n".join([self.rag(q) for q in search_queries])
		return {"context": context}

	def write_section(self, state: SectionState):
		section = state["section"]
		context = state["context"]
		f_section_writer_instructions = SECTION_WRITER_INSTRUCTIONS.format(title = section.title, 
																			description = section.description, 
																			context = context)
		message = self.main_llm.invoke([HumanMessage(content = f_section_writer_instructions)])
		section.content = message.content
		return {"completed_sections": [section]}

	def build_section_graph(self):
		flow = StateGraph(SectionState, output = SectionOutputState)
		flow.add_node("generate_section_queries", self.generate_section_queries)
		flow.add_node("prepare_context", self.prepare_context)
		flow.add_node("write_section", self.write_section)
		flow.add_edge(START, "generate_section_queries")
		flow.add_edge("generate_section_queries", "prepare_context")
		flow.add_edge("prepare_context", "write_section")
		flow.add_edge("write_section", END)
		self.section_writer = flow.compile()

	# report write nodes and graph #
	# helper 
	@staticmethod
	def _format_section(sections: List[Section]) -> str:
		formatted_str = ""
		for idx, section in enumerate(sections):
			formatted_str += f"""{'='*60}
                        Section {idx}: {section.title}
                        {'='*60}
                        Description:
                        {section.description}
                        Requires Research: 
                        {section.research}
                        Content:
                            {section.content if section.content else '[Not yet written]'}"""
		return formatted_str

	def initiate_section_writing(self, state: ReportState):
		return [Send("write_section_with_research", {"section": s, "max_number_of_questions": self.max_questions})
		  for s in state["sections"] if s.research]
	
	def gather_completed_sections(self, state: ReportState):
		completed_report_sections = self._format_section(state["completed_sections"])
		return {"report_sections_from_research": completed_report_sections}
	
	def initiate_final_section_writing(self, state: ReportState):
		return [Send("write_final_sections", 
			   {"section": s, "report_sections_from_research": state["report_sections_from_research"]})
				 for s in state["sections"] if not s.research]
	
	def write_final_sections(self, state: SectionState):
		section = state["section"]
		completed_sections = state["report_sections_from_research"]
		f_final_section_writer_instructions = FINAL_SECTION_WRITER_INSTRUCTIONS.format(topic = section.title, 
																		description = section.description, 
																	sections_from_research = completed_sections)
		message = self.main_llm.invoke([HumanMessage(content = f_final_section_writer_instructions)])
		section.content = message.content
		return {"completed_sections": [section]}
	
	def compile_final_report(self, state: ReportState):
		sections = state["sections"]
		completed_sections = {s.title: s.content for s in state["completed_sections"]}
		completed_keys = completed_sections.keys()
		for section in sections:
			if section.title in completed_keys:
				section.content = completed_sections[section.title]
		all_sections = "\n\n".join([s.content for s in sections])
		return {"final_report": all_sections}
	
	def build_report_graph(self):
		flow = StateGraph(ReportState, input = ReportInputState, output = ReportOutputState)
		flow.add_node("plan_report", self.plan_report)
		flow.add_node("write_section_with_research", self.section_writer)
		flow.add_node("gather_completed_sections", self.gather_completed_sections)
		flow.add_node("write_final_sections", self.write_final_sections)
		flow.add_node("compile_final_report", self.compile_final_report)

		flow.add_edge(START, "plan_report")
		flow.add_conditional_edges("plan_report", self.initiate_section_writing, ["write_section_with_research"])
		flow.add_edge("write_section_with_research", "gather_completed_sections")
		flow.add_conditional_edges("gather_completed_sections", self.initiate_final_section_writing, ["write_final_sections"])
		flow.add_edge("write_final_sections", "compile_final_report")
		flow.add_edge("compile_final_report", END)

		self.report_writer = flow.compile()
	
	def build(self):
		self.build_section_graph()
		self.build_report_graph()
		self.built = True 
	
	def write(self, topic: str = None, main_points: str = None):
		if not self.built:
			self.build()
		state = {"topic": topic, "main_points": main_points}
		report = self.report_writer.invoke(state)
		return report["final_report"]
	
	def __call__(self, *args, **kwargs):
		pass 

		

		
	


		
		



	




	

											




													
	
	


	

	



	

		
		





