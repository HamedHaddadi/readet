# ################################################## #
# include agentic systems for report writing 
# ################################################## #
import operator 
from typing import Union, List, Optional, Literal, Dict, Any, Annotated, TypedDict 
from pydantic import BaseModel, Field 
from os import PathLike 
from .. utils import docs, models 
from .. core.rags import CitationRAG
from .. core.retrievers import Retriever
# langchain and langgraph 
from langchain_core.messages import HumanMessage, SystemMessage 
from langgraph.graph import StateGraph, START, END 
from langgraph.constants import Send 

# ################################################## 
# Prompts
# ################################################## 
DEFAULT_STRUCTURE = """
This report type focuses on describing a concept applicable to a scientific field.
The report structure should include:
1. Introduction (no research needed)
   	- Brief overview of the topic area
   	- Context for the scientific concepts
2. Main Body Sections:
   - One dedicated section for each concept related to the main concept
   - Each section should examine in detail:core concepts and example use cases
3. Conclusion 
   - distill the article and give a gist of results
   - include recommendations for further study
"""

REPORT_PLANNER_QUERY_WRITER = """
You are an expert technical writer, helping to plan a report. 
The report will be focused on the following topic:
{topic}
The report structure will follow these guidelines:
{report_structure}
Your goal is to generate {number_of_queries} queries that will help gather comprehensive information for planning the report sections. 

The query should:
1. Be related to the topic 
2. Help satisfy the requirements specified in the report organization
Make the query specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.
"""

REPORT_PLANNER = """
You are an expert technical writer, helping to plan a report.
Your goal is to generate the outline of the sections of the report. 
The overall topic of the report is:
{topic}
The report should follow this structure: 
{report_structure}
You should reflect on the topic, report structure and the following context to plan the sections of the report: 
{context}

Now, generate the sections of the report. Each section should have the following fields:
- Name - Name for this section of the report.
- Description - Brief overview of the main topics and concepts to be covered in this section.
- Research - Whether to ask for more context for this section of the report.
- Content - The content of the section, which you will leave blank for now.

Consider which sections require more context and research.  
For example, introduction and conclusion will not require more context because 
    they will distill information from other parts of the report.
"""

# section writer prompts (Query and main writer)
QUERY_FOR_SECTION_WRITER = """
Your goal is to generate queries that will gather comprehensive information for writing a technical report section.
Description for this section:
{section_description}
When generating {number_of_queries} search queries, ensure they:
1. Cover different aspects of the topic (e.g., features and applications if you can find these information)
2. Include specific technical terms related to the topic
Your queries should be:
- Specific enough to avoid generic results
- Technical enough to capture detailed information
- Diverse enough to cover all aspects of the section plan
- Focused on authoritative sources (documentation, technical blogs, academic papers)
"""

SECTION_WRITER = """
You are an expert technical writer crafting one section of a technical report.

title of this section is:
{section_title}

and here is a description:
{section_description}

Guidelines for writing:

1. Technical Accuracy:
- Include specific information 
- Reference concrete metrics
- Cite official documentation
- Use technical terminology precisely

2. Length and Style:
- Strict 400-500 word limit
- No marketing language
- Technical focus
- Write in simple, clear language
- Start with your most important insight in **bold**
- Use short paragraphs (4-5 sentences max)

3. Structure:
- Use ## for section title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
- End with ### Sources that references the below source material formatted as:
  * List each source with titleand  page number
  * Format: `- Title : page number`

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the section content
- Focus on your single most important point

4. Use this source material to help write the section:
{context}

5. Quality Checks:
- Exactly 400-500 words (excluding title and sources)
- Careful use of only ONE structural element (table or list) and only if it helps clarify your point
- Starts with bold insight
- No preamble prior to creating the section content
- Sources cited at end
"""

FINAL_SECTION_WRITER = """
You are an expert technical writer crafting the final section for a report. 
The report title is 
{section_title}

and the section description is 
{section_description}

and here is the result of context research for other sections:
{completed_sections}

1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 400-500 word limit
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 400-500 word limit
- End with specific next steps or implications

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point

4. Quality Checks:
- For introduction: 150-200 word limit, # for report title, no structural elements, no sources section
- For conclusion: 1200-300 word limit, ## for section title, only ONE structural element at most, no sources section
- Markdown format
- Do not include word count or any preamble in your response
"""


# Models defined in the global scope to avoid unnecessary workarounds
class Section(BaseModel):
	title: str = Field(description = "The title of the section")
	description: str = Field(description = "brief overview of the main topics and concepts in this report")
	research: bool = Field(description = "whether this section requires research for more contextor not")
	content: str = Field(description = "the content of the section")
	
class Sections(BaseModel):
	sections: List[Section] = Field(description = "list of sections in the report")

class ContextQuery(BaseModel):
	context_query: str = Field(None, description = "query for retrieving context from a Retrieval Augmented Generation system")

class Queries(BaseModel):
	queries: List[ContextQuery] = Field(description = "list of queries for retrieving context from a Retrieval Augmented Generation system")

# state of the section writer graph
class SectionState(TypedDict):
	""" State of the section generation graph"""
	section: Section
	number_of_queries: int 
	search_queries: List[ContextQuery]
	context_str: str 
	report_sections_from_research: str 
	completed_sections: List[Section]

class SectionOutputState(TypedDict):
	completed_sections: List[Section]

# state of the final report writer graph
class ReportState(TypedDict):
	""" State of the document generation graph"""
	topic: str
	report_structure: str 
	number_of_queries: int 
	sections: List[Section]
	completed_sections: Annotated[List, operator.add]
	report_sections_from_research: str 
	final_report: str 

class ReportOutputState(TypedDict):
	final_report: str 


# ################################################## #
# ReportWriterWithRAG
# ################################################## #
class ReportWriterWithRAG:
	"""
	Report writing agent that uses RAG to create context
	RAG is populated with PDF files
	"""

	def __init__(self, pdf_files: List[PathLike] | str, report_structure: Union[str, PathLike] = DEFAULT_STRUCTURE,
			  retriever: Union[str, Retriever] = 'parent-document', num_queries: int = 5,
					embeddings: str = 'openai-text-embedding-3-large',
						store_path: Optional[str] = None, load_version_number: Optional[Literal['last'] | int] = None,
							pkl_object: Optional[str| Dict] = None, chat_model: str = 'openai-gpt-4o-mini',
					document_loader: Literal['pypdf', 'pymupdf'] = 'pypdf', 
						splitter: Literal['recursive', 'token'] = 'recursive',
							kwargs: Dict[str, Any] = {}) -> None:
		self.report_structure = report_structure 
		pdf_files =  docs.get_pdf_files(pdf_files)
		self.researcher = CitationRAG(pdf_files, retriever = retriever, embeddings = embeddings, store_path = store_path,
					load_version_number = load_version_number, pkl_object = pkl_object, chat_model = chat_model,
						document_loader = document_loader, splitter = splitter, kwargs = kwargs)
		self.llm = models.configure_chat_model(chat_model, temperature = 0)
		self.num_queries = num_queries

		# main agents 
		self.report_graph = None 
		self.section_graph = None 
		self.built = False 

	def research_context(self, queries: List[str]) -> List[str]:
		return [self.researcher(q) for q in queries]
	
	def generate_report_plan(self, state: ReportState) -> List[Section]:
		topic = state["topic"]
		report_structure = state["report_structure"]
		number_of_queries = state["number_of_queries"]
		query_llm = self.llm.with_structured_output(Queries)

		formatted_query_prompt = REPORT_PLANNER_QUERY_WRITER.format(topic = topic, 
															  report_structure = report_structure,
															    number_of_queries = number_of_queries)
		query_results = query_llm.invoke([SystemMessage(content = formatted_query_prompt), 
											HumanMessage(content = """ Generate search queries that 
												will help with planning the sections of the report.""")])
		query_list = [q.context_query for q in query_results.queries]
		research_context = self.research_context(query_list)
		
		formatted_planner_prompt = REPORT_PLANNER.format(topic = topic, 
														report_structure = report_structure,
														context = research_context)
		section_planner_llm = self.llm.with_structured_output(Sections)
		section_planner_results = section_planner_llm.invoke([SystemMessage(content = formatted_planner_prompt), 
						HumanMessage(content = """Generate the sections of the report.
							 Your response must include a 'sections' field containing a list of sections. 
                Each section must have: name, description, plan, research, and content fields.""")])
		return {"sections": section_planner_results.sections}	
	
	def generate_section_queries(self, state: SectionState):
		number_of_queries = state["number_of_queries"]
		section = state["section"]
		section_query_llm = self.llm.with_structured_output(Queries)

		system_message = QUERY_FOR_SECTION_WRITER.format(section_description = section.description,
												   	number_of_queries = number_of_queries)
		queries = section_query_llm.invoke([SystemMessage(content = system_message), 
											HumanMessage(content = """Generate search queries that 
												will help with writing this section of the report.""")])
		return {"search_queries": queries}
	

	def prepare_context(self,state: SectionState):
		search_queries = state["search_queries"]
		query_list = [q.context_query for q in search_queries.queries]
		docs = self.research_context(query_list)
		context_str = ",".join(docs)
		return {"context_str": context_str}
	

	def write_section(self, state: SectionState):
		section = state["section"]
		context_str = state["context_str"]

		system_message = SECTION_WRITER.format(section_title = section.title,
												section_description = section.description,
												context = context_str)

		section_content = self.llm.invoke([SystemMessage(content = system_message)] + 
										[HumanMessage(content = "generate a report section based on the context")])
		section.content = section_content.content
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
		self.section_graph = flow.compile()
	
	def initiate_section_writing(self,state: ReportState):
		""" dispatcher function for section writing"""
		return [Send("write_section_with_context_research", {"section": s,
													    "number_of_queries": self.num_queries})
															for s in state["sections"]]
	
	def write_final_section(self, state: SectionState):
		section = state["section"]
		report_sections_from_research = state["report_sections_from_research"]
		system_instructions = FINAL_SECTION_WRITER.format(section_title = section.title,
														section_description = section.description,
														completed_sections = report_sections_from_research)
		
		section_writer_results = self.llm.invoke([SystemMessage(content = system_instructions),
															HumanMessage(content = "Generate a reports ection based on the provided resources")])
		section.content = section_writer_results.content
		return {"completed_sections": [section]}
	
	@staticmethod
	def format_section(completed_sections: List[Section]) -> str:
		formatted_str = ""
		for ids,section in enumerate(completed_sections,1):
			formatted_str += f""" {'*'*40} Section: {ids} {'*'*40}
			Title: {section.title}
			Description: {section.description}
			Requires Research: {section.research}
			Content: {section.content if section.content else '[Not yet Written]'} 
			"""
		return formatted_str
	
	def gather_completed_sections(self, state: ReportState):
		completed_sections = state["completed_sections"]
		report_sections_from_research = self.format_section(completed_sections)
		return {"report_sections_from_research": report_sections_from_research}
	
	def initiate_final_section_writing(self, state: ReportState):
		return [Send("write_final_section", {"section": s,
												"report_sections_from_research": state["report_sections_from_research"]})
														for s in state["sections"] if not s.research]
	
	def compile_final_report(self, state: ReportState):
		sections = state["sections"]
		completed_sections = {s.title: s.content for s in state["completed_sections"]}
		for section in sections:
			section.content = completed_sections[section.title]
		final_report = "\n\n".join([s.content for s in sections])
		return {"final_report": final_report}
	
	def build_report_graph(self):
		flow = StateGraph(ReportState, output = ReportOutputState)
		flow.add_node("generate_report_plan", self.generate_report_plan)
		flow.add_node("write_section_with_context_research", self.section_graph)
		flow.add_node("gather_completed_sections", self.gather_completed_sections)
		flow.add_node("write_final_section", self.write_final_section)
		flow.add_node("compile_final_report", self.compile_final_report)
		flow.add_edge(START, "generate_report_plan")
		
		flow.add_conditional_edges("generate_report_plan", self.initiate_section_writing,
							  ["write_section_with_context_research"])
		flow.add_edge("write_section_with_context_research", "gather_completed_sections")
		flow.add_conditional_edges("gather_completed_sections", self.initiate_final_section_writing, ["write_final_section"])
		flow.add_edge("write_final_section", "compile_final_report")
		flow.add_edge("compile_final_report", END)
		self.report_graph = flow.compile()
	
	def build(self):
		self.build_section_graph()
		self.build_report_graph()
		self.built = True 
	
	def __call__(self, topic: str):
		if not self.built:
			self.build()
		report = self.report_graph.invoke({"topic": topic, "report_structure": self.report_structure, 
								"number_of_queries": self.num_queries})
		return report
		

													
	
	


	

	



	

		
		





