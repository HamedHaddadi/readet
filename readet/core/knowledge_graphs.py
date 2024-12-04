# ################################# #
# methods to build knowledge graphs #
# ################################# #
from os import path, listdir, PathLike 
from pathlib import Path 
from typing import TypeVar, Optional, List, Union, Any  
from collections.abc import Callable 
from langchain_community.graphs import Neo4jGraph 
from langchain_experimental.graph_transformers import LLMGraphTransformer 
from langchain_core.documents import Document 
from .. utils import models 
from . import summarizers 


# ################################### #
# Knowledge Graph Builder			  #
# ################################### #
KG = TypeVar('KG', bound = 'KnowledgeGraph')
class KnowledgeGraph(Callable):
	"""
	Builds a knowledge graph from a text
	KnowledgeGraph builder is a chain.
	"""
	def __init__(self, summaries: str, store_graph: bool = True, 
				allowed_nodes: Optional[List[str]] = None,
					 allowed_relations: Optional[List[str]] = None):
		self.summaries = summaries 
		self.store_graph = store_graph 
		self.allowed_nodes = allowed_nodes 
		self.allowed_relations = allowed_relations 
		self.graph = Neo4jGraph()
		self.graph_doc = None 
	
	def _build(self):
		llm = models.configure_chat_model(model = 'openai-gpt-4o-mini', temperature = 0)
		llm_transformer = LLMGraphTransformer(llm = llm, allowed_nodes = self.allowed_nodes, 
					allowed_relationships = self.allowed_relations)
		documents = [Document(self.summaries)]
		self.graph_doc = llm_transformer.convert_to_graph_documents(documents)
	
	@property 
	def nodes(self) -> List:
		if self.graph_doc is None:
			self._build()
		return self.graph_doc[0].nodes 
	
	@property
	def relations(self) -> List:
		if self.graph_doc is None:
			self._build()
		return self.graph_doc[0].relationships 
	
	def __call__(self) -> None:
		self._build()
		if self.store_graph:
			self.graph.add_graph_documents(self.graph_doc)

	@classmethod 
	def from_pdf(cls, pdf: Union[str, PathLike, List], chat_model: str = 'openai-gpt-4o-mini', temperature: int = 0,
				 store_graph: bool = True,  summarizer: str = 'plain', allowed_nodes: Optional[List[str]] = None, 
						allowed_relations: Optional[List[str]] = None, **summarizer_kw: Any) -> KG:

		if not isinstance(pdf, (list, tuple)):
			p = Path(pdf)
			if p.is_dir():
				pdf = [path.join(p, pdf_file) for pdf_file in listdir(p) if '.pdf' in pdf_file]
			elif p.is_file:
				pdf = [p]
				
		summarize_method = {'plain': summarizers.PlainSummarizer}[summarizer]
		summaries = '\n'.join([summarize_method.from_pdf(pdf_file, chat_model = chat_model, 
								temperature = temperature, **summarizer_kw)() for pdf_file in pdf])
		return cls(summaries, store_graph = store_graph, allowed_nodes = allowed_nodes, 
										allowed_relations = allowed_relations)