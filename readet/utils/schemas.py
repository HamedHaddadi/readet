# ################################# #
# schemas for info extractor models #
# ################################# #

from typing import Optional, List, Sequence 
from langchain_core.pydantic_v1 import BaseModel, Field 

class Suspensions(BaseModel):
	"""
	the schema of the Suspension. Information extractor chain uses this schema class
		to extract information from a RAG generated text from a pdf file. 
	:param citation: citation of the article 
	:type citation: str
	:param solid_fractions : solid fractions mentioned in the text
	:type solid_fractions : str
	:characterization_method : the characterization methods used by authors
	:type characterization_method: str
	:materials : materials used by authors in this study 
	:type materials: str
	:particles : shape and type of particles; example can be sphreical hard spheres
	:type particles : str
	:particle_size: size of particles in this study
	:type particle_size: str
	:particle_density: density of particles 
	:type particle_density: str 
	:fluid_density: density of the suspending fluid 
	:type fluid_density: str 
	:fluid_viscosity: viscosity of the suspending fluid 
	:type fluid_viscosity: str
	:dimensionless_numbers: dimensionless numbers in this work
	:type dimensionless_numbers: str 
	"""
	#authors: Optional[str] = Field(default = None, description="authors of the work")
	citation: Optional[str] = Field(default = None, description="citation of the current article")
	solid_fractions: Optional[str] = Field(default = None, description="the volume fractions in the study")
	characterization_method: Optional[str] = Field(default = None, description="smethod of characterization used by authors")
	materials: Optional[str] = Field(default = None, description="materials used by authors")
	particles: Optional[str] = Field(default = None, description="shape and type of particles in the suspension")
	particle_size: Optional[str] = Field(default = None, description="size of particles used in the study")
	particle_density: Optional[str] = Field(default = None, description = "density of particles")
	fluid_density: Optional[str] = Field(default = None, description="density of the suspending fluid")
	fluid_viscosity: Optional[str] = Field(default = None, description = "viscosity of the suspending fluid")
	dimensionless_numbers: Optional[str] = Field(default = None, description = "dimensionless numbers that are important in this work")

# schemas that can be used to extract information from a paper
class RawMaterials(BaseModel):
	"""
	The schema of the Raw Materials. 
	Information extractor chain uses this schema class
	to extract information from a RAG generated output that queries a pdf file. 
	:param results : names of the raw materials mentioned in the text
	:type results : List[str]
	"""
	results: List[str]

class General(BaseModel):
	"""
	Information extractor chain uses this schema class
	to store information from a RAG retrieval.
	RAG queries a document and stores the retrieved information in this schema.  
	:param results : names mentioned in the text and retrieved by the RAG
	:type results : List[str]
	"""
	results: List[str]

class GoogleScholarSchema(BaseModel):
	"""
	The schema to structure google scholar search results 
	"""
	Titles: Sequence[str] = Field(default = [], description = "the list of titles of the manuscripts")
	Authors: Sequence[str] = Field(default = [], description = "the list of names of the authors")
	PDF_Links: Sequence[str] = Field(default = [], description = "the list of links to the pdf files")
	Citation_Counts: Sequence[str] = Field(default = [], description = "the list of number of citations")


SCHEMAS = {'suspensions': Suspensions,
			'raw_materials': RawMaterials, 
				'general': General, 
					'google_scholar': GoogleScholarSchema}
