
from readet.core.chains import TitleExtractor

def test_title_extractor_with_openai_models():
	""" tests initialization of title extractor with openai models """
	model = ['openai-gpt-4o-mini', 'openai-gpt-4o']
	for m in model:
		title_extractor = TitleExtractor(chat_model = m)
		assert isinstance(title_extractor, TitleExtractor)

def test_title_extractor_call_with_different_inputs():
	""" tests the call method of title extractor with different inputs """
	title_extractor = TitleExtractor()
	titles = ["rheology of suspensions with finite inertia", 
		   		"the effect of grain size on the stiffness of pavements", 
					"Direct air capture: a solution to the climate crisis", 
						"The use of MEMS for detection of melanoma", 
							"Use of cell membranes for tissue engineering", 
								"permability of Hydrogels using pore-netwoek analysis"]
	inputs = [f"summarize {titles[0]} for me!", 
		   		f"what is {titles[1]}  all about", 
					f"who are the authors of {titles[2]}", 
						f"what is the material used in {titles[3]}"]
	for t, i in zip(titles, inputs):
		title = title_extractor(i)
		assert title.strip().lower() == t.strip().lower()

