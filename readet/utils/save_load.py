import pickle 
from typing import Any

def save_to_pickle(obj: Any, file_path: str) -> None:
	with open(file_path, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_from_pickle(file_path: str) -> Any:
	with open(file_path, 'rb') as f:
		return pickle.load(f)