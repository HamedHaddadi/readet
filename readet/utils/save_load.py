import pickle 
import json 
from typing import Any

def save_to_pickle(obj: Any, file_path: str) -> None:
	with open(file_path, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_from_pickle(file_path: str) -> Any:
	with open(file_path, 'rb') as f:
		return pickle.load(f)

def save_to_json(obj: Any, file_path: str) -> None:
	with open(file_path, 'w') as f:
		json.dump(obj, f)

def load_from_json(file_path: str) -> Any:
	with open(file_path, 'r') as f:
		return json.load(f)
