from typing import Union
from os import PathLike
from dotenv import load_dotenv

def load_keys(path: Union[str, PathLike]) -> bool:
	return load_dotenv(path)
