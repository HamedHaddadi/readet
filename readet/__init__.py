from .agents import structured_reports
from .agents import helper_agents

# Version information
__version__ = "0.1.3"
__author__ = "Hamed Haddadi"
__author_email__ = "haddadi.h81@gmail.com"
__license__ = "MIT"  # or whatever license you're using

# Core functionality imports
from .core import chains
from .core import knowledge_graphs
from .core import rags
from .core import retrievers
from .core import summarizers
from .core import tools

# Optional: expose specific classes/functions at package level
__all__ = [
    'structured_reports',
    'helper_agents',
    'chains',
    'knowledge_graphs',
    'rags',
    'retrievers',
    'summarizers',
    'tools'
]


def get_version():
	return __version__

def get_author():
	return __author__

