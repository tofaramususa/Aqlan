from langchain_community.tools.tavily_search import TavilySearchResults
import os
import getpass
from dotenv import load_dotenv

load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("TAVILY_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "aqlan-request-assistant"

### Search

web_search_tool = TavilySearchResults(k=3)