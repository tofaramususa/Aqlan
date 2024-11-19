import requests
import re
from langchain_core.tools import tool

arxiv_id = "2401.04088"

res = requests.get(
    f"https://export.arxiv.org/abs/{arxiv_id}"
)

abstract_pattern = re.compile(
    r'\s*Abstract:\s*(.*?)\s*',
    re.DOTALL
)

re_match = abstract_pattern.search(res.text)

print(re_match.group(1))

@tool("fetch_arxiv")
def fetch_arxiv(arxiv_id: str):
    """Gets the abstract from an ArXiv paper given the arxiv ID. Useful for
    finding high-level context about a specific paper."""
    res = requests.get(
        f"https://export.arxiv.org/abs/{arxiv_id}"
    )
    re_match = abstract_pattern.search(res.text)
    return re_match.group(1)