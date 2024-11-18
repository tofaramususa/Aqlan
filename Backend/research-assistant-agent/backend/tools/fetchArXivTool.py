#Paper fetches
# Change this tool to use wikipedia or something else relevant to target audience
# https://python.langchain.com/v0.1/docs/modules/tools/
import requests
import re
from langchain_core.tools import tool

# we will test with the mixtral paper
arxiv_id = "2401.04088"

res = requests.get(
    f"https://export.arxiv.org/abs/{arxiv_id}"
)
res.text

# our regex
abstract_pattern = re.compile(
    r'\s*Abstract:\s*(.*?)\s*',
    re.DOTALL
)

# we search
re_match = abstract_pattern.search(res.text)

# and now let's see what we got
print(re_match.group(1))
     
#This is a custom tool created- we can change this
@tool("fetch_arxiv")
def fetch_arxiv(arxiv_id: str):
    """Gets the abstract from an ArXiv paper given the arxiv ID. Useful for
    finding high-level context about a specific paper."""
    # get paper page in html
    res = requests.get(
        f"https://export.arxiv.org/abs/{arxiv_id}"
    )
    # search html for abstract
    re_match = abstract_pattern.search(res.text)
    # return abstract text
    return re_match.group(1)