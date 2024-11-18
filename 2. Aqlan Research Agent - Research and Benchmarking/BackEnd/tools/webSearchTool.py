from tavily import TavilyClient
import os
from getpass import getpass
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool("web_search")
def web_search(query: str) -> str:
    """
    Finds general knowledge information using Tavily search API.
    Can also be used to augment more 'general' knowledge to a previous specialist query.
    """
    # Get API key from environment or prompt
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        api_key = getpass("Tavily API key: ")
        
    # Initialize Tavily client
    client = TavilyClient(api_key=api_key)
    
    # Execute search
    response = client.search(query=query, search_depth="basic", max_results=5)
    
    # Handle case where no results are found
    if not response.get("results"):
        return "No results found for the given query."
        
    # Format and combine results
    formatted_results = [
        "\n".join([
            result.get("title", ""),
            result.get("content", ""),
            result.get("url", "")
        ]) for result in response["results"]
    ]
    
    return "\n---\n".join(formatted_results)

# Usage example
if __name__ == "__main__":
    results = web_search("What can you tell me about the paper arcix id 2309.14065")
    print(results)