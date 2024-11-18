from langchain_core.tools import tool
import os
from getpass import getpass
from semantic_router.encoders import HuggingFaceEncoder
from pinecone import Pinecone
from pinecone import ServerlessSpec
import time
from tqdm.auto import tqdm
from datasets import load_dataset

encoder = HuggingFaceEncoder()

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") or getpass("Pinecone API key: ")

# configure client
pc = Pinecone(api_key=api_key)


spec = ServerlessSpec(
    cloud="aws", region="us-east-1"  # us-east-1
)

dims = len(encoder(["some random text"])[0])
dims

index_name = "groq-research-agent"

# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    dataset = load_dataset("jamescalam/ai-arxiv2-semantic-chunks", split="train")
    pc.create_index(
        index_name,
        dimension=dims,  # dimensionality of embed 3
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    # easier to work with dataset as pandas dataframe
    data = dataset.to_pandas().iloc[:10000]

    batch_size = 128

    for i in tqdm(range(0, len(data), batch_size)):
        i_end = min(len(data), i+batch_size)
        batch = data[i:i_end].to_dict(orient="records")
        # get batch of data
        metadata = [{
            "title": r["title"],
            "content": r["content"],
            "arxiv_id": r["arxiv_id"],
            "references": r["references"].tolist()
        } for r in batch]
        # generate unique ids for each chunk
        ids = [r["id"] for r in batch]
        # get text content to embed
        content = [r["content"] for r in batch]
        # embed text
        embeds = encoder(content)
        # add to Pinecone
        index.upsert(vectors=zip(ids, embeds, metadata))

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()


def format_rag_contexts(matches: list):
    contexts = []
    for x in matches:
        text = (
            f"Title: {x['metadata']['title']}\n"
            f"Content: {x['metadata']['content']}\n"
            f"ArXiv ID: {x['metadata']['arxiv_id']}\n"
            f"Related Papers: {x['metadata']['references']}\n"
        )
        contexts.append(text)
    context_str = "\n---\n".join(contexts)
    return context_str

@tool("rag_search_filter")
def rag_search_filter(query: str, arxiv_id: str):
    """Finds information from our ArXiv database using a natural language query
    and a specific ArXiv ID. Allows us to learn more details about a specific paper."""
    xq = encoder([query])
    xc = index.query(vector=xq, top_k=6, include_metadata=True, filter={"arxiv_id": arxiv_id})
    context_str = format_rag_contexts(xc["matches"])
    return context_str

@tool("rag_search")
def rag_search(query: str):
    """Finds specialist information on AI using a natural language query."""
    xq = encoder([query])
    xc = index.query(vector=xq, top_k=4, include_metadata=True)
    context_str = format_rag_contexts(xc["matches"])
    return context_str