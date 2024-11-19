from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

urls = [
    "https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai",
    "https://www.weforum.org/stories/2024/10/generative-ai-impact-latest-research/",
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(k=3)
