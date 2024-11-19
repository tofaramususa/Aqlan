from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Union

def load_and_process_documents(urls: List[str] = None, file_paths: List[str] = None, 
                             chunk_size: int = 1000, chunk_overlap: int = 200,
                             model_name: str = "all-MiniLM-L6-v2") -> tuple:
    """
    Load and process documents from both web URLs and text files.
    
    Args:
        urls (List[str], optional): List of URLs to process
        file_paths (List[str], optional): List of file paths to process
        chunk_size (int): Size of each chunk for splitting
        chunk_overlap (int): Overlap between chunks
        model_name (str): Name of the HuggingFace embeddings model
    
    Returns:
        tuple: (vectorstore, retriever) - The vector store and retriever objects
    """
    documents = []
    
    # Process web URLs if provided
    if urls:
        web_docs = [WebBaseLoader(url).load() for url in urls]
        web_docs_list = [item for sublist in web_docs for item in sublist]
        documents.extend(web_docs_list)
    
    # Process text files if provided
    if file_paths:
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                doc = Document(page_content=text)
                documents.append(doc)
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split all documents
    doc_splits = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Create vector store
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(k=3)
    
    return vectorstore, retriever

# Example usage
urls = [
    "https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai",
    "https://www.weforum.org/stories/2024/10/generative-ai-impact-latest-research/",
]
file_paths = ["maritime.txt"]

# Process both web and text documents
vectorstore, retriever = load_and_process_documents(
    urls=urls,
    file_paths=file_paths
)