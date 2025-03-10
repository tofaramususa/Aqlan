from retrieverDB import retriever
from langchain_core.messages import HumanMessage
from chatModels.chatGroqModel import llm
from webSearch import web_search_tool

rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


