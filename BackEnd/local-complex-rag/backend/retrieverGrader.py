from retrieverDB import retriever
from chatModels.chatGroqModel import llm_json_mode
import json
from langchain_core.messages import HumanMessage, SystemMessage

### Retrieval Grader

# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

# # Test
# question = "What is Chain of thought prompting?"
# docs = retriever.invoke(question)
# doc_txt = docs[1].page_content
# doc_grader_prompt_formatted = doc_grader_prompt.format(
#     document=doc_txt, question=question
# )
# result = llm_json_mode.invoke(
#     [SystemMessage(content=doc_grader_instructions)]
#     + [HumanMessage(content=doc_grader_prompt_formatted)]
# )
# json.loads(result.content)