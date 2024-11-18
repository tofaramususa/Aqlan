
### Hallucination Grader

# Hallucination grader instructions
hallucination_grader_instructions = """

You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

# Test using documents and generation from above

# Test
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

# hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
#     documents=docs_txt, generation=generation.content
# )
# result = llm_json_mode.invoke(
#     [SystemMessage(content=hallucination_grader_instructions)]
#     + [HumanMessage(content=hallucination_grader_prompt_formatted)]
# )
# json.loads(result.content)