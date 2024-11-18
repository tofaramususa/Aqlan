### Router
# from chatModels.chatGroqModel import llm_json_mode
# import json
# from langchain_core.messages import HumanMessage, SystemMessage


# Prompt
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

# # Test router
# test_web_search = llm_json_mode.invoke(
#     [SystemMessage(content=router_instructions)]
#     + [
#         HumanMessage(
#             content="Who is favored to win the NFC Championship game in the 2024 season?"
#         )
#     ]
# )
# test_web_search_2 = llm_json_mode.invoke(
#     [SystemMessage(content=router_instructions)]
#     + [HumanMessage(content="What are the models released today for llama3.2?")]
# )
# test_vector_store = llm_json_mode.invoke(
#     [SystemMessage(content=router_instructions)]
#     + [HumanMessage(content="What are the types of agent memory?")]
# )
# json.dumps(test_web_search),
# json.dumps(test_web_search_2),
# json.dumps(test_vector_store),

# print(
#     json.loads(json.dumps(test_web_search)),
#     json.loads(json.dumps(test_web_search_2)),
#     json.loads(json.dumps(test_vector_store)),
# )
# print(
#     test_web_search.content,
#     test_web_search_2.content,
#     test_vector_store.content,
# )