from langgraph.graph import StateGraph
from IPython.display import Image, display
from langgraph.graph import END
from nodesAndEdges import GraphState, route_question, web_search, retrieve, grade_documents, decide_to_generate, generate, grade_generation_v_documents_and_question

from dotenv import load_dotenv

load_dotenv()


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch", #match output to the node
        "vectorstore": "retrieve", #match output to the node
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

# Compile
graph = workflow.compile()
display(Image(graph.get_graph().draw_mermaid_png()))


# Initial State values
inputs = {"question": "What is the strategy for the abu dhabi executive office for 2024?", "max_retries": 3}
for event in graph.stream(inputs, stream_mode="values"):
    print(event)
    
# Current Events
# # Test on current events
# inputs = {
#     "question": "What are the models released today for llama3.2?",
#     "max_retries": 3,
# }
# for event in graph.stream(inputs, stream_mode="values"):
#     print(event)