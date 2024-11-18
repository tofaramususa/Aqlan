import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import json
from typing import List, Dict

# Import your custom modules
from nodesAndEdges import (
    GraphState, 
    route_question, 
    web_search, 
    retrieve, 
    grade_documents, 
    decide_to_generate, 
    generate, 
    grade_generation_v_documents_and_question
)

from dotenv import load_dotenv
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="RAG Workflow Tracer", page_icon="🔍", layout="wide")

# Custom state display function
def display_state(state, stage):
    """
    Render the current state in a more readable format
    """
    st.sidebar.markdown(f"### Current Stage: {stage}")
    
    # Create expandable sections for each state key
    for key, value in state.items():
        with st.sidebar.expander(f"{key.upper()}"):
            if isinstance(value, list):
                # If it's a list of documents, show content
                if key == 'documents':
                    for i, doc in enumerate(value, 1):
                        st.text(f"Document {i} (first 300 chars):")
                        st.code(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                else:
                    st.json(value)
            elif isinstance(value, dict):
                st.json(value)
            else:
                st.text(str(value))

# Initialize the workflow
def create_workflow():
    workflow = StateGraph(GraphState)

    # Define the nodes with state tracking
    def tracked_web_search(state):
        st.sidebar.info("Executing WEB SEARCH Node")
        result = web_search(state)
        display_state(result, "Web Search")
        return result

    def tracked_retrieve(state):
        st.sidebar.info("Executing RETRIEVE Node")
        result = retrieve(state)
        display_state(result, "Document Retrieval")
        return result

    def tracked_grade_documents(state):
        st.sidebar.info("Executing GRADE DOCUMENTS Node")
        result = grade_documents(state)
        display_state(result, "Document Grading")
        return result

    def tracked_generate(state):
        st.sidebar.info("Executing GENERATE Node")
        result = generate(state)
        display_state(result, "Answer Generation")
        return result

    # Define the nodes
    workflow.add_node("websearch", tracked_web_search)
    workflow.add_node("retrieve", tracked_retrieve)
    workflow.add_node("grade_documents", tracked_grade_documents)
    workflow.add_node("generate", tracked_generate)

    # Tracked routing function
    def tracked_route_question(state):
        st.sidebar.info("Executing ROUTE QUESTION")
        route = route_question(state)
        st.sidebar.text(f"Routing to: {route}")
        return route

    # Tracked decide to generate
    def tracked_decide_to_generate(state):
        st.sidebar.info("Executing DECIDE TO GENERATE")
        decision = decide_to_generate(state)
        st.sidebar.text(f"Decision: {decision}")
        return decision

    # Tracked generation grading
    def tracked_grade_generation(state):
        st.sidebar.info("Executing GRADE GENERATION")
        grade = grade_generation_v_documents_and_question(state)
        st.sidebar.text(f"Generation Grade: {grade}")
        return grade

    # Build graph
    workflow.set_conditional_entry_point(
        tracked_route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        tracked_decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_conditional_edges(
        "generate",
        tracked_grade_generation,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
            "max retries": END,
        },
    )

    return workflow.compile()

def main():
    st.title("RAG Workflow Tracer 🕵️‍♀️")
    st.write("This app shows the detailed workflow of Retrieval-Augmented Generation (RAG)")

    # Initialize session state for query history if not exists
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    # Initialize graph
    graph = create_workflow()

    # Sidebar for previous queries
    st.sidebar.header("Previous Queries")
    
    # Pagination for query history
    queries_per_page = 5
    total_queries = len(st.session_state.query_history)
    
    # Add pagination controls
    if total_queries > 0:
        current_page = st.sidebar.number_input(
            "Page", 
            min_value=1, 
            max_value=max(1, (total_queries - 1) // queries_per_page + 1),
            value=1
        )
        
        # Calculate start and end indices for the current page
        start_idx = (current_page - 1) * queries_per_page
        end_idx = start_idx + queries_per_page
        
        # Slice the query history for the current page
        page_queries = st.session_state.query_history[start_idx:end_idx]
        
        # Display queries for the current page
        for idx, query_data in enumerate(reversed(page_queries), 1):
            with st.sidebar.expander(f"Query: {query_data['question'][:50]}..."):
                st.write(f"**Question:** {query_data['question']}")
                st.write(f"**Response:** {query_data['response'][:200]}...")
                
                # Unique key using index and a unique identifier
                details_key = f"details_{start_idx + len(page_queries) - idx}"
                
                # Detailed view button with unique key
                if st.button(f"View Full Details - {query_data['question'][:20]}...", key=details_key):
                    # Main area for displaying selected query details
                    st.subheader(f"Query: {query_data['question']}")
                    st.write(f"**Full Response:** {query_data['response']}")
                    
                    # Workflow steps
                    with st.expander("Workflow Detailed Steps"):
                        for i, step in enumerate(query_data['workflow_steps'], 1):
                            st.write(f"Step {i}:")
                            st.json(step)

    # Chat input
    if prompt := st.chat_input("Enter your question"):
        # Reset sidebar specific components for new query
        st.sidebar.empty()

        # Display user message
        st.chat_message("user").write(prompt)

        # Create a container for streaming assistant response
        with st.chat_message("assistant"):
            # Prepare input state
            inputs = {
                "question": prompt, 
                "max_retries": 3,
                "loop_step": 0,
                "chat_history": st.session_state.get('chat_history', [])
            }

            try:
                # Stream the response with full state tracking
                full_response = ""
                workflow_steps = []

                for event in graph.stream(inputs, stream_mode="values"):
                    # Collect workflow steps
                    workflow_steps.append(event)

                    # Check if the event has an output
                    if isinstance(event, dict) and 'generation' in event:
                        # Append to full response
                        full_response += event['generation']
                        # Update the display
                        st.write(full_response)

                # Store query in history
                st.session_state.query_history.append({
                    'question': prompt,
                    'response': full_response,
                    'workflow_steps': workflow_steps
                })

                # Display workflow steps
                with st.expander("Workflow Detailed Steps"):
                    for i, step in enumerate(workflow_steps, 1):
                        st.write(f"Step {i}:")
                        st.json(step)

                # Store response in chat history
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append({
                    "user": prompt,
                    "assistant": full_response
                })

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)

# Run the app
if __name__ == "__main__":
    main()