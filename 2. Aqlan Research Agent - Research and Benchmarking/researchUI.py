import streamlit as st
from dotenv import load_dotenv
import json

# Import your existing research agent components
from researchAgent import (
    runnable,
    build_report
)

def format_intermediate_steps(intermediate_steps):
    """
    Format intermediate steps for better readability
    """
    formatted_steps = []
    for step in intermediate_steps:
        step_details = {
            "Tool": step.tool,
            "Input": step.tool_input,
            "Output": step.log[:500] + "..." if len(step.log) > 500 else step.log
        }
        formatted_steps.append(step_details)
    return formatted_steps

def main():
    st.title("Research Agent 🕵️‍♀️")
    st.write("An AI-powered research assistant that explores and synthesizes information")

    # Initialize session state for query history if not exists
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

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
                st.write(f"**Summary:** {query_data['summary'][:200]}...")
                
                # Unique key using index and a unique identifier
                details_key = f"details_{start_idx + len(page_queries) - idx}"
                
                # Detailed view button with unique key
                if st.button(f"View Full Report - {query_data['question'][:20]}...", key=details_key):
                    # Main area for displaying selected query details
                    st.subheader(f"Research Report: {query_data['question']}")
                    st.write(query_data['full_report'])
                    
                    # Workflow steps with detailed display
                    with st.expander("Research Workflow Steps"):
                        for i, step in enumerate(query_data['workflow_steps'], 1):
                            st.markdown(f"**Step {i}:**")
                            st.json(step)

    # Chat input
    if prompt := st.chat_input("What would you like to research?"):
        # Reset sidebar specific components for new query
        st.sidebar.empty()

        # Display user message
        st.chat_message("user").write(prompt)

        # Create a container for streaming research response
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                try:
                    # Run the research agent
                    out = runnable.invoke({
                        "input": prompt, 
                        "chat_history": st.session_state.get('chat_history', [])
                    })

                    # Extract the final output
                    final_step = out["intermediate_steps"][-1]
                    
                    # Build a comprehensive report
                    full_report = build_report(output=final_step.tool_input)
                    
                    # Generate a summary
                    summary = full_report.split("\n\n")[2]  # Using the "REPORT" section as summary
                    
                    # Display the full report
                    st.write(full_report)

                    # Format intermediate steps
                    formatted_steps = format_intermediate_steps(out["intermediate_steps"])

                    # Display workflow steps
                    with st.expander("Detailed Research Steps"):
                        for i, step in enumerate(formatted_steps, 1):
                            st.markdown(f"**Step {i}:**")
                            st.json(step)

                    # Store query in history
                    st.session_state.query_history.append({
                        'question': prompt,
                        'full_report': full_report,
                        'summary': summary,
                        'workflow_steps': formatted_steps
                    })

                    # Store in chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append({
                        "user": prompt,
                        "assistant": full_report
                    })

                except Exception as e:
                    st.error(f"An error occurred during research: {e}")
                    st.exception(e)

# Run the app
if __name__ == "__main__":
    load_dotenv()
    main()