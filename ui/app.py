import streamlit as st
import requests, os

# 2. Page config
st.set_page_config(page_title="RepoSage", layout="centered")

# 3. Header section with title and subtitle
st.title("RepoSage: AI-Powered Empyrical Code Assistant")
st.markdown("Ask questions, generate verified code snippets, or explore the source code of the `empyrical` library.")

# 4. text_area() for input + button
query = st.text_area(
    "Enter your query or prompt:",
    placeholder="e.g., Write a python snippet using empyrical to calculate the Sharpe ratio of a sample pandas Series."
)
submit_button = st.button("Ask RepoSage")

# FastAPI Backend Endpoint URL
API_URL = os.getenv("API_URL", "http://localhost:8000/query")

# 5. On button click: requests.post(...) and handle response
if submit_button:
    if not query.strip():
        st.warning("Please enter a valid query before submitting.")
    else:
        with st.spinner("RepoSage is retrieving context, generating code, and executing verification..."):
            try:
                response = requests.post(API_URL, json={"query": query}, timeout=90)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    answer = data.get("answer", "")
                    execution_success = data.get("execution_success", True)
                    execution_result = data.get("execution_result", "")
                    chunks = data.get("chunks", [])

                    # 6. Display answer with st.markdown()
                    st.subheader("Answer")
                    st.markdown(answer)

                    # 7. Show execution status with st.success(), st.info(), or st.error()
                    st.subheader("Execution Status")
                    if "Skipped" in execution_result:
                        st.info(f"Status: {execution_result}")
                    elif execution_success:
                        st.success("Execution Passed Successfully!")
                        if execution_result:
                            with st.expander("Execution Output / Stdout"):
                                st.code(execution_result, language="text")
                    else:
                        st.error("Execution Failed or Encountered Errors.")
                        with st.expander("Error Details / Traceback"):
                            st.code(execution_result, language="text")

                    # 8. st.expander() for the chunks
                    with st.expander(f"Retrieved Context Chunks ({len(chunks)})"):
                        if not chunks:
                            st.markdown("No context chunks retrieved.")
                        else:
                            for idx, chunk in enumerate(chunks, start=1):
                                st.markdown(f"**Chunk {idx}: `{chunk.get('name', 'Unknown')}`**")
                                st.text(f"File: {chunk.get('file', 'N/A')} | Line: {chunk.get('line', 'N/A')}")
                                if chunk.get("docstring"):
                                    st.markdown(f"*Docstring:* {chunk.get('docstring')}")
                                if chunk.get("source"):
                                    st.code(chunk.get("source"), language="python")
                                st.markdown("---")
                                
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the RepoSage backend API. Ensure your FastAPI server is running (`uvicorn api.main:app --reload`).")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
