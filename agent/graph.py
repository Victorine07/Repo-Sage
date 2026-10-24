import os
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from retrieve.hybrid import hybrid_search
from retrieve.reranker import rerank_candidates
from agent.prompts import qa_prompt_template, retry_prompt_template
from agent.executor import extract_code, run_code

# 1. Load environment variables
load_dotenv()

# 2. Initialize LLM once at module level (using correct stable model version)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)


# 3. Expanded AgentState TypedDict
class AgentState(TypedDict):
    query: str
    chunks: List[Dict[str, Any]]
    answer: str
    execution_result: str
    execution_success: bool
    retry_count: int


# 4. Retrieve Node
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    print(f"\n[Retrieve Node] Running hybrid search and reranking for: '{query}'")
    
    candidates = hybrid_search(query, top_k=40)
    top_chunks = rerank_candidates(query, candidates, top_k=5)
    
    print(f"[Retrieve Node] Retrieved and reranked {len(top_chunks)} chunks.")
    return {"chunks": top_chunks}


# 5. Generate Node (supports initial pass & self-correction retry pass)
def generate_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    chunks = state["chunks"]
    execution_result = state.get("execution_result", "")
    retry_count = state.get("retry_count", 0)
    
    # Format chunks neatly
    formatted_chunks_list = []
    for chunk in chunks:
        name = chunk.get("name", "Unknown")
        file_path = chunk.get("file", "Unknown file")
        line = chunk.get("line", "N/A")
        docstring = chunk.get("docstring", "").strip()
        source = chunk.get("source", "").strip()
        
        chunk_str = f"Function: {name} ({file_path}, line {line})\n"
        if docstring:
            chunk_str += f"Docstring: {docstring}\n"
        chunk_str += f"Source:\n{source}\n" + "-" * 40
        formatted_chunks_list.append(chunk_str)
        
    chunks_context_text = "\n".join(formatted_chunks_list) if formatted_chunks_list else "No relevant code chunks found."

    # Choose prompt template based on whether this is a retry
    if execution_result and not state.get("execution_success", True):
        print(f"[Generate Node] Generating correction (Retry #{retry_count + 1})...")
        chain = retry_prompt_template | llm
        response = chain.invoke({
            "chunks": chunks_context_text,
            "query": query,
            "error": execution_result
        })
        retry_increment = retry_count + 1
    else:
        print("[Generate Node] Generating initial answer...")
        chain = qa_prompt_template | llm
        response = chain.invoke({
            "chunks": chunks_context_text,
            "query": query
        })
        retry_increment = retry_count

    answer = response.content if hasattr(response, "content") else str(response)
    print("[Generate Node] Answer generated successfully.")
    
    return {
        "answer": answer,
        "retry_count": retry_increment
    }


# 6. Execute Node
#def execute_node(state: AgentState) -> Dict[str, Any]:
#    answer = state.get("answer", "")
#    print("[Execute Node] Extracting and running code snippet...")
    
#    code = extract_code(answer)
#    if not code:
#        print("[Execute Node] No code block found.")
#        return {
#            "execution_result": "No python code block found in the response.",
#            "execution_success": True  # If no code requested/provided, don't force a retry loop
#        }
        
#    success, output = run_code(code)
#    print(f"[Execute Node] Execution Status -> Success: {success}")
#    print(f"Output / Error:\n{output}")
    
#    return {
#        "execution_result": output,
#        "execution_success": success
#    }

# 6. Execute Node with Intent Detection
def execute_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("query", "").strip()
    answer = state.get("answer", "")
    
    # Define action keywords indicating the user wants runnable code
    action_keywords = {"write", "show", "compute", "calculate", "create", "implement", "demonstrate", "generate"}
    
    # Get the first word of the query safely
    first_word = query.split()[0].lower() if query else ""
    
    # If it's an explanation query (doesn't start with an action keyword), skip execution
    if first_word not in action_keywords:
        print(f"[Execute Node] Explanation query detected ('{first_word}'). Skipping execution.")
        return {
            "execution_result": "Skipped: explanation query",
            "execution_success": True
        }
        
    print("[Execute Node] Action query detected. Extracting and running code snippet...")
    
    code = extract_code(answer)
    if not code:
        print("[Execute Node] No code block found.")
        return {
            "execution_result": "No python code block found in the response.",
            "execution_success": True  # Don't trigger a retry loop if no code was expected
        }
        
    success, output = run_code(code)
    print(f"[Execute Node] Execution Status -> Success: {success}")
    print(f"Output / Error:\n{output}")
    
    return {
        "execution_result": output,
        "execution_success": success
    }


# 7. Router function for conditional edge
def decide_to_retry(state: AgentState) -> str:
    success = state.get("execution_success", True)
    retry_count = state.get("retry_count", 0)
    
    if success or retry_count >= 2:
        return "end"
    else:
        return "retry"


# 8. Wire the StateGraph Workflow with Conditional Edges
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("execute", execute_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "execute")

workflow.add_conditional_edges(
    "execute",
    decide_to_retry,
    {
        "retry": "generate",
        "end": END
    }
)

agent = workflow.compile()


# --- Test Execution ---
if __name__ == "__main__":
    test_query = "write a python snippet using empyrical to calculate the sharpe ratio of a sample pandas Series of returns."
    print(f"Executing agent graph with query: '{test_query}'\n")
    
    initial_state = {
        "query": test_query,
        "chunks": [],
        "answer": "",
        "execution_result": "",
        "execution_success": True,
        "retry_count": 0
    }
    
    result = agent.invoke(initial_state)
    
    print("\n" + "=" * 80)
    print("FINAL AGENT RESPONSE & EXECUTION SUMMARY:")
    print("=" * 80)
    print(result["answer"])
    print(f"\nFinal Execution Success: {result['execution_success']}")
    print(f"Total Retries Used: {result['retry_count']}")
