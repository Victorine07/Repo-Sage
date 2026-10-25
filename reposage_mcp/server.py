import json
from mcp.server.mcpserver import MCPServer

from retrieve.hybrid import hybrid_search
from agent.executor import run_code
from agent.graph import agent

# 1. Initialize server instance
server = MCPServer("reposage")


# 2. Tool Registration via @server.tool() decorators
@server.tool()
def search_empyrical(query: str) -> str:
    """Perform hybrid vector and BM25 search over the empyrical library codebase to retrieve relevant code chunks and documentation."""
    chunks = hybrid_search(query, top_k=5)
    return json.dumps(chunks, indent=2, default=str)


@server.tool()
def execute_snippet(code: str) -> str:
    """Safely execute a Python code snippet in a sandboxed subprocess with a 5-second timeout and return stdout/stderr results."""
    success, output = run_code(code)
    result_dict = {
        "success": success,
        "output": output
    }
    return json.dumps(result_dict, indent=2)


@server.tool()
def ask_reposage(query: str) -> str:
    """Run the full LangGraph agent workflow to answer a question, retrieve context, generate code or explanation, and execute self-correction loops if needed."""
    initial_state = {
        "query": query,
        "chunks": [],
        "answer": "",
        "execution_result": "",
        "execution_success": True,
        "retry_count": 0
    }
    result_state = agent.invoke(initial_state)
    return result_state.get("answer", "No answer generated.")


# 3. Entry Point
if __name__ == "__main__":
    server.run()