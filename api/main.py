import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.models import QueryRequest, QueryResponse, ChunkResponse
from agent.graph import agent

load_dotenv()

app = FastAPI(
    title="RepoSage Agent API",
    description="API-powered agent to retrieve, analyze, and execute code from the empyrical library.",
    version="1.0.0"
)

# Enable CORS for local testing and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to the RepoSage Agent API!"}

@app.get("/health")
def health_check():
    """Standard health check endpoint for Docker, Kubernetes, and load balancers."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def run_agent_query(payload: QueryRequest):
    """
    Executes the LangGraph agent workflow for a given code or explanation query.
    """
    try:
        initial_state = {
            "query": payload.query,
            "chunks": [],
            "answer": "",
            "execution_result": "",
            "execution_success": True,
            "retry_count": 0
        }
        
        # Invoke the compiled LangGraph agent
        result_state = agent.invoke(initial_state)
        
        # Map raw chunks to Pydantic models
        formatted_chunks = [
            ChunkResponse(
                name=c.get("name"),
                file=c.get("file"),
                line=c.get("line"),
                docstring=c.get("docstring"),
                source=c.get("source")
            )
            for c in result_state.get("chunks", [])
        ]
        
        return QueryResponse(
            query=result_state.get("query", payload.query),
            answer=result_state.get("answer", ""),
            execution_success=result_state.get("execution_success", True),
            execution_result=result_state.get("execution_result", ""),
            chunks=formatted_chunks
        )
        
    except Exception as e:
        #raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

    
    
