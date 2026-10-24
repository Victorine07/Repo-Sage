from pydantic import BaseModel, Field
from typing import List, Optional, Any

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user question or code generation prompt for the agent.")

class ChunkResponse(BaseModel):
    name: Optional[str] = Field(None, description="Function or class name")
    file: Optional[str] = Field(None, description="File path in the repository")
    line: Optional[int] = Field(None, description="Line number")
    docstring: Optional[str] = Field(None, description="Docstring summary")
    source: Optional[str] = Field(None, description="Source code snippet")

class QueryResponse(BaseModel):
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="The generated response/code from the LLM agent")
    execution_success: bool = Field(..., description="Whether code execution passed successfully")
    execution_result: str = Field(..., description="Stdout, error traceback, or skip status")
    chunks: List[ChunkResponse] = Field(..., description="Retrieved context chunks used by the agent")