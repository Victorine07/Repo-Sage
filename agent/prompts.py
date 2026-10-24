from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are an expert software engineering and quantitative finance code assistant. 
You have direct access to the source code and documentation of the 'empyrical' library (and related repositories) via retrieved context chunks.

Your responsibilities:
- Answer the user's questions accurately using only the provided code chunks and docstrings.
- Reference specific function names, file paths, and line numbers when explaining code or calculations.
- If writing Python code, enclose it in standard python markdown code blocks (```python ... ```).
- If a calculation or function is requested, explain how the retrieved function implements it based on the source code.
- If the answer cannot be found in the provided context, politely state that you cannot find the information in the indexed codebase rather than making things up.
"""

USER_PROMPT_TEMPLATE = """Here is the relevant code context retrieved from the codebase:
--------------------------------------------------
{chunks}
--------------------------------------------------

User Question: {query}

Please provide a clear, precise, and well-structured answer based on the code context above."""

qa_prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT_TEMPLATE)
])

# New Retry Prompt Template for Self-Correction
RETRY_PROMPT_TEMPLATE = """Here is the relevant code context retrieved from the codebase:
--------------------------------------------------
{chunks}
--------------------------------------------------

User Question: {query}

Your previous answer produced the following execution error:
{error}

Please fix the code, address the error, and provide a corrected solution."""

retry_prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", RETRY_PROMPT_TEMPLATE)
])