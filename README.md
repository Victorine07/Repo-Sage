# RepoSage

**An agentic RAG system that retrieves, generates, and verifies code against a live codebase.**

RepoSage is built over [`empyrical`](https://github.com/stefan-jansen/empyrical-reloaded), a Python library for financial risk metrics (Sharpe ratio, max drawdown, alpha/beta, etc.). Given a natural-language query, it retrieves relevant source functions, generates a grounded answer or code snippet, executes that code in a sandbox, and self-corrects on failure — all orchestrated through a stateful LangGraph agent.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Agent](https://img.shields.io/badge/agent-LangGraph-purple)
![VectorDB](https://img.shields.io/badge/vectordb-Qdrant-red)
![API](https://img.shields.io/badge/api-FastAPI-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Architecture

```
User query
    |
    v
FastAPI  (/query endpoint)
    |
    v
LangGraph Agent
    |-- retrieve_node   hybrid search (dense + BM25) -> cross-encoder rerank
    |-- generate_node   Gemini / Claude LLM with retrieved context
    |-- execute_node    intent detection -> sandbox execution -> pass/fail
    |-- decide          success or retry (max 2 attempts)
    |
    v
JSON response  (answer, execution_result, retrieved chunks)
```

**Retrieval pipeline:** AST-parsed function-level chunks (122 functions indexed) -> BAAI/bge-small-en-v1.5 dense embeddings stored in Qdrant Cloud -> hybrid search merging dense (top-20) and BM25 (top-20) via Reciprocal Rank Fusion -> cross-encoder reranking (ms-marco-MiniLM-L-6-v2) -> top-5 chunks passed to the LLM.

**Execution loop:** for code queries, generated snippets run in a subprocess sandbox with a 5-second timeout. On failure, the error is fed back to the LLM for a bounded self-correction loop (max 2 retries).

---

## Evaluation Results

Evaluated on a 15-query held-out test set (7 code generation, 4 explanation, 4 edge case). LLM-as-judge scoring using Gemini 2.5 Flash (temperature=0.0) across three dimensions.

| Metric | Score |
|---|---|
| Pass@1 (code generation) | **93.33%** (14/15) |
| Context Recall | 0.92 |
| Faithfulness | 0.88 |
| Answer Correctness | 1.00 |

Pass@1 measures whether generated code executes successfully on the first attempt. The one failure was a multi-step numerical edge case requiring pandas alignment not present in the retrieved context.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Vector DB | Qdrant Cloud | Production-grade, supports hybrid search natively |
| Embeddings | BAAI/bge-small-en-v1.5 | Strong retrieval quality at 384 dims, runs on CPU |
| Retrieval | Dense + BM25 + RRF | Handles both semantic and exact-match queries |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2 | Bi-encoder recall + cross-encoder precision |
| Orchestration | LangGraph StateGraph | Stateful cycles needed for the retry loop |
| LLM | Gemini 2.5 Flash / Claude | Swappable via LangChain interface |
| Chunking | Python `ast` module | Function-level semantic units, not fixed windows |
| Execution | `subprocess` + `tempfile` | Sandboxed, timeout-bounded, no eval() |
| API | FastAPI + uvicorn | Auto-docs, Pydantic validation, async-ready |
| Deployment | Docker + Docker Compose | Reproducible environment, cloud-portable |

---

## Repository Structure

```
reposage/
  ingest/       chunker.py          AST parser, function-level extraction
  retrieve/     hybrid.py           Dense + BM25 + RRF hybrid search
                reranker.py         Cross-encoder reranking
  agent/        graph.py            LangGraph StateGraph definition
                executor.py         Sandbox code execution
                prompts.py          System + retry prompt templates
  eval/         run_eval.py         LLM-as-judge evaluation harness
                test_set.py         15-query gold test set
  api/          main.py             FastAPI app, /query and /health endpoints
                models.py           Pydantic request/response models
  Dockerfile
  docker-compose.yml
  requirements.txt
```

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold, environment, Qdrant, empyrical source | Done |
| 1 | AST chunking + embedding + Qdrant indexing | Done |
| 2 | Hybrid retrieval + cross-encoder reranking | Done |
| 3 | LangGraph agent + LLM integration | Done |
| 4 | Sandbox execution + self-correction loop | Done |
| 5 | LLM-as-judge evaluation harness | Done |
| 6 | FastAPI REST endpoint + Docker | Done |
| 7 | MCP server (Model Context Protocol) | In progress |
| 8 | Streamlit UI + HuggingFace Spaces deployment | Planned |
| 9 | DPO alignment on execution feedback | Planned |

---

## Quick Start

```bash
git clone https://github.com/Victorine07/Repo-Sage.git
cd Repo-Sage
conda create -n reposage python=3.11 && conda activate reposage
pip install -r requirements.txt
cp .env.example .env        # add your QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY
uvicorn api.main:app --reload
# open http://localhost:8000/docs
```

Or with Docker:

```bash
docker compose up --build
# open http://localhost:8000/docs
```

---

## Author

**Victorine Younang, Ph.D.** — Machine Learning / GenAI
[LinkedIn](https://www.linkedin.com/in/victorine-wakam-younang-4947b712b) · [GitHub](https://github.com/Victorine07) · [Google Scholar](https://scholar.google.com/citations?user=8LaXG4AAAAAJ)
