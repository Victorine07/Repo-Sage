# RepoSage 🧭

> An **agentic code-&-docs intelligence and verification engine** — ask questions across a real
> codebase's *code and documentation*, and for code tasks watch it **generate → execute →
> self-correct** against the library's own test suite.

![Status](https://img.shields.io/badge/status-in%20active%20development-orange)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Serving](https://img.shields.io/badge/serving-vLLM-green)
![Agent](https://img.shields.io/badge/agent-LangGraph-purple)
![VectorDB](https://img.shields.io/badge/vectordb-Qdrant-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> 🚧 **Active development.** See the [Roadmap](#-roadmap--status) for exactly what's implemented today.

---

## 📖 Overview

**RepoSage** is a retrieval-augmented, agentic assistant built on top of a real open-source Python
library ([`empyrical`](https://github.com/stefan-jansen/empyrical-reloaded), a financial
performance/risk-metrics library — Sharpe ratio, drawdown, alpha/beta, etc.).

Most "chat with your code" tools do one of two things: they retrieve snippets and hope the model's
answer is correct, or they generate code with no idea whether it actually *runs*. RepoSage does both
better. It maintains **two knowledge bases** — one over the library's **code**, one over its
**documentation** — routes each question to the right source(s), and for any "write/fix code" request it
**executes the generated code in a sandbox and checks it against the library's real test suite**,
looping to self-correct on failure. Every answer is grounded and, for code, *verified*.

It is designed to run a self-hosted model (served with **vLLM**), orchestrate a stateful **LangGraph**
agent, optionally align the model with **DPO** using execution feedback, and report quality through a
rigorous **evaluation harness** — the full production-LLM stack, end to end.

---

## 🎯 The Problem It Solves

Two real gaps in today's code assistants:

1. **Single-source retrieval can't answer cross-cutting questions.** "How do I *use* `sharpe_ratio`
   (docs) and what does its *implementation* actually assume (code)?" requires reasoning over both docs
   and code together. RepoSage retrieves from both and synthesizes one cited answer.
2. **Generated code is rarely *verified*.** A fluent-but-wrong function is worse than useless in
   finance. RepoSage treats the library's own tests as ground truth and only trusts code that
   **executes correctly**, mirroring reference-based verification from formal methods.

---

## ✨ What It Does

- **Dual-source RAG** — retrieves over the library's **code** (AST-parsed, function-level) and its
  **docs** (README, docstrings), with hybrid dense + keyword search.
- **Agentic routing** — a LangGraph agent decides whether a question is about usage (docs),
  implementation (code), or both, and retrieves accordingly.
- **Generate → execute → self-correct** — for code tasks, it runs the model's output in a sandbox,
  checks it against `empyrical`'s reference/tests, and feeds failures back for a bounded repair loop.
- **Self-hosted, high-throughput serving** — a code LLM (Qwen2.5-Coder) served via **vLLM** with
  4-bit quantization and KV-cache management.
- **Preference alignment (DPO)** — fine-tunes the model to prefer *verified-correct* code, using
  pass/fail execution results as an automatic preference signal.
- **Rigorous, quantified evaluation** — RAG quality (faithfulness, relevancy, context precision/recall),
  code correctness (Pass@1), retrieval (recall@k), and routing accuracy — reported honestly.
- **Clean interfaces** — FastAPI backend + Streamlit demo UI with source citations.

---

## 🧭 Goals & Objectives

**Goal:** demonstrate end-to-end, production-grade LLM engineering — serving, agents, alignment, and
evaluation — on top of research-level rigor, through a system whose outputs are *verifiable*.

**Concrete objectives** (measured, not estimated — filled in as phases land):

| # | Objective | Target metric |
|---|---|---|
| 1 | Dual-source (code + docs) retrieval | recall@k on a labeled query set |
| 2 | Self-host the model with vLLM | tokens/sec throughput vs. a naïve HF pipeline |
| 3 | Agentic self-correction improves correctness | Pass@1 before vs. after the repair loop |
| 4 | DPO alignment improves correctness further | Pass@1 delta after DPO |
| 5 | Trustworthy answers | RAGAS faithfulness / answer relevancy |

---

## 🏗️ Architecture

```
              ┌───────────── Streamlit UI ─────────────┐
              └──────────────────┬─────────────────────┘
                                 │ HTTP
                        ┌────────▼────────┐
                        │  FastAPI (api)  │
                        └────────┬────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │      LangGraph agent      │
                    │  route → retrieve → (gen  │
                    │  → execute → self-correct)│
                    │  → synthesize + cite      │
                    └───┬───────────┬───────┬───┘
                        │           │       │
                 Code KB│    Docs KB│       │ Code LLM (vLLM server)
                (Qdrant)│   (Qdrant)│       │ Qwen2.5-Coder, 4-bit
                        ▲           ▲       │
                AST chunker    docs loader  └── sandboxed execution + empyrical tests
```

Full design rationale — including *why* each component was chosen, the alternatives, and the interview
questions each one covers — lives in [`../RepoSage_Build_Spec.md`](../RepoSage_Build_Spec.md).

---

## ⚙️ How It Works

1. **Ingest** — parse `empyrical`'s source into function-level chunks (via Python's `ast`) and load its
   docs; embed both and store them in two **Qdrant** collections.
2. **Retrieve** — embed the user's question, pull the most relevant code and/or doc chunks (hybrid search).
3. **Reason (agent)** — a **LangGraph** graph routes the query, gathers context, and for code tasks
   **generates code, executes it in a sandbox, and self-corrects** against `empyrical`'s tests.
4. **Serve** — generation runs on a **vLLM**-served code model (OpenAI-compatible endpoint).
5. **Answer** — the agent returns a grounded, cited answer (and, for code, a verified snippet).
6. **Evaluate** — an offline harness scores RAG quality, code Pass@1, retrieval, and routing accuracy.

---

## 📊 Evaluation & Results

Rigorous evaluation is the heart of this project, not an afterthought. All numbers below are **measured
on a held-out gold set** — never estimated — and populated as each phase completes.

| Metric | Baseline (API model) | + vLLM serving | + DPO |
|---|---|---|---|
| Pass@1 (code tasks) | _TBD_ | _TBD_ | _TBD_ |
| Faithfulness (RAGAS) | _TBD_ | _TBD_ | _TBD_ |
| Answer relevancy (RAGAS) | _TBD_ | _TBD_ | _TBD_ |
| Retrieval recall@5 | _TBD_ | _TBD_ | _TBD_ |
| Routing accuracy | _TBD_ | _TBD_ | _TBD_ |
| Throughput (tokens/sec) | _TBD_ | _TBD_ | _TBD_ |

_Populated after Phase 6 (evaluation). A note on method: code correctness is checked by **execution
against `empyrical`'s reference**, and RAG metrics use LLM-as-judge (RAGAS/DeepEval) — with its known
biases accounted for._

---

## 🧱 Tech Stack

| Layer | Tools |
|---|---|
| Serving | **vLLM** (Qwen2.5-Coder, 4-bit, KV-cache) |
| Retrieval | **Qdrant** vector DB, hybrid dense + keyword search |
| Orchestration | **LangGraph** agent (stateful, self-correcting) |
| Alignment | **DPO** via TRL + QLoRA (execution-feedback preferences) |
| Evaluation | **RAGAS / DeepEval**, Pass@1, retrieval + routing metrics |
| Interfaces | **FastAPI** + **Streamlit** |
| Tracking / Infra | **Weights & Biases / MLflow**, **Docker** |
| GPU | RunPod (serving/dev) · HPC cluster (DPO training) |

---

## 🗺️ Roadmap / Status

- [x] **Phase 0** — Project scaffold, environment, Qdrant, `empyrical` source
- [ ] **Phase 1** — Ingest + index (code + docs) → retrieval
- [ ] **Phase 2** — Baseline RAG Q&A + FastAPI + Streamlit UI
- [ ] **Phase 3** — Self-hosted vLLM serving
- [ ] **Phase 4** — LangGraph agent (routing)
- [ ] **Phase 5** — Generate → execute → self-correct verification loop
- [ ] **Phase 6** — Evaluation harness + results
- [ ] **Phase 7** — DPO alignment (research extension)
- [ ] **Phase 8** — Dockerize, polish, deploy

---

## 📂 Repository Structure

```
reposage/
  ingest/     ast_chunker.py   docs_loader.py     # parse code + docs into chunks
  index/      embed.py         qdrant_store.py    # embed + upsert to Qdrant
  retrieve/   retriever.py                        # dense → hybrid → rerank
  serve/      vllm_launch.md   client.py          # vLLM serving (GPU box)
  agent/      graph.py  nodes.py  sandbox.py      # LangGraph agent + safe execution
  align/      build_prefs.py   dpo_train.py       # DPO from execution feedback
  eval/       goldset.jsonl    run_eval.py        # the evaluation harness
  api/        main.py                             # FastAPI backend
  ui/         app.py                              # Streamlit demo
  external/   empyrical-reloaded/                 # the target library (git-ignored)
  README.md   requirements.txt   requirements-gpu.txt   docker-compose.yml
```

---

## 🚀 Getting Started

```bash
# 1) Install local deps
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

# 2) Start Qdrant (requires Docker Desktop running)
docker compose up -d            # dashboard at http://localhost:6333/dashboard

# 3) Configure
cp .env.example .env            # add your OPENAI_API_KEY
```

GPU-only dependencies (vLLM, TRL/DPO) are in `requirements-gpu.txt` and install on a CUDA machine
(RunPod / HPC cluster), not a laptop.

---

## 🧠 Key Engineering Decisions

A running log of *what was built, why that approach, and the alternatives* is kept in
[`NOTES.md`](NOTES.md) — one entry per phase. A few highlights:

- **Function-level (AST) chunking** over fixed-size windows — keeps each retrievable unit semantically
  whole.
- **Execution-based verification** over judge-only scoring — the library's tests provide *objective*
  ground truth for code correctness.
- **DPO** over full RLHF/PPO — preference alignment without a separate reward model or unstable RL,
  with labels supplied automatically by pass/fail execution.

---

## 👤 Author

**Victorine Younang, Ph.D.** — Machine Learning / GenAI (LLM fine-tuning, RAG, evaluation).
[LinkedIn](https://www.linkedin.com/in/victorine-wakam-younang-4947b712b) ·
[GitHub](https://github.com/Victorine07) ·
[Google Scholar](https://scholar.google.com/citations?user=8LaXG4AAAAAJ)

## 📄 License

MIT — see `LICENSE`.
