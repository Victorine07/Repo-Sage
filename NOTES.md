# RepoSage — Learning Log

One entry per phase: **what I built · why this approach · alternatives considered · interview takeaway.**
This file becomes your interview cheat-sheet — keep it honest and specific.

## Phase 0 — Scaffold
- Created repo skeleton, virtualenv, dependency lists, Qdrant via Docker, and cloned `empyrical`
  source (in `external/`) for indexing.
- **Why split `requirements.txt` vs `requirements-gpu.txt`:** vLLM and bitsandbytes need CUDA/Linux —
  they run on the GPU box (RunPod/Matilda), not the Mac. Keeping them separate avoids broken installs.
- **Interview takeaway:** dev/serving separation — you prototype the RAG/agent logic locally against an
  API model, then swap in a self-hosted vLLM endpoint for production serving.
