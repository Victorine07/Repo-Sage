from sentence_transformers import CrossEncoder


# Load Cross-Encoder model | module-level to avoid re-loads at each run
print("Loading CrossEncoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_candidates(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Takes the original query and candidate chunks, scores every (query, chunk) pair 
    using a CrossEncoder model, and returns the top-k most relevant chunks.
    """
    if not candidates:
        return []

    # Construct input pairs for the cross-encoder
    pairs = []
    for chunk in candidates:
        name = chunk.get("name", "")
        docstring = chunk.get("docstring", "")
        source = chunk.get("source", "")
        combined_text = f"{name} {docstring} {source}"
        pairs.append((query, combined_text))

    # Score all pairs
    scores = model.predict(pairs)

    # Attach score to each candidate and sort descending
    scored_candidates = []
    for chunk, score in zip(candidates, scores):
        chunk_copy = chunk.copy()
        chunk_copy["rerank_score"] = float(score)
        scored_candidates.append(chunk_copy)

    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return [scored_candidate for scored_candidate in scored_candidates if scored_candidate["rerank_score"]>0 ] 
