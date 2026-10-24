import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def hybrid_search(query: str, top_k: int = 40) -> list[dict]:
    """
    Runs a hybrid search combining dense Qdrant vector retrieval (top-20) 
    and BM25 keyword retrieval, merging and deduplicating them into up to 40 candidates.
    """
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable is missing.")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    collection_name = "reposage"

    # 1. Fetch all indexed chunks from Qdrant (since collection size is small ~122 items)
    scroll_result, _ = client.scroll(
        collection_name=collection_name,
        with_payload=True,
        with_vectors=False,
        limit=200
    )
    
    all_chunks = [point.payload for point in scroll_result if point.payload]
    if not all_chunks:
        return []

    # 2. Run Dense Search (Qdrant)
    query_vector = model.encode(query).tolist()
    
    dense_response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=20,
        with_payload=True
    ).points
    
    dense_chunks = [hit.payload for hit in dense_response if hit.payload]

    # 3. Run BM25 Keyword Search
    # Prepare tokenized corpus from chunk texts (name + docstring + source)
    corpus_texts = [
        f"{c.get('name', '')} {c.get('docstring', '')} {c.get('source', '')}" 
        for c in all_chunks
    ]
    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Sort corpus indices by BM25 score descending and take top 20
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    bm25_chunks = [all_chunks[i] for i in top_bm25_indices if bm25_scores[i] > 0]

    # 4. Merge and Deduplicate using Reciprocal Rank Fusion (RRF)
    # RRF formula: score = sum(1 / (60 + rank)) across lists
    rrf_scores = {}
    chunk_map = {}

    def add_to_fusion(chunk_list):
        for rank, chunk in enumerate(chunk_list, start=1):
            name = chunk.get("name")
            if not name:
                continue
            chunk_map[name] = chunk
            rrf_scores[name] = rrf_scores.get(name, 0.0) + (1.0 / (60 + rank))

    add_to_fusion(dense_chunks)
    add_to_fusion(bm25_chunks)

    # Sort merged candidates by RRF score descending
    sorted_names = sorted(rrf_scores.keys(), key=lambda n: rrf_scores[n], reverse=True)
    
    candidates = [chunk_map[name] for name in sorted_names][:top_k]
    return candidates
