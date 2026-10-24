from retrieve.hybrid import hybrid_search
from retrieve.reranker import rerank_candidates

if __name__ == "__main__":
    query = "compute the Sharpe ratio"
    
    print(f"Running Hybrid Search for: '{query}'")
    candidates = hybrid_search(query, top_k=40)
    print(f"Retrieved {len(candidates)} hybrid candidates.")
    
    print("Reranking candidates...")
    top_results = rerank_candidates(query, candidates, top_k=5)
    
    print("\n" + "=" * 60)
    print(f"FINAL TOP 5 RESULTS FOR: '{query}'")
    print("=" * 60)
    for idx, res in enumerate(top_results, start=1):
        print(f"\n{idx}. Function: {res.get('name')} (Score: {res.get('rerank_score'):.4f})")
        print(f"   File: {res.get('file')} (Line {res.get('line')})")
