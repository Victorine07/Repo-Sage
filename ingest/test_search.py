import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

def run_search_test():
    # 1. Load environment variables
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable is missing.")

    # 2. Connect to Qdrant client
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    collection_name = "reposage"

    # 3. Load the same embedding model used during indexing
    print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # 4. Define your search queries
    queries = [
        "compute the Sharpe ratio",
        "maximum loss from peak to trough"
    ]

    # 5. Run search for each query
    for query_text in queries:
        print("\n" + "=" * 80)
        print(f"SEARCH QUERY: '{query_text}'")
        print("=" * 80)

        # Generate embedding for the query string
        query_vector = model.encode(query_text).tolist()

        # Query Qdrant for top 5 closest points
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=5,
            with_payload=True
        ).points

        # Display results
        for idx, hit in enumerate(search_result, start=1):
            payload = hit.payload
            print(f"\n--- Result {idx} (Score: {hit.score:.4f}) ---")
            print(f"Function Name : {payload.get('name')}")
            print(f"File Path     : {payload.get('file')}")
            print(f"Line Number   : {payload.get('line')}")
            print(f"Docstring     : {payload.get('docstring').strip().splitlines()[0] if payload.get('docstring') else 'None'}")

if __name__ == "__main__":
    run_search_test()