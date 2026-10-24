import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from ingest.chunker import parse_directory

def index_chunks(chunks: list[dict]) -> None:
    """
    Takes a list of function chunk dicts, generates vector embeddings using 
    BAAI/bge-small-en-v1.5, and uploads them into a Qdrant collection named 'reposage'.
    """
    if not chunks:
        print("No chunks provided to index.")
        return

    # 1. Load environment variables
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable is missing.")

    # 2. Connect to Qdrant
    # Pass api_key if available (e.g. Qdrant Cloud), otherwise local connection works via URL/host
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    collection_name = "reposage"

    # 3. Create or recreate the collection
    # Using recreate_collection ensures a clean slate on re-runs (available in qdrant-client)
    print(f"Setting up Qdrant collection '{collection_name}'...")
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,  # Matches BAAI/bge-small-en-v1.5 embedding dimension
            distance=Distance.COSINE
        )
    )

    # 4. Load the embedding model
    print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # 5. Build the text to embed for each chunk (name + docstring + source)
    print("Preparing text representations for embedding...")
    texts_to_embed = []
    for chunk in chunks:
        name = chunk.get("name", "")
        docstring = chunk.get("docstring", "")
        source = chunk.get("source", "")
        
        combined_text = f"{name} {docstring} {source}"
        texts_to_embed.append(combined_text)

    # 6. Embed all chunks in one batch
    print(f"Generating embeddings for {len(texts_to_embed)} chunks...")
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)

    # 7. Upload to Qdrant as PointStruct objects
    print("Uploading points to Qdrant...")
    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload=chunk
        )
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    # Upsert points in batches or all at once (batching is great if chunk counts are large)
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )

    # 8. Confirm upload count
    count_result = client.count(collection_name=collection_name, exact=True)
    print(f"Successfully indexed! Total points in '{collection_name}': {count_result.count}")


if __name__ == "__main__":
    # 1. Parse all public functions from your target library directory
    print("Parsing directory...")
    chunks = parse_directory("external/empyrical")
    print(f"Parsed {len(chunks)} chunks.")

    # 2. Embed and index them into Qdrant
    if chunks:
        index_chunks(chunks)
