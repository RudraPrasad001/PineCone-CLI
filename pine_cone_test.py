# Dummy 1536-dimensional embedding vector (usually from OpenAI)
dummy_embedding = [0.01] * 1536

# Some dummy chunk of text
dummy_chunk_text = "Refunds are processed within 30 days of purchase if eligibility conditions are met."
    
# Another dummy query vector (same length)
dummy_query = [0.02] * 1536

from pinecone import Pinecone, ServerlessSpec
import uuid

def pine_cone_test(embedding_vector: list, chunk_text: str, query_vector: list):
    pc = Pinecone(api_key="pcsk_5eZ6Mn_9qEdhgRUVUGwaT2SyYntXWz7ZENoSsyfmRmuVNuo5bgsYAGuGv3qPsQUbbXumtQ")

    index_name = "policy-docs"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=len(embedding_vector),
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",  # or "gcp"
                region="us-east-1"  # choose your preferred region
            )
        )

    # 3. Connect to the index
    index = pc.Index(index_name)

    # 4. Upsert the chunk
    chunk_id = str(uuid.uuid4())
    index.upsert([
        {
            "id": chunk_id,
            "values": embedding_vector,
            "metadata": {
                "text": chunk_text,
                "doc_id": "test-doc"
            }
        }
    ])

    print(f"Inserted chunk: {chunk_id}")

    # 5. Query the index
    query_response = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    # 6. Display results
    print("\n--- Query Results ---")
    for match in query_response["matches"]:
        print(f"\nMatch ID: {match['id']}")
        print(f"Score: {match['score']}")
        print(f"Text: {match['metadata']['text']}")

pine_cone_test(dummy_embedding,dummy_chunk_text,dummy_query)
