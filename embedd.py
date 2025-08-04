from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import time

# Config
PINECONE_API_KEY = "pcsk_5eZ6Mn_9qEdhgRUVUGwaT2SyYntXWz7ZENoSsyfmRmuVNuo5bgsYAGuGv3qPsQUbbXumtQ"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "policy-docs"
EMBED_DIM = 384

# Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)

pc.delete_index("policy-docs")

# Create index if not exists
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    print("⏳ Waiting for index to be ready...")
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(2)
# Load index
index = pc.Index(INDEX_NAME)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

print(len(model.encode("test", convert_to_numpy=True)))  # will print 384

#convert text to embed vector
def embed_text(text: str) -> list[float]:
    return model.encode(text).tolist()

#chunck into pinecone
def store_chunk(chunk_id: str, chunk_text: str):
    vector = embed_text(chunk_text)
    index.upsert([
        {
            "id": chunk_id,
            "values": vector,
            "metadata": {"text": chunk_text}
        }
    ])
    print(f"✅ Stored chunk '{chunk_id}'")

#search for similar chunk
def search_similar_chunks(query_text: str, top_k: int = 3):
    vector = embed_text(query_text)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    if not results["matches"]:
        print("❌ No matches found.")
    else:
        print("🔍 Top matches:")
        for i, match in enumerate(results["matches"], start=1):
            print(f"\n{i}. (score={match['score']:.3f})")
            print(match["metadata"]["text"])

def main():
    while True:
        print("\n🔧 Choose an option:")
        print("1. Store a chunk")
        print("2. Convert text to embedding (debug)")
        print("3. Search with a question")
        print("4. Exit")
        choice = input("Enter choice [1-4]: ").strip()

        if choice == "1":
            cid = input("Enter chunk ID: ").strip()
            ctext = input("Enter chunk text: ").strip()
            store_chunk(cid, ctext)

        elif choice == "2":
            text = input("Enter text to embed: ").strip()
            vec = embed_text(text)
            print("🔢 Vector:", vec[:10], "...")  # print only first 10 dims

        elif choice == "3":
            query = input("Enter your question: ").strip()
            search_similar_chunks(query)

        elif choice == "4":
            print("👋 Exiting.")
            break

        else:
            print("⚠️ Invalid choice. Try again.")

if __name__ == "__main__":
    main()



