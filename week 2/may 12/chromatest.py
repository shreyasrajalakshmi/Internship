from chromadb import Client  # type: ignore
from chromadb.config import Settings  # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore


# ___________ Step 1: Initialize ChromaDB client with default settings (in-memory DB)___________ 
client = Client(Settings())

#___________  2: Initialize SentenceTransformer model for embeddings___________ 
model = SentenceTransformer('all-MiniLM-L6-v2')

# ___________ Step 3: Create or get a collection___________ 
collection = client.get_or_create_collection(name="my_collection")

# ___________ Step 4: Add a sample document with embeddings___________ 
document_text = "This is a sample document about natural language processing."
embedding = model.encode(document_text).tolist()  # Convert to list for ChromaDB

collection.add(
    embeddings=[embedding],         
    documents=[document_text],
    metadatas=[{"source": "intro"}],
    ids=["doc1"]
)

#___________ Step 5: Query the database using embeddings___________ 
query_embedding = model.encode("language processing").tolist()
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

#___________  Step 6: Print the result___________ 
print("Query result:", results)
