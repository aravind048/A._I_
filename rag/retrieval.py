from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss
from chunker import load_document, chunk_markdown

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

document_path = Path("knowledge/references.md")

document = load_document(document_path)

chunks = chunk_markdown(document)


texts = [chunk["content"] for chunk in chunks]

embeddings = model.encode(
    texts,
    convert_to_numpy=True
)
print(embeddings.shape)

dimension = embeddings.shape[1]

# uses Euclidean distance, smaller values indicate more similarity
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

query = "I thought y = x creates a copy of the list."
query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)

distances, indices = index.search(
    query_embedding,
    k=3
)

for position, chunk_index in enumerate(indices[0]):
    chunk = chunks[chunk_index]

    print(f"\nRank {position + 1}")
    print("Chunk ID:", chunk["chunk_id"])
    print("Title:", chunk["title"])
    print("Type:", chunk["content_type"])
    print("Content:", chunk["content"])
