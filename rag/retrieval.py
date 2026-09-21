from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from rag.chunker import load_document, chunk_markdown


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOCUMENT_PATH = Path("knowledge/references.md")


model = SentenceTransformer(MODEL_NAME)


def build_index():

    document = load_document(DOCUMENT_PATH)

    chunks = chunk_markdown(document)

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    # Uses Euclidean distance.
    # Smaller values indicate greater similarity.
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, chunks


def retrieve(query, top_k=3):

    index, chunks = build_index()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        k=top_k
    )

    results = []

    for position, chunk_index in enumerate(indices[0]):

        chunk = chunks[chunk_index]

        results.append({
            "rank": position + 1,
            "distance": float(distances[0][position]),
            "chunk_id": chunk["chunk_id"],
            "title": chunk["title"],
            "content_type": chunk["content_type"],
            "content": chunk["content"]
        })

    return results


if __name__ == "__main__":

    query = (
        "Python references misconception that assigning "
        "one variable to another creates a separate copy "
        "of a list. Provide a worked example."
    )

    results = retrieve(query)

    print("Retrieval Query:")
    print(query)

    for result in results:

        print(
            f"\nRank {result['rank']}"
        )

        print(
            "Distance:",
            result["distance"]
        )

        print(
            "Chunk ID:",
            result["chunk_id"]
        )

        print(
            "Title:",
            result["title"]
        )

        print(
            "Type:",
            result["content_type"]
        )

        print(
            "Content:",
            result["content"]
        )