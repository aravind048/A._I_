import faiss

from sentence_transformers import SentenceTransformer

from rag.chunker import load_all_chunks


MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

model = SentenceTransformer(
    MODEL_NAME
)


def build_index():

    chunks = load_all_chunks()

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index, chunks


def retrieve(
    query,
    top_k=3
):

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

    for position, chunk_index in enumerate(
        indices[0]
    ):

        chunk = chunks[chunk_index]

        results.append({
            "rank": position + 1,
            "distance": float(
                distances[0][position]
            ),
            "chunk_id": chunk["chunk_id"],
            "concept": chunk["concept"],
            "source": chunk["source"],
            "title": chunk["title"],
            "content_type": chunk[
                "content_type"
            ],
            "content": chunk["content"]
        })

    return results