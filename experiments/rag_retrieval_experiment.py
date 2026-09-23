from rag.query_builder import build_retrieval_query
from rag.retrieval import retrieve


def run_experiment():

    test_cases = [
        {
            "concept": "variables",
            "error_pattern": "reassignment_misunderstanding",
            "question": "x = 10; x = x + 2; print(x)",
            "pedagogical_action": "EXPLANATION"
        },
        {
            "concept": "conditions",
            "error_pattern": "condition_misunderstanding",
            "question": 'if x > 5: print("A") else print("B")',
            "pedagogical_action": "EXPLANATION"
        },
        {
            "concept": "loops",
            "error_pattern": "loop_misunderstanding",
            "question": "for i in range(3): print(i)",
            "pedagogical_action": "EXPLANATION"
        },
        {
            "concept": "references",
            "error_pattern": "reference_aliasing_misunderstanding",
            "question": "x = [1,2,3]; y = x; y.append(4); print(x)",
            "pedagogical_action": "WORKED_EXAMPLE"
        }
    ]

    print("\n==========================================")
    print("       ACMF RAG RETRIEVAL EXPERIMENT")
    print("==========================================")

    for test in test_cases:

        query = build_retrieval_query(
            concept=test["concept"],
            error_pattern=test["error_pattern"],
            question=test["question"],
            pedagogical_action=test["pedagogical_action"]
        )

        results = retrieve(query, top_k=3)

        print("\n==================================================")
        print(f"CONCEPT: {test['concept'].upper()}")
        print("==================================================")

        print("\nQuery:")
        print(query)

        print("\nRetrieved chunks:")

        for result in results:
            print("\n------------------------------------------")
            print(f"Rank: {result['rank']}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Source concept: {result['concept']}")
            print(f"Title: {result['title']}")
            print(f"Content type: {result['content_type']}")

    print("\n==========================================")
    print("      RAG EXPERIMENT COMPLETED")
    print("==========================================")


if __name__ == "__main__":
    run_experiment()