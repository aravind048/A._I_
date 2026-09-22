from rag.retrieval import retrieve
from rag.query_builder import build_retrieval_query


def test_retrieval():

    test_cases = [
        {
            "concept": "variables",
            "error_pattern": "reassignment_misunderstanding",
            "question": """
x = 10
x = x + 2
print(x)
""",
            "action": "EXPLANATION"
        },
        {
            "concept": "conditions",
            "error_pattern": "condition_evaluation_misunderstanding",
            "question": """
x = 10
if x > 5:
    print("A")
else:
    print("B")
""",
            "action": "EXPLANATION"
        },
        {
            "concept": "loops",
            "error_pattern": "range_sequence_misunderstanding",
            "question": """
for i in range(3):
    print(i)
""",
            "action": "HINT"
        },
        {
            "concept": "references",
            "error_pattern":
                "reference_aliasing_misunderstanding",
            "question": """
a = [10, 20]
b = a
b.append(30)
print(a)
""",
            "action": "WORKED_EXAMPLE"
        }
    ]

    print("\n==========================================")
    print("       ACMF LEARNER-SPECIFIC RAG TEST")
    print("==========================================")

    for case in test_cases:

        query = build_retrieval_query(
            concept=case["concept"],
            error_pattern=case["error_pattern"],
            question=case["question"],
            pedagogical_action=case["action"]
        )

        results = retrieve(
            query,
            top_k=3
        )

        print("\n------------------------------------------")
        print("Concept:", case["concept"])
        print("Action:", case["action"])

        print("\nRetrieval query:")
        print(query)

        print("\nRetrieved knowledge:")

        for result in results:

            print(
                f"\nRank {result['rank']}"
            )

            print(
                "Distance:",
                result["distance"]
            )

            print(
                "Concept:",
                result["concept"]
            )

            print(
                "Title:",
                result["title"]
            )

            print(
                "Type:",
                result["content_type"]
            )


if __name__ == "__main__":
    test_retrieval()