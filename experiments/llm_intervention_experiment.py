from rag.query_builder import build_retrieval_query
from rag.retrieval import retrieve
from llm.prompt_builder import build_intervention_prompt
from llm.ollama_client import generate_intervention


def run_experiment():

    learner_state = {
        "concept": "references",
        "mastery": 0.35,
        "error_pattern": "reference_aliasing_misunderstanding",
        "repeated_error": True
    }

    question = (
        "x = [1,2,3]; "
        "y = x; "
        "y.append(4); "
        "print(x)"
    )

    actions = [
        "EXPLANATION",
        "HINT",
        "WORKED_EXAMPLE",
        "CHALLENGE"
    ]

    print("\n==========================================")
    print("     ACMF LLM INTERVENTION EXPERIMENT")
    print("==========================================")

    for action in actions:

        print("\n==================================================")
        print(f"PEDAGOGICAL ACTION: {action}")
        print("==================================================")

        query = build_retrieval_query(
            concept=learner_state["concept"],
            error_pattern=learner_state["error_pattern"],
            question=question,
            pedagogical_action=action
        )

        retrieved_chunks = retrieve(query, top_k=3)

        prompt = build_intervention_prompt(
            learner_state=learner_state,
            pedagogical_action=action,
            retrieved_chunks=retrieved_chunks
        )

        intervention = generate_intervention(prompt)

        print("\nRetrieved knowledge:")

        for chunk in retrieved_chunks:
            print(
                f"- Rank {chunk['rank']}: "
                f"{chunk['title']} "
                f"({chunk['content_type']})"
            )

        print("\nGenerated intervention:")
        print("------------------------------------------")
        print(intervention)
        print("------------------------------------------")


if __name__ == "__main__":
    run_experiment()