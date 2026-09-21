from learner.learner_model import LearnerModel

from decision.decision_engine import choose_action

from rag.query_builder import build_retrieval_query

from rag.retrieval import retrieve

from llm.prompt_builder import build_intervention_prompt

from llm.ollama_client import generate_intervention


if __name__ == "__main__":

    # --------------------------------
    # 1. Create learner
    # --------------------------------

    learner = LearnerModel("L001")

    learner.initialize_concept(
        "references",
        0.35
    )


    # --------------------------------
    # 2. Record learner evidence
    # --------------------------------

    learner.record_response(False)

    learner.record_error(
        "references",
        "reference_aliasing_misunderstanding"
    )

    learner.record_error(
        "references",
        "reference_aliasing_misunderstanding"
    )


    # --------------------------------
    # 3. Get current learner state
    # --------------------------------

    learner_state = learner.get_state(
        "references",
        "reference_aliasing_misunderstanding"
    )


    # --------------------------------
    # 4. Select pedagogical action
    # --------------------------------

    conceptual_error = (
        learner_state["error_pattern"] is not None
    )

    pedagogical_action = choose_action(
        mastery=learner_state["mastery"],
        conceptual_error=conceptual_error,
        repeated_error=learner_state["repeated_error"]
    )


    # --------------------------------
    # 5. Build retrieval query
    # --------------------------------

    query = build_retrieval_query(
        concept=learner_state["concept"],
        error_pattern=learner_state["error_pattern"],
        pedagogical_action=pedagogical_action
    )


    # --------------------------------
    # 6. Retrieve educational knowledge
    # --------------------------------

    retrieved_chunks = retrieve(
        query,
        top_k=3
    )


    # --------------------------------
    # 7. Build LLM prompt
    # --------------------------------

    prompt = build_intervention_prompt(
        learner_state=learner_state,
        pedagogical_action=pedagogical_action,
        retrieved_chunks=retrieved_chunks
    )


    # --------------------------------
    # 8. Generate intervention
    # --------------------------------

    intervention = generate_intervention(
        prompt
    )


    # --------------------------------
    # 9. Display result
    # --------------------------------

    print("\n========== LEARNER STATE ==========")
    print(learner_state)

    print("\n========== PEDAGOGICAL ACTION ==========")
    print(pedagogical_action)

    print("\n========== RETRIEVAL QUERY ==========")
    print(query)

    print("\n========== RETRIEVED KNOWLEDGE ==========")

    for chunk in retrieved_chunks:
        print(
            f"\nRank {chunk['rank']}"
        )

        print(
            "Chunk ID:",
            chunk["chunk_id"]
        )

        print(
            "Type:",
            chunk["content_type"]
        )

        print(
            "Content:",
            chunk["content"]
        )

    print("\n========== GENERATED INTERVENTION ==========")
    print(intervention)