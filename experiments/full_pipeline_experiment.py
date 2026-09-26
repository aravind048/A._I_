import json
from pathlib import Path

from learner.learner_model import LearnerModel
from decision.decision_engine import choose_action
from evaluation.effectiveness import evaluate_intervention
from rag.query_builder import build_retrieval_query
from rag.retrieval import retrieve
from llm.prompt_builder import build_intervention_prompt
from llm.ollama_client import generate_intervention


def run_experiment():

    concept = "references"

    error_pattern = (
        "reference_aliasing_misunderstanding"
    )

    question = """
        a = [10, 20]
        b = a
        b.append(30)

        print(a)
    """

    expected_answer = "[10,20,30]"

    # Controlled learner response sequence
    responses = [
        "[10,20]",
        "[10,20]",
        "[10,20,30]"
    ]

    learner = LearnerModel("L001")

    learner.initialize_concept(
        concept,
        mastery=0.35
    )

    experiment_results = []

    print("\n==========================================")
    print("       ACMF FULL PIPELINE EXPERIMENT")
    print("==========================================")

    for cycle, user_answer in enumerate(
        responses,
        start=1
    ):

        # ==================================
        # 1. UNDERSTAND
        # ==================================

        learner_state = learner.get_state(
            concept,
            error_pattern
        )

        previous_effectiveness = (
            learner.get_last_intervention_effectiveness(concept)
        )

        mastery_before = learner_state["mastery"]

        # ==================================
        # 2. DECIDE
        # ==================================

        conceptual_error = True

        action = choose_action(
            mastery=mastery_before,
            conceptual_error=conceptual_error,
            repeated_error=learner_state[
                "repeated_error"
            ],
            previous_effectiveness=(
                previous_effectiveness
            )
        )

        # ==================================
        # 3. BUILD RAG QUERY
        # ==================================

        query = build_retrieval_query(
            concept=concept,
            error_pattern=error_pattern,
            question=question,
            pedagogical_action=action
        )

        # ==================================
        # 4. RETRIEVE KNOWLEDGE
        # ==================================

        retrieved_chunks = retrieve(
            query,
            top_k=3
        )

        # ==================================
        # 5. BUILD LLM PROMPT
        # ==================================

        prompt = build_intervention_prompt(
            learner_state=learner_state,
            pedagogical_action=action,
            retrieved_chunks=retrieved_chunks
        )

        # ==================================
        # 6. GENERATE INTERVENTION
        # ==================================

        intervention = generate_intervention(
            prompt
        )

        # ==================================
        # 7. CHECK LEARNER RESPONSE
        # ==================================

        from interaction.question_handler import (
            check_answer
        )

        correct = check_answer(
            user_answer,
            expected_answer
        )

        # ==================================
        # 8. RECORD EVIDENCE
        # ==================================

        if correct:

            learner.record_evidence(
                concept=concept,
                correct=True
            )

        else:

            learner.record_evidence(
                concept=concept,
                correct=False,
                error_pattern=error_pattern
            )

        # ==================================
        # 9. EVALUATE INTERVENTION
        # ==================================

        effectiveness = evaluate_intervention(
            follow_up_correct=correct,
            explanation_provided=False,
            explanation_correct=False
        )

        # ==================================
        # 10. UPDATE LEARNER MODEL
        # ==================================

        updated_mastery = learner.update_knowledge(
            concept,
            correct
        )

        learner.record_intervention(
            concept=concept,
            action=action,
            effectiveness=effectiveness
        )

        # ==================================
        # 11. STORE RESULT
        # ==================================

        cycle_result = {
            "cycle": cycle,
            "mastery_before": mastery_before,
            "error_count_before": learner_state[
                "error_count"
            ],
            "repeated_error_before": learner_state[
                "repeated_error"
            ],
            "previous_effectiveness": (
                previous_effectiveness
            ),
            "action": action,
            "retrieval_query": query,
            "retrieved_chunks": [
                {
                    "rank": chunk["rank"],
                    "chunk_id": chunk["chunk_id"],
                    "title": chunk["title"],
                    "content_type": chunk[
                        "content_type"
                    ]
                }
                for chunk in retrieved_chunks
            ],
            "intervention": intervention,
            "learner_answer": user_answer,
            "correct": correct,
            "effectiveness": effectiveness,
            "mastery_after": updated_mastery
        }

        experiment_results.append(
            cycle_result
        )

        # ==================================
        # OUTPUT
        # ==================================

        print("\n------------------------------------------")
        print("Cycle:", cycle)
        print("Mastery before:", mastery_before)
        print(
            "Error count:",
            learner_state["error_count"]
        )
        print(
            "Repeated error:",
            learner_state["repeated_error"]
        )
        print(
            "Previous effectiveness:",
            previous_effectiveness
        )
        print("Action:", action)
        print("Learner answer:", user_answer)
        print("Correct:", correct)
        print(
            "Effectiveness:",
            effectiveness
        )
        print(
            "Mastery after:",
            updated_mastery
        )

    # ======================================
    # 12. FINAL STATE
    # ======================================

    final_state = {
        "knowledge": learner.knowledge,
        "error_history": learner.reasoning[
            "error_history"
        ],
        "progress": learner.progress,
        "intervention_history": learner.reasoning[
            "intervention_history"
        ]
    }

    output = {
        "experiment": "ACMF Full Pipeline Experiment",
        "concept": concept,
        "error_pattern": error_pattern,
        "initial_mastery": 0.35,
        "cycles": experiment_results,
        "final_learner_state": final_state
    }

    # ======================================
    # 13. SAVE RESULTS
    # ======================================

    output_path = Path(
        "results/full_pipeline_experiment.json"
    )

    output_path.parent.mkdir(
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print("\n==========================================")
    print("Experiment completed.")
    print(
        "Results saved to:",
        output_path
    )
    print("==========================================")


if __name__ == "__main__":
    run_experiment()
