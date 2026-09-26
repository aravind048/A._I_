import json
from pathlib import Path

from learner.learner_model import LearnerModel
from decision.decision_engine import choose_action
from evaluation.effectiveness import evaluate_intervention
from rag.query_builder import build_retrieval_query
from rag.retrieval import retrieve
from llm.prompt_builder import build_intervention_prompt
from llm.ollama_client import generate_intervention
from interaction.question_handler import check_answer


def load_questions():

    path = Path(
        "experiments/questions.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)["questions"]


def run_experiment():

    questions = load_questions()

    learner = LearnerModel("L001")

    # Initial mastery for each concept
    concepts = {
        "variables": 0.35,
        "conditions": 0.35,
        "loops": 0.35,
        "references": 0.35
    }

    for concept, mastery in concepts.items():

        learner.initialize_concept(
            concept,
            mastery=mastery
        )

    # Controlled response sequences
    #
    # Each concept has:
    # incorrect → incorrect → correct
    #
    response_sequences = {
        "variables": [
            "10",
            "10",
            "10"
        ],
        "conditions": [
            "B",
            "A",
            "Yes"
        ],
        "loops": [
            "1\n2\n3",
            "1\n2\n3",
            "Hello\nHello"
        ],
        "references": [
            "[10,20]",
            "[10,20]",
            "[1,2,3]"
        ]
    }

    experiment_results = []

    print("\n==========================================")
    print("       ACMF MULTI-CONCEPT EXPERIMENT")
    print("==========================================")

    for concept in concepts:

        print("\n")
        print("=" * 50)
        print("CONCEPT:", concept.upper())
        print("=" * 50)

        concept_questions = [
            question
            for question in questions
            if question["concept"] == concept
        ]

        responses = response_sequences[concept]

        for cycle, (
            question,
            user_answer
        ) in enumerate(
            zip(
                concept_questions,
                responses
            ),
            start=1
        ):

            error_pattern = (
                f"{concept}_misunderstanding"
            )

            # ==================================
            # 1. UNDERSTAND
            # ==================================

            learner_state = learner.get_state(
                concept,
                error_pattern
            )

            mastery_before = (
                learner_state["mastery"]
            )

            previous_effectiveness = (
                learner.get_last_intervention_effectiveness(concept)
            )

            # ==================================
            # 2. DECIDE
            # ==================================

            conceptual_error = True

            action = choose_action(
                mastery=mastery_before,
                conceptual_error=conceptual_error,
                repeated_error=(
                    learner_state[
                        "repeated_error"
                    ]
                ),
                previous_effectiveness=(
                    previous_effectiveness
                )
            )

            # ==================================
            # 3. RAG QUERY
            # ==================================

            query = build_retrieval_query(
                concept=concept,
                error_pattern=error_pattern,
                question=question["question"],
                pedagogical_action=action
            )

            # ==================================
            # 4. RETRIEVE
            # ==================================

            retrieved_chunks = retrieve(
                query,
                top_k=3
            )

            # ==================================
            # 5. PROMPT
            # ==================================

            prompt = build_intervention_prompt(
                learner_state=learner_state,
                pedagogical_action=action,
                retrieved_chunks=retrieved_chunks
            )

            # ==================================
            # 6. LLM INTERVENTION
            # ==================================

            intervention = generate_intervention(
                prompt
            )

            # ==================================
            # 7. CHECK RESPONSE
            # ==================================

            correct = check_answer(
                user_answer,
                question["expected_answer"]
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
            # 9. EVALUATE
            # ==================================

            effectiveness = (
                evaluate_intervention(
                    follow_up_correct=correct,
                    explanation_provided=False,
                    explanation_correct=False
                )
            )

            # ==================================
            # 10. UPDATE
            # ==================================

            updated_mastery = (
                learner.update_knowledge(
                    concept,
                    correct
                )
            )

            learner.record_intervention(
                concept=concept,
                action=action,
                effectiveness=effectiveness
            )

            # ==================================
            # 11. STORE RESULT
            # ==================================

            result = {
                "concept": concept,
                "cycle": cycle,
                "question_id": question["id"],
                "mastery_before": mastery_before,
                "error_count_before": (
                    learner_state[
                        "error_count"
                    ]
                ),
                "repeated_error_before": (
                    learner_state[
                        "repeated_error"
                    ]
                ),
                "previous_effectiveness": (
                    previous_effectiveness
                ),
                "action": action,
                "retrieval_query": query,
                "retrieved_chunks": [
                    {
                        "rank": chunk["rank"],
                        "chunk_id": chunk["chunk_id"],
                        "concept": chunk["concept"],
                        "title": chunk["title"],
                        "content_type": (
                            chunk["content_type"]
                        )
                    }
                    for chunk in retrieved_chunks
                ],
                "learner_answer": user_answer,
                "correct": correct,
                "effectiveness": effectiveness,
                "mastery_after": updated_mastery
            }

            experiment_results.append(result)

            print("\n------------------------------------------")
            print("Cycle:", cycle)
            print("Question:", question["id"])
            print(
                "Mastery:",
                mastery_before,
                "→",
                updated_mastery
            )
            print(
                "Repeated error:",
                learner_state[
                    "repeated_error"
                ]
            )
            print(
                "Action:",
                action
            )
            print(
                "Correct:",
                correct
            )
            print(
                "Effectiveness:",
                effectiveness
            )

    # ======================================
    # FINAL STATE
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
        "experiment":
            "ACMF Multi-Concept Experiment",
        "concepts": list(concepts.keys()),
        "total_cycles":
            len(experiment_results),
        "results":
            experiment_results,
        "final_learner_state":
            final_state
    }

    output_path = Path(
        "results/multi_concept_experiment.json"
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
    print("      EXPERIMENT COMPLETED")
    print("==========================================")
    print(
        "Total cycles:",
        len(experiment_results)
    )
    print(
        "Results saved to:",
        output_path
    )


if __name__ == "__main__":
    run_experiment()
