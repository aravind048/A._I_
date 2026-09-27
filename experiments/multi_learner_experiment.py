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
    path = Path("experiments/questions.json")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)["questions"]


def run_cycle(learner, question, user_answer):
    concept = question["concept"]
    error_pattern = f"{concept}_misunderstanding"

    learner_state = learner.get_state(concept, error_pattern)
    mastery_before = learner_state["mastery"]
    previous_effectiveness = learner.get_last_intervention_effectiveness(concept)

    # The experiment deliberately presents a conceptual difficulty so that
    # the decision engine can be tested against different learner states.
    conceptual_error = True

    action = choose_action(
        mastery=mastery_before,
        conceptual_error=conceptual_error,
        repeated_error=learner_state["repeated_error"],
        previous_effectiveness=previous_effectiveness,
    )

    query = build_retrieval_query(
        concept=concept,
        error_pattern=error_pattern,
        question=question["question"],
        pedagogical_action=action,
    )

    retrieved_chunks = retrieve(query, top_k=3)

    prompt = build_intervention_prompt(
        learner_state=learner_state,
        pedagogical_action=action,
        retrieved_chunks=retrieved_chunks,
    )

    intervention = generate_intervention(prompt)

    correct = check_answer(user_answer, question["expected_answer"])

    learner.record_evidence(
        concept=concept,
        correct=correct,
        error_pattern=None if correct else error_pattern,
    )

    effectiveness = evaluate_intervention(
        follow_up_correct=correct,
        explanation_provided=False,
        explanation_correct=False,
    )

    mastery_after = learner.update_knowledge(concept, correct)

    learner.record_intervention(
        concept=concept,
        action=action,
        effectiveness=effectiveness,
    )

    return {
        "mastery_before": mastery_before,
        "error_count_before": learner_state["error_count"],
        "repeated_error_before": learner_state["repeated_error"],
        "previous_effectiveness": previous_effectiveness,
        "action": action,
        "retrieval_query": query,
        "retrieved_chunks": [
            {
                "rank": chunk["rank"],
                "chunk_id": chunk["chunk_id"],
                "concept": chunk["concept"],
                "title": chunk["title"],
                "content_type": chunk["content_type"],
            }
            for chunk in retrieved_chunks
        ],
        "intervention_generated": bool(intervention),
        "learner_answer": user_answer,
        "correct": correct,
        "effectiveness": effectiveness,
        "mastery_after": mastery_after,
    }


def run_experiment():
    questions = load_questions()
    questions_by_concept = {}

    for question in questions:
        questions_by_concept.setdefault(question["concept"], []).append(question)

    concepts = ["variables", "conditions", "loops", "references"]

    # Controlled learner profiles. These are experimental profiles, not real
    # participants or demographic learner groups.
    learner_profiles = {
        "L001": {
            "profile": "developing",
            "initial_mastery": 0.35,
            "responses": [0, 1, 2],
        },
        "L002": {
            "profile": "intermediate",
            "initial_mastery": 0.60,
            "responses": [0, 2, 1],
        },
        "L003": {
            "profile": "advanced",
            "initial_mastery": 0.80,
            "responses": [2, 2, 2],
        },
    }

    # For each concept, the three entries correspond to the three questions
    # already defined in experiments/questions.json.
    results = []

    for learner_id, profile in learner_profiles.items():
        learner = LearnerModel(learner_id)

        for concept in concepts:
            learner.initialize_concept(
                concept,
                mastery=profile["initial_mastery"],
            )

        for concept in concepts:
            concept_questions = questions_by_concept[concept]

            for cycle, question_index in enumerate(profile["responses"], start=1):
                question = concept_questions[question_index]

                # Responses are deliberately selected from the existing
                # question set so that correctness is evaluated by the same
                # answer-checking mechanism used by the ACMF prototype.
                if profile["profile"] == "developing":
                    answers = [
                        "incorrect",
                        "incorrect",
                        question["expected_answer"],
                    ]
                elif profile["profile"] == "intermediate":
                    answers = [
                        question["expected_answer"],
                        "incorrect",
                        question["expected_answer"],
                    ]
                else:
                    answers = [question["expected_answer"]] * 3

                # Use an intentionally incorrect answer that cannot equal the
                # expected answer for the controlled negative cases.
                user_answer = answers[cycle - 1]
                if user_answer == "incorrect":
                    user_answer = "__CONTROLLED_INCORRECT_RESPONSE__"

                cycle_result = run_cycle(
                    learner,
                    question,
                    user_answer,
                )

                cycle_result.update(
                    {
                        "learner_id": learner_id,
                        "profile": profile["profile"],
                        "concept": concept,
                        "cycle": cycle,
                        "question_id": question["id"],
                    }
                )

                results.append(cycle_result)

    output = {
        "experiment": "ACMF Multi-Learner Multi-Concept Experiment",
        "learner_profiles": {
            learner_id: {
                "profile": data["profile"],
                "initial_mastery": data["initial_mastery"],
            }
            for learner_id, data in learner_profiles.items()
        },
        "concepts": concepts,
        "total_learners": len(learner_profiles),
        "total_concepts": len(concepts),
        "total_cycles": len(results),
        "results": results,
    }

    output_path = Path("results/multi_learner_experiment.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("==========================================")
    print(" ACMF MULTI-LEARNER MULTI-CONCEPT TEST")
    print("==========================================")
    print("Learners:", len(learner_profiles))
    print("Concepts:", len(concepts))
    print("Total cycles:", len(results))
    print("Results saved to:", output_path)


if __name__ == "__main__":
    run_experiment()
