import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learner.learner_model import LearnerModel
from decision.decision_engine import choose_action
from evaluation.effectiveness import evaluate_intervention
from rag.query_builder import build_retrieval_query
from rag.retrieval import retrieve
from llm.prompt_builder import build_intervention_prompt
from llm.ollama_client import generate_intervention
from interaction.question_handler import check_answer


QUESTIONS_PATH = PROJECT_ROOT / "experiments" / "questions.json"
RESULTS_PATH = PROJECT_ROOT / "results" / "multi_learner_experiment.json"

CONCEPTS = ["variables", "conditions", "loops", "references"]

LEARNER_PROFILES = {
    "L001": {
        "profile": "developing",
        "initial_mastery": 0.35,
        "responses": ["incorrect", "incorrect", "correct"],
    },
    "L002": {
        "profile": "intermediate",
        "initial_mastery": 0.60,
        "responses": ["correct", "incorrect", "correct"],
    },
    "L003": {
        "profile": "advanced",
        "initial_mastery": 0.80,
        "responses": ["correct", "correct", "correct"],
    },
}


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)["questions"]


def group_questions_by_concept(questions):
    grouped = defaultdict(list)
    for question in questions:
        grouped[question["concept"]].append(question)
    return grouped


def build_controlled_answer(question, response_type):
    if response_type == "correct":
        return question["expected_answer"]

    return "__CONTROLLED_INCORRECT_RESPONSE__"


def run_cycle(learner, question, user_answer):
    concept = question["concept"]
    error_pattern = f"{concept}_misunderstanding"

    learner_state = learner.get_state(concept, error_pattern)
    mastery_before = learner_state["mastery"]
    previous_effectiveness = learner.get_last_intervention_effectiveness(concept)

    # Controlled experiment condition: conceptual difficulty is supplied so
    # the decision engine can be evaluated against different learner states.
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

    retrieved_chunks = retrieve(query, top_k=2)

    # Keep the exact assessment question in the generation context so the
    # intervention remains aligned with the question evaluated below.
    prompt = build_intervention_prompt(
        learner_state=learner_state,
        pedagogical_action=action,
        retrieved_chunks=retrieved_chunks,
        question=question["question"],
    )

    intervention = generate_intervention(prompt)

    correct = check_answer(user_answer, question["expected_answer"])

    effectiveness = evaluate_intervention(
        follow_up_correct=correct,
        explanation_provided=False,
        explanation_correct=False,
    )

    learner.record_evidence(
        concept=concept,
        correct=correct,
        error_pattern=None if correct else error_pattern,
    )

    mastery_after = learner.update_knowledge(concept, correct)

    learner.record_intervention(
        concept=concept,
        action=action,
        effectiveness=effectiveness,
    )

    return {
        "mastery_before": mastery_before,
        "mastery_after": mastery_after,
        "error_count_before": learner_state["error_count"],
        "repeated_error_before": learner_state["repeated_error"],
        "previous_effectiveness": previous_effectiveness,
        "action": action,
        "question_id": question["id"],
        "learner_answer": user_answer,
        "correct": correct,
        "effectiveness": effectiveness,
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
    }


def summarize_results(results):
    action_counts = Counter(result["action"] for result in results)
    effectiveness_counts = Counter(
        result["effectiveness"] for result in results
    )
    correct_count = sum(result["correct"] for result in results)

    learner_summary = {}
    for learner_id in LEARNER_PROFILES:
        learner_results = [
            result for result in results
            if result["learner_id"] == learner_id
        ]
        learner_summary[learner_id] = {
            "profile": LEARNER_PROFILES[learner_id]["profile"],
            "initial_mastery": LEARNER_PROFILES[learner_id]["initial_mastery"],
            "cycles": len(learner_results),
            "correct_responses": sum(
                result["correct"] for result in learner_results
            ),
            "final_mastery_by_concept": {},
        }

        for concept in CONCEPTS:
            concept_results = [
                result for result in learner_results
                if result["concept"] == concept
            ]
            learner_summary[learner_id]["final_mastery_by_concept"][concept] = (
                concept_results[-1]["mastery_after"]
            )

    concept_summary = {}
    for concept in CONCEPTS:
        concept_results = [
            result for result in results
            if result["concept"] == concept
        ]
        concept_summary[concept] = {
            "cycles": len(concept_results),
            "correct_responses": sum(
                result["correct"] for result in concept_results
            ),
            "action_counts": dict(
                Counter(result["action"] for result in concept_results)
            ),
            "effectiveness_counts": dict(
                Counter(
                    result["effectiveness"]
                    for result in concept_results
                )
            ),
        }

    return {
        "correct_response_rate": round(correct_count / len(results), 4),
        "action_counts": dict(action_counts),
        "effectiveness_counts": dict(effectiveness_counts),
        "learner_summary": learner_summary,
        "concept_summary": concept_summary,
    }


def run_experiment():
    questions = load_questions()
    questions_by_concept = group_questions_by_concept(questions)

    for concept in CONCEPTS:
        if len(questions_by_concept.get(concept, [])) < 3:
            raise ValueError(
                f"Concept '{concept}' must contain at least three questions."
            )

    results = []

    for learner_id, profile in LEARNER_PROFILES.items():
        learner = LearnerModel(learner_id)

        for concept in CONCEPTS:
            learner.initialize_concept(
                concept,
                mastery=profile["initial_mastery"],
            )

        for concept in CONCEPTS:
            concept_questions = questions_by_concept[concept]

            for cycle, response_type in enumerate(profile["responses"], start=1):
                question = concept_questions[cycle - 1]
                user_answer = build_controlled_answer(
                    question,
                    response_type,
                )

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
                    }
                )
                results.append(cycle_result)

    output = {
        "experiment": "ACMF Multi-Learner Multi-Concept Experiment",
        "experimental_note": (
            "Controlled learner profiles are simulated experimental conditions, "
            "not observations from real learners."
        ),
        "learner_profiles": {
            learner_id: {
                "profile": data["profile"],
                "initial_mastery": data["initial_mastery"],
                "response_pattern": data["responses"],
            }
            for learner_id, data in LEARNER_PROFILES.items()
        },
        "concepts": CONCEPTS,
        "total_learners": len(LEARNER_PROFILES),
        "total_concepts": len(CONCEPTS),
        "cycles_per_learner_per_concept": 3,
        "total_cycles": len(results),
        "summary": summarize_results(results),
        "results": results,
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("==========================================")
    print(" ACMF MULTI-LEARNER MULTI-CONCEPT TEST")
    print("==========================================")
    print("Learners:", len(LEARNER_PROFILES))
    print("Concepts:", len(CONCEPTS))
    print("Cycles per learner/concept: 3")
    print("Total cycles:", len(results))
    print("Correct response rate:", output["summary"]["correct_response_rate"])
    print("Action counts:", output["summary"]["action_counts"])
    print("Effectiveness counts:", output["summary"]["effectiveness_counts"])
    print("Results saved to:", RESULTS_PATH)


if __name__ == "__main__":
    run_experiment()
