from learner.learner_model import LearnerModel
from decision.decision_engine import choose_action
from evaluation.effectiveness import evaluate_intervention
from interaction.question_handler import ask_question
from rag.query_builder import build_retrieval_query
from rag.retrieval import retrieve
from llm.prompt_builder import build_intervention_prompt
from llm.ollama_client import generate_intervention


def run_cycle(
    learner,
    concept,
    error_pattern,
    question,
    expected_answer
):
    """
    Run one complete ACMF learning cycle.

    Flow:
    UNDERSTAND
        -> DECIDE
        -> INTERVENE
        -> EVALUATE
        -> UPDATE
    """

    # ==========================================
    # 1. UNDERSTAND
    # ==========================================

    learner_state = learner.get_state(
        concept,
        error_pattern
    )

    conceptual_error = (
        learner_state["error_pattern"] is not None
    )

    previous_effectiveness = (
        learner.get_last_intervention_effectiveness()
    )

    # ==========================================
    # 2. DECIDE
    # ==========================================

    action = choose_action(
        mastery=learner_state["mastery"],
        conceptual_error=conceptual_error,
        repeated_error=learner_state["repeated_error"],
        previous_effectiveness=previous_effectiveness
    )

    # ==========================================
    # 3. INTERVENE
    # ==========================================
    #
    # In this prototype, the intervention itself
    # is represented by the selected action.
    #
    # RAG + LLM generation will be connected here
    # when we run the complete AI intervention.

    query = build_retrieval_query(
        concept=concept,
        error_pattern=error_pattern,
        pedagogical_action=action
    )

    retrieved_chunks = retrieve(
        query,
        top_k=3
    )

    prompt = build_intervention_prompt(
        learner_state=learner_state,
        pedagogical_action=action,
        retrieved_chunks=retrieved_chunks
    )

    intervention = generate_intervention(prompt)

    print("\n========== INTERVENTION ==========")
    print("Pedagogical action:", action)

    print("\nRetrieval query:")
    print(query)

    print("\nGenerated intervention:")
    print(intervention)

    # ==========================================
    # 4. LEARNER RESPONSE
    # ==========================================

    previous_correct = (
        learner.progress["last_response_correct"]
    )

    user_answer, correct = ask_question(
        question,
        expected_answer
    )

    print("\nLearner submitted:")
    print(user_answer)
    print("Correct:", correct)

    # ==========================================
    # 5. RECORD EVIDENCE
    # ==========================================

    observed_error = None

    if not correct:
        observed_error = error_pattern

    learner.record_evidence(
        concept=concept,
        correct=correct,
        error_pattern=observed_error
    )

    # ==========================================
    # 6. EVALUATE INTERVENTION
    # ==========================================

    effectiveness = evaluate_intervention(
        follow_up_correct=correct,
        explanation_provided=False,
        explanation_correct=False
    )

    # ==========================================
    # 7. UPDATE LEARNER MODEL
    # ==========================================

    updated_mastery = learner.update_knowledge(
        concept,
        correct
    )

    learner.record_intervention(
        action=action,
        response_correct=correct,
        effectiveness=effectiveness
    )

    # ==========================================
    # 8. RETURN UPDATED STATE
    # ==========================================

    updated_state = learner.get_state(
        concept,
        error_pattern
    )

    return {
        "action": action,
        "user_answer": user_answer,
        "correct": correct,
        "effectiveness": effectiveness,
        "updated_mastery": updated_mastery,
        "learner_state": updated_state
    }


if __name__ == "__main__":

    learner = LearnerModel("L001")

    learner.initialize_concept(
        "references",
        0.35
    )

    question = """
        a = [10, 20]
        b = a
        b.append(30)

        print(a)
"""

    expected_answer = "[10, 20, 30]"

    error_pattern = (
        "reference_aliasing_misunderstanding"
    )

    print("==========================================")
    print("        ACMF LEARNING CYCLE")
    print("==========================================")

    print("\nInitial mastery:")
    print(learner.knowledge["references"])

    print("\n\n==========================================")
    print("              CYCLE 1")
    print("==========================================")

    result_1 = run_cycle(
        learner=learner,
        concept="references",
        error_pattern=error_pattern,
        question=question,
        expected_answer=expected_answer
    )

    print("\n========== CYCLE 1 SUMMARY ==========")
    print("Action:", result_1["action"])
    print("Correct:", result_1["correct"])
    print("Effectiveness:", result_1["effectiveness"])
    print("Mastery:", result_1["updated_mastery"])

    print("\n\n==========================================")
    print("              CYCLE 2")
    print("==========================================")

    result_2 = run_cycle(
        learner=learner,
        concept="references",
        error_pattern=error_pattern,
        question=question,
        expected_answer=expected_answer
    )

    print("\n========== CYCLE 2 SUMMARY ==========")
    print("Action:", result_2["action"])
    print("Correct:", result_2["correct"])
    print("Effectiveness:", result_2["effectiveness"])
    print("Mastery:", result_2["updated_mastery"])

    print("\n\n==========================================")
    print("              CYCLE 3")
    print("==========================================")

    result_3 = run_cycle(
        learner=learner,
        concept="references",
        error_pattern=error_pattern,
        question=question,
        expected_answer=expected_answer
    )

    print("\n========== CYCLE 3 SUMMARY ==========")
    print("Action:", result_3["action"])
    print("Correct:", result_3["correct"])
    print("Effectiveness:", result_3["effectiveness"])
    print("Mastery:", result_3["updated_mastery"])

    print("\n\n==========================================")
    print("        FINAL LEARNER STATE")
    print("==========================================")

    print("Knowledge:")
    print(learner.knowledge)

    print("\nError history:")
    print(learner.reasoning["error_history"])

    print("\nProgress:")
    print(learner.progress)

    print("\nIntervention history:")
    print(
        learner.reasoning["intervention_history"]
    )
