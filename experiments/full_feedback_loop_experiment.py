from learner.learner_model import LearnerModel
from decision.decision_engine import choose_action
from evaluation.effectiveness import evaluate_intervention


def run_cycle(
    learner,
    concept,
    error_pattern,
    correct
):
    # ----------------------------------------
    # 1. UNDERSTAND
    # ----------------------------------------
    state = learner.get_state(
        concept,
        error_pattern
    )

    previous_effectiveness = (
        learner.get_last_intervention_effectiveness(
            concept
        )
    )

    # Controlled assumption:
    # learner's error is conceptual
    conceptual_error = True

    # ----------------------------------------
    # 2. DECIDE
    # ----------------------------------------

    print("\nDECISION INPUTS")
    print("Mastery:", state["mastery"])
    print("Repeated error:", state["repeated_error"])
    print("Previous effectiveness:", previous_effectiveness)
    print("Conceptual error:", conceptual_error)

    action = choose_action(
        mastery=state["mastery"],
        conceptual_error=conceptual_error,
        repeated_error=state["repeated_error"],
        previous_effectiveness=previous_effectiveness
    )

    # ----------------------------------------
    # 3. INTERVENE
    # ----------------------------------------
    print("\nINTERVENTION:", action)

    # ----------------------------------------
    # 4. EVALUATE
    # ----------------------------------------
    effectiveness = evaluate_intervention(
        follow_up_correct=correct
    )

    # ----------------------------------------
    # 5. UPDATE LEARNER MODEL
    # ----------------------------------------
    learner.record_evidence(
        concept=concept,
        correct=correct,
        error_pattern=error_pattern
    )

    learner.update_knowledge(
        concept=concept,
        correct=correct
    )

    learner.record_intervention(
        concept=concept,
        action=action,
        effectiveness=effectiveness
    )

    # ----------------------------------------
    # RESULT
    # ----------------------------------------
    print("Response correct:", correct)
    print("Effectiveness:", effectiveness)
    print("Mastery after:", learner.knowledge[concept])

    return {
        "mastery_before": state["mastery"],
        "previous_effectiveness": previous_effectiveness,
        "action": action,
        "correct": correct,
        "effectiveness": effectiveness,
        "mastery_after": learner.knowledge[concept]
    }


def main():

    print("=" * 50)
    print("       ACMF FULL FEEDBACK LOOP TEST")
    print("=" * 50)

    learner = LearnerModel("L001")

    concept = "references"
    error_pattern = "reference_aliasing_misunderstanding"

    learner.initialize_concept(
        concept,
        mastery=0.65
    )

    # ----------------------------------------
    # CYCLE 1
    # ----------------------------------------

    print("\n" + "-" * 50)
    print("CYCLE 1")
    print("-" * 50)

    cycle1 = run_cycle(
        learner=learner,
        concept=concept,
        error_pattern=error_pattern,
        correct=False
    )

    # ----------------------------------------
    # CYCLE 2
    # ----------------------------------------

    print("\n" + "-" * 50)
    print("CYCLE 2")
    print("-" * 50)

    cycle2 = run_cycle(
        learner=learner,
        concept=concept,
        error_pattern=error_pattern,
        correct=False
    )

    # ----------------------------------------
    # CYCLE 3
    # ----------------------------------------

    print("\n" + "-" * 50)
    print("CYCLE 3")
    print("-" * 50)

    cycle3 = run_cycle(
        learner=learner,
        concept=concept,
        error_pattern=error_pattern,
        correct=False
    )

    # ----------------------------------------
    # FINAL RESULT
    # ----------------------------------------

    print("\n" + "=" * 50)


    print("FINAL RESULT")
    print("=" * 50)

    print("Cycle 1 action:",
        cycle1["action"])

    print("Cycle 1 effectiveness:",
        cycle1["effectiveness"])

    print("Cycle 2 action:",
        cycle2["action"])

    print("Cycle 2 effectiveness:",
        cycle2["effectiveness"])

    print("Cycle 3 repeated error:",
        learner.get_state(
            concept,
            error_pattern
        )["repeated_error"])

    print("Cycle 3 previous effectiveness:",
        cycle3["previous_effectiveness"])

    print("Cycle 3 action:",
        cycle3["action"])

    if (
        cycle1["action"] == "HINT"
        and
        cycle1["effectiveness"] == "INEFFECTIVE"
        and
        cycle2["action"] == "EXPLANATION"
        and
        cycle2["effectiveness"] == "INEFFECTIVE"
        and
        cycle3["previous_effectiveness"] == "INEFFECTIVE"
        and
        cycle3["action"] == "WORKED_EXAMPLE"
    ):
        print("\nFULL FEEDBACK LOOP: PASS")
    else:
        print("\nFULL FEEDBACK LOOP: FAIL")

    


if __name__ == "__main__":
    main()
