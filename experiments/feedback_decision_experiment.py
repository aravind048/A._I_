from decision.decision_engine import choose_action


def run_experiment():

    print("\n==========================================")
    print("       ACMF FEEDBACK DECISION TEST")
    print("==========================================")

    # Same learner state in both cases
    mastery = 0.65
    conceptual_error = True
    repeated_error = True

    print("\n------------------------------------------")
    print("Case 1: No previous intervention outcome")
    print("------------------------------------------")

    action_without_feedback = choose_action(
        mastery=mastery,
        conceptual_error=conceptual_error,
        repeated_error=repeated_error,
        previous_effectiveness=None
    )

    print("Mastery:", mastery)
    print("Repeated error:", repeated_error)
    print("Previous effectiveness:", None)
    print("Selected action:", action_without_feedback)

    print("\n------------------------------------------")
    print("Case 2: Previous intervention ineffective")
    print("------------------------------------------")

    action_with_feedback = choose_action(
        mastery=mastery,
        conceptual_error=conceptual_error,
        repeated_error=repeated_error,
        previous_effectiveness="INEFFECTIVE"
    )

    print("Mastery:", mastery)
    print("Repeated error:", repeated_error)
    print("Previous effectiveness:", "INEFFECTIVE")
    print("Selected action:", action_with_feedback)

    print("\n==========================================")
    print("RESULT")
    print("==========================================")

    if action_without_feedback != action_with_feedback:
        print("FEEDBACK INFLUENCES DECISION: PASS")
    else:
        print("FEEDBACK INFLUENCES DECISION: NOT OBSERVED")


if __name__ == "__main__":
    run_experiment()