from decision.decision_engine import choose_action


def run_experiment():

    test_cases = [
        {
            "case": "A",
            "mastery": 0.80,
            "conceptual_error": False,
            "repeated_error": False,
            "previous_effectiveness": None
        },
        {
            "case": "B",
            "mastery": 0.40,
            "conceptual_error": True,
            "repeated_error": False,
            "previous_effectiveness": None
        },
        {
            "case": "C",
            "mastery": 0.25,
            "conceptual_error": True,
            "repeated_error": True,
            "previous_effectiveness": None
        },
        {
            "case": "D",
            "mastery": 0.65,
            "conceptual_error": True,
            "repeated_error": False,
            "previous_effectiveness": None
        }
    ]

    baseline_action = "EXPLANATION"

    print("\n==========================================")
    print("     ACMF BASELINE COMPARISON")
    print("==========================================")

    adaptive_actions = []
    baseline_actions = []

    for test in test_cases:

        adaptive_action = choose_action(
            mastery=test["mastery"],
            conceptual_error=test["conceptual_error"],
            repeated_error=test["repeated_error"],
            previous_effectiveness=test["previous_effectiveness"]
        )

        adaptive_actions.append(adaptive_action)
        baseline_actions.append(baseline_action)

        print("\n------------------------------------------")
        print(f"Case: {test['case']}")
        print(f"Mastery: {test['mastery']}")
        print(f"Conceptual error: {test['conceptual_error']}")
        print(f"Repeated error: {test['repeated_error']}")
        print(f"Previous effectiveness: {test['previous_effectiveness']}")
        print(f"Baseline action: {baseline_action}")
        print(f"ACMF action: {adaptive_action}")

    adaptive_unique_actions = len(set(adaptive_actions))
    baseline_unique_actions = len(set(baseline_actions))

    adaptive_changes = sum(
        adaptive != baseline
        for adaptive, baseline
        in zip(adaptive_actions, baseline_actions)
    )

    print("\n==========================================")
    print("             SUMMARY")
    print("==========================================")

    print(f"Test cases: {len(test_cases)}")
    print(f"Baseline unique actions: {baseline_unique_actions}")
    print(f"ACMF unique actions: {adaptive_unique_actions}")
    print(f"Cases where ACMF differs from baseline: {adaptive_changes}")

    print("\nBaseline actions:")
    print(baseline_actions)

    print("ACMF actions:")
    print(adaptive_actions)

    if adaptive_unique_actions > baseline_unique_actions:
        print("\nEXPERIMENT RESULT: ADAPTIVE BEHAVIOR OBSERVED")
    else:
        print("\nEXPERIMENT RESULT: NO ADDITIONAL ACTION DIVERSITY")


if __name__ == "__main__":
    run_experiment()