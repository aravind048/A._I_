from decision.decision_engine import choose_action


def run_experiment():

    test_cases = [
        {
            "case": "A",
            "mastery": 0.80,
            "conceptual_error": False,
            "repeated_error": False,
            "previous_effectiveness": None,
            "expected": "CHALLENGE"
        },
        {
            "case": "B",
            "mastery": 0.40,
            "conceptual_error": True,
            "repeated_error": False,
            "previous_effectiveness": None,
            "expected": "EXPLANATION"
        },
        {
            "case": "C",
            "mastery": 0.25,
            "conceptual_error": True,
            "repeated_error": True,
            "previous_effectiveness": None,
            "expected": "WORKED_EXAMPLE"
        },
        {
            "case": "D",
            "mastery": 0.25,
            "conceptual_error": True,
            "repeated_error": True,
            "previous_effectiveness": "INEFFECTIVE",
            "expected": "WORKED_EXAMPLE"
        }
    ]

    passed = 0

    print("\n==========================================")
    print("   ACMF DECISION SENSITIVITY EXPERIMENT")
    print("==========================================")

    for test in test_cases:

        action = choose_action(
            mastery=test["mastery"],
            conceptual_error=test["conceptual_error"],
            repeated_error=test["repeated_error"],
            previous_effectiveness=test["previous_effectiveness"]
        )

        result = action == test["expected"]

        if result:
            passed += 1

        print("\n------------------------------------------")
        print(f"Case: {test['case']}")
        print(f"Mastery: {test['mastery']}")
        print(f"Conceptual error: {test['conceptual_error']}")
        print(f"Repeated error: {test['repeated_error']}")
        print(f"Previous effectiveness: {test['previous_effectiveness']}")
        print(f"Selected action: {action}")
        print(f"Expected action: {test['expected']}")
        print(f"Result: {'PASS' if result else 'FAIL'}")

    print("\n==========================================")
    print(f"Passed: {passed}/{len(test_cases)}")

    if passed == len(test_cases):
        print("EXPERIMENT RESULT: PASS")
    else:
        print("EXPERIMENT RESULT: FAIL")


if __name__ == "__main__":
    run_experiment()