from evaluation.effectiveness import evaluate_intervention


def run_experiment():
    test_cases = [
        {
            "case": "A",
            "follow_up_correct": False,
            "explanation_provided": False,
            "explanation_correct": False,
            "expected": "INEFFECTIVE"
        },
        {
            "case": "B",
            "follow_up_correct": True,
            "explanation_provided": False,
            "explanation_correct": False,
            "expected": "POSITIVE_RESPONSE"
        },
        {
            "case": "C",
            "follow_up_correct": True,
            "explanation_provided": True,
            "explanation_correct": True,
            "expected": "EFFECTIVE"
        },
        {
            "case": "D",
            "follow_up_correct": True,
            "explanation_provided": True,
            "explanation_correct": False,
            "expected": "POSITIVE_RESPONSE"
        }
    ]

    passed = 0

    print("=" * 50)
    print("     ACMF EFFECTIVENESS EVALUATION")
    print("=" * 50)

    for test in test_cases:
        result = evaluate_intervention(
            follow_up_correct=test["follow_up_correct"],
            explanation_provided=test["explanation_provided"],
            explanation_correct=test["explanation_correct"]
        )

        success = result == test["expected"]

        print("------------------------------------------")
        print(f"Case: {test['case']}")
        print(f"Follow-up correct: {test['follow_up_correct']}")
        print(f"Explanation provided: {test['explanation_provided']}")
        print(f"Explanation correct: {test['explanation_correct']}")
        print(f"Expected: {test['expected']}")
        print(f"Actual: {result}")
        print(f"Status: {'PASS' if success else 'FAIL'}")

        if success:
            passed += 1

    print("=" * 50)
    print(f"Passed: {passed}/{len(test_cases)}")

    if passed == len(test_cases):
        print("EXPERIMENT RESULT: EFFECTIVENESS LOGIC VALIDATED")
    else:
        print("EXPERIMENT RESULT: EFFECTIVENESS LOGIC NEEDS REVIEW")


if __name__ == "__main__":
    run_experiment()