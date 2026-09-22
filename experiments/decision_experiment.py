from decision.decision_engine import choose_action


def run_experiment():

    test_cases = [
        {
            "name": "Low mastery",
            "mastery": 0.35,
            "conceptual_error": True,
            "repeated_error": False,
            "previous_effectiveness": None
        },
        {
            "name": "Medium mastery",
            "mastery": 0.65,
            "conceptual_error": True,
            "repeated_error": False,
            "previous_effectiveness": None
        },
        {
            "name": "High mastery",
            "mastery": 0.85,
            "conceptual_error": True,
            "repeated_error": False,
            "previous_effectiveness": None
        },
        {
            "name": "Repeated error",
            "mastery": 0.35,
            "conceptual_error": True,
            "repeated_error": True,
            "previous_effectiveness": None
        },
        {
            "name": "Repeated error + ineffective intervention",
            "mastery": 0.35,
            "conceptual_error": True,
            "repeated_error": True,
            "previous_effectiveness": "INEFFECTIVE"
        }
    ]

    print("\n==========================================")
    print("       ACMF DECISION ADAPTATION TEST")
    print("==========================================")

    for case in test_cases:

        action = choose_action(
            mastery=case["mastery"],
            conceptual_error=case["conceptual_error"],
            repeated_error=case["repeated_error"],
            previous_effectiveness=case["previous_effectiveness"]
        )

        print("\n------------------------------------------")
        print("Case:", case["name"])
        print("Mastery:", case["mastery"])
        print("Conceptual error:", case["conceptual_error"])
        print("Repeated error:", case["repeated_error"])
        print(
            "Previous effectiveness:",
            case["previous_effectiveness"]
        )
        print("Selected action:", action)


if __name__ == "__main__":
    run_experiment()