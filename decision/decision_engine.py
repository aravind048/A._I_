from learner.learner_model import LearnerModel


def choose_action(
    mastery,
    conceptual_error,
    repeated_error,
    previous_effectiveness=None
):

    if repeated_error and previous_effectiveness == "INEFFECTIVE":
        return "WORKED_EXAMPLE"

    if mastery < 0.50 and repeated_error:
        return "WORKED_EXAMPLE"

    if mastery < 0.50 and conceptual_error:
        return "EXPLANATION"

    if 0.50 <= mastery < 0.75 and conceptual_error:
        return "HINT"

    if mastery >= 0.75:
        return "CHALLENGE"

    return "GUIDED_PRACTICE"


if __name__ == "__main__":
    learner = LearnerModel("L001")

    learner.initialize_concept(
        "references",
        0.35
    )

    learner.record_response(False)

    learner.record_error(
        "references",
        "reference_aliasing_misunderstanding"
    )

    learner.record_error(
        "references",
        "reference_aliasing_misunderstanding"
    )

    state = learner.get_state(
        "references",
        "reference_aliasing_misunderstanding"
    )

    conceptual_error = state["error_pattern"] is not None

    action = choose_action(
        mastery=state["mastery"],
        conceptual_error=conceptual_error,
        repeated_error=state["repeated_error"]
    )

    print("Learner State:")
    print(state)

    print("Selected action:", action)
