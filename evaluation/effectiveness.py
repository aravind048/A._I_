def evaluate_intervention(
    follow_up_correct,
    explanation_provided=False,
    explanation_correct=False
):
    if follow_up_correct:
        if explanation_provided and explanation_correct:
            return "EFFECTIVE"

        return "POSITIVE_RESPONSE"

    return "INEFFECTIVE"


if __name__ == "__main__":

    previous_correct = False

    # Learner answers the follow-up question correctly
    follow_up_correct = True

    # Assume the learner also explained the concept correctly
    explanation_provided = True
    explanation_correct = True

    effectiveness = evaluate_intervention(
        previous_correct=previous_correct,
        follow_up_correct=follow_up_correct,
        explanation_provided=explanation_provided,
        explanation_correct=explanation_correct
    )

    print("Previous response correct:", previous_correct)
    print("Follow-up response correct:", follow_up_correct)
    print("Intervention effectiveness:", effectiveness)
