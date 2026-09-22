def build_retrieval_query(
    concept,
    error_pattern,
    question,
    pedagogical_action
):
    error_description = {
        "reference_aliasing_misunderstanding":
            "misconception that assigning one variable to another creates a separate copy of a list"
    }

    error_text = error_description.get(
        error_pattern,
        error_pattern
    )

    action_phrase = {
        "EXPLANATION": "an explanation",
        "WORKED_EXAMPLE": "a worked example",
        "HINT": "a hint",
        "CHALLENGE": "a challenge",
        "GUIDED_PRACTICE": "guided practice"
    }.get(
        pedagogical_action,
        pedagogical_action.lower().replace("_", " ")
    )

    query = (
        f"Python {concept}. "
        f"Learner issue: {error_text}. "
        f"Question context: {question.strip()}. "
        f"Provide {action_phrase}."
    )

    return query


if __name__ == "__main__":

    query = build_retrieval_query(
        "references",
        "reference_aliasing_misunderstanding",
        "WORKED_EXAMPLE"
    )

    print("Retrieval Query:")
    print(query)
