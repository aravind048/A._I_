def build_intervention_prompt(
    learner_state,
    pedagogical_action,
    retrieved_chunks
):
    knowledge = "\n\n".join(
        chunk["content"]
        for chunk in retrieved_chunks
    )

    action_instructions = {
        "EXPLANATION": """
Explain the concept clearly and directly.
The learner should understand why the behavior occurs.
You may use examples when helpful.
""",

        "HINT": """
Give only a short guiding hint.
Do NOT provide the final answer.
Do NOT fully explain the misconception.
Guide the learner toward the next reasoning step.
""",

        "WORKED_EXAMPLE": """
Provide a step-by-step worked example.
Show the intermediate reasoning and the resulting outcome.
""",

        "CHALLENGE": """
Give the learner a problem or question to solve.
Do NOT provide the solution or final answer.
Do NOT solve the challenge for the learner.
The learner should perform the reasoning themselves.
"""
    }

    selected_instruction = action_instructions.get(
        pedagogical_action,
        "Follow the selected pedagogical action."
    )

    prompt = f"""
You are an AI programming tutor.

Learner state:
Concept: {learner_state["concept"]}
Mastery: {learner_state["mastery"]}
Error pattern: {learner_state["error_pattern"]}
Repeated error: {learner_state["repeated_error"]}

Pedagogical action:
{pedagogical_action}

Action-specific instructions:
{selected_instruction}

Retrieved educational knowledge:
{knowledge}

General instructions:
- Address the learner's specific misconception.
- Use the retrieved knowledge as the basis for your response.
- Follow the selected pedagogical action exactly.
- Do not substitute another pedagogical action.
- Use simple language appropriate for a beginner.
"""

    return prompt.strip()

if __name__ == "__main__":

    learner_state = {
        "concept": "references",
        "mastery": 0.35,
        "error_pattern":
            "reference_aliasing_misunderstanding",
        "repeated_error": True
    }

    pedagogical_action = "WORKED_EXAMPLE"

    retrieved_chunks = [
        {
            "content": (
                "When we write x = [1, 2, 3] "
                "and y = x, both variables refer "
                "to the same list object."
            )
        },
        {
            "content": (
                "A common misconception is that "
                "y = x creates an independent copy "
                "of the list. It does not."
            )
        },
        {
            "content": (
                "If x = [1, 2, 3], y = x, and "
                "y.append(4), printing x produces "
                "[1, 2, 3, 4]."
            )
        }
    ]

    prompt = build_intervention_prompt(
        learner_state,
        pedagogical_action,
        retrieved_chunks
    )

    print(prompt)