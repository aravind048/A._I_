def build_intervention_prompt(
    learner_state,
    pedagogical_action,
    retrieved_chunks,
    question="",
):
    knowledge = "\n\n".join(
        chunk["content"]
        for chunk in retrieved_chunks
    )

    action_instructions = {
        "EXPLANATION": """
Explain the concept clearly and directly using the current assessment question.
Do not introduce an unrelated problem.
Keep the intervention concise and beginner-friendly.
""",

        "HINT": """
Give only a short guiding hint for the current assessment question.
Do NOT provide the final answer.
Do not introduce an unrelated problem.
""",

        "WORKED_EXAMPLE": """
Provide a concise step-by-step worked example related to the current assessment question.
Do not introduce an unrelated problem.
""",

        "CHALLENGE": """
Turn the current assessment question into a concise challenge for the learner.
Do NOT create a different question or a new scenario.
Do NOT provide the solution or final answer.
The learner must reason about the assessment question shown below.
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

Current assessment question:
{question}

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
- Keep the intervention consistent with the current assessment question.
- Use simple language appropriate for a beginner.
- Keep the response concise; target 60-100 words.
"""

    return prompt.strip()


if __name__ == "__main__":

    learner_state = {
        "concept": "references",
        "mastery": 0.35,
        "error_pattern": "reference_aliasing_misunderstanding",
        "repeated_error": True
    }

    pedagogical_action = "WORKED_EXAMPLE"

    question = """x = [1, 2, 3]
y = x
y.append(4)
print(x)"""

    retrieved_chunks = [
        {"content": "When y = x, both variables refer to the same list object."},
        {"content": "A common misconception is that y = x creates an independent copy."},
        {"content": "If y.append(4), the change is visible through x because both refer to the same object."}
    ]

    prompt = build_intervention_prompt(
        learner_state,
        pedagogical_action,
        retrieved_chunks,
        question=question,
    )

    print(prompt)
