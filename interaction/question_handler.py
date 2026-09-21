import ast


def check_answer(user_answer, expected_answer):
    """
    Check whether two simple Python output values
    are semantically equivalent.
    """

    try:
        user_value = ast.literal_eval(
            user_answer.strip()
        )

        expected_value = ast.literal_eval(
            expected_answer.strip()
        )

        return user_value == expected_value

    except (ValueError, SyntaxError):
        # Fallback for plain text answers
        normalized_user_answer = (
            user_answer.strip().lower()
        )

        normalized_expected_answer = (
            expected_answer.strip().lower()
        )

        return (
            normalized_user_answer
            == normalized_expected_answer
        )


def ask_question(question, expected_answer):
    """
    Display a question, collect the learner's answer,
    and determine whether it is correct.
    """

    print("\n========== FOLLOW-UP QUESTION ==========")
    print(question)

    user_answer = input("\nYour answer: ")

    correct = check_answer(
        user_answer,
        expected_answer
    )

    return user_answer, correct


if __name__ == "__main__":

    question = """
a = [10, 20]
b = a
b.append(30)

print(a)
"""

    expected_answer = "[10, 20, 30]"

    user_answer, correct = ask_question(
        question,
        expected_answer
    )

    print("\n========== RESULT ==========")
    print("Learner answer:", user_answer)
    print("Correct:", correct)