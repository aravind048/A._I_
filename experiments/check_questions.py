import json
from pathlib import Path


def load_questions():

    path = Path(
        "experiments/questions.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def check_questions():

    data = load_questions()

    questions = data["questions"]

    print("\n==========================================")
    print("       ACMF QUESTION DATASET")
    print("==========================================")

    print("Total questions:", len(questions))

    concepts = {}

    for question in questions:

        concept = question["concept"]

        if concept not in concepts:
            concepts[concept] = 0

        concepts[concept] += 1

    print("\nQuestions by concept:")

    for concept, count in concepts.items():

        print(
            f"{concept}: {count}"
        )

    print("\nQuestion IDs:")

    for question in questions:

        print(
            question["id"],
            "→",
            question["concept"]
        )


if __name__ == "__main__":
    check_questions()