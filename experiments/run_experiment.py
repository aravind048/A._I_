from learner.learner_model import LearnerModel
from loop.closed_loop import run_cycle


def create_learner(
    learner_id,
    concept,
    mastery
):
    learner = LearnerModel(learner_id)
    learner.initialize_concept(
        concept,
        mastery=mastery
    )
    return learner


def run_experiment():
    concept = "references"
    error_pattern = "reference_aliasing_misunderstanding"

    learners = [
        {
            "id": "L001",
            "mastery": 0.35
        },
        {
            "id": "L002",
            "mastery": 0.65
        },
        {
            "id": "L003",
            "mastery": 0.85
        }
    ]

    question = """
a = [10, 20]
b = a
b.append(30)

print(a)
"""

    expected_answer = "[10,20,30]"

    for profile in learners:

        print("\n")
        print("=" * 60)
        print("LEARNER:", profile["id"])
        print("INITIAL MASTERY:", profile["mastery"])
        print("=" * 60)

        learner = create_learner(
            learner_id=profile["id"],
            concept=concept,
            mastery=profile["mastery"]
        )

        result = run_cycle(
            learner=learner,
            concept=concept,
            error_pattern=error_pattern,
            question=question,
            expected_answer=expected_answer
        )

        print("\n========== EXPERIMENT RESULT ==========")
        print("Learner:", profile["id"])
        print("Action:", result["action"])
        print("Correct:", result["correct"])
        print("Effectiveness:", result["effectiveness"])
        print("Updated mastery:", result["updated_mastery"])


if __name__ == "__main__":
    run_experiment()