from bkt.bkt import update_mastery


class LearnerModel:

    def __init__(self, learner_id):
        self.learner_id = learner_id

        self.knowledge = {}

        self.reasoning = {
            "error_history": {}
        }

        self.behaviour = {
            "attempt_count": 0,
            "hint_requests": 0
        }

        self.progress = {
            "questions_attempted": 0,
            "questions_correct": 0,
            "last_response_correct": None
        }

    def initialize_concept(self, concept, mastery=0.40):
        self.knowledge[concept] = mastery

    def record_response(self, correct):
        self.progress["questions_attempted"] += 1

        if correct:
            self.progress["questions_correct"] += 1

        self.progress["last_response_correct"] = correct

    def record_error(self, concept, error_pattern):

        if concept not in self.reasoning["error_history"]:
            self.reasoning["error_history"][concept] = {}

        errors = self.reasoning["error_history"][concept]

        if error_pattern not in errors:
            errors[error_pattern] = 0

        errors[error_pattern] += 1

    def get_error_count(self, concept, error_pattern):

        return self.reasoning["error_history"].get(
            concept, {}
        ).get(
            error_pattern, 0
        )

    def get_state(self, concept, error_pattern):

        mastery = self.knowledge.get(
            concept,
            0.0
        )

        error_count = self.get_error_count(
            concept,
            error_pattern
        )

        return {
            "concept": concept,
            "mastery": mastery,
            "error_pattern": error_pattern,
            "error_count": error_count,
            "repeated_error": error_count >= 2
        }

    def update_knowledge(self, concept, correct):

        current_mastery = self.knowledge.get(
            concept,
            0.40
        )

        updated_mastery = update_mastery(
            current_mastery,
            correct
        )

        self.knowledge[concept] = updated_mastery

        return updated_mastery

    def record_intervention(
        self,
        action,
        response_correct,
        effectiveness
    ):
        if "intervention_history" not in self.reasoning:
            self.reasoning["intervention_history"] = []

        self.reasoning["intervention_history"].append({
            "action": action,
            "response_correct": response_correct,
            "effectiveness": effectiveness
        })

    def get_last_intervention_effectiveness(self):
        history = self.reasoning.get(
            "intervention_history",
            []
        )

        if not history:
            return None

        return history[-1]["effectiveness"]

    def record_evidence(
        self,
        concept,
        correct,
        error_pattern=None
    ):
        # Record basic response information
        self.record_response(correct)

        # Record the error only when the response is incorrect
        if not correct and error_pattern:
            self.record_error(
                concept,
                error_pattern
            )


if __name__ == "__main__":
    learner = LearnerModel("L001")

    learner.initialize_concept(
        "references",
        0.35
    )

    print("Initial mastery:")
    print(learner.knowledge["references"])

    # Follow-up response was correct
    updated_mastery = learner.update_knowledge(
        "references",
        True
    )

    print("\nUpdated mastery:")
    print(updated_mastery)

    # Intervention was evaluated
    learner.record_intervention(
        action="WORKED_EXAMPLE",
        response_correct=True,
        effectiveness="EFFECTIVE"
    )

    print("\nLearner knowledge:")
    print(learner.knowledge)

    print("\nIntervention history:")
    print(
        learner.reasoning["intervention_history"]
    )
