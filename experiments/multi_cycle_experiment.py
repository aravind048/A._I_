import learner
from learner.learner_model import LearnerModel
from decision.decision_engine import choose_action


def run_experiment():

    concept = "references"
    error_pattern = "reference_aliasing_misunderstanding"

    learner = LearnerModel("L001")

    learner.initialize_concept(
        concept,
        mastery=0.35
    )

    # Controlled learner response sequence
    responses = [
        False,
        False,
        True
    ]

    print("\n==========================================")
    print("       ACMF MULTI-CYCLE ADAPTATION")
    print("==========================================")

    for cycle, correct in enumerate(responses, start=1):

        # ----------------------------------
        # 1. UNDERSTAND
        # ----------------------------------

        learner_state = learner.get_state(
            concept,
            error_pattern
        )

        previous_effectiveness = (
            learner.get_last_intervention_effectiveness(concept)
        )

        # ----------------------------------
        # 2. DECIDE
        # ----------------------------------

        conceptual_error = True

        action = choose_action(
            mastery=learner_state["mastery"],
            conceptual_error=conceptual_error,
            repeated_error=learner_state["repeated_error"],
            previous_effectiveness=previous_effectiveness
        )

        # ----------------------------------
        # 3. RECORD LEARNER RESPONSE
        # ----------------------------------

        if correct:
            learner.record_evidence(
                concept=concept,
                correct=True
            )

        else:
            learner.record_evidence(
                concept=concept,
                correct=False,
                error_pattern=error_pattern
            )

        # ----------------------------------
        # 4. UPDATE KNOWLEDGE
        # ----------------------------------

        updated_mastery = learner.update_knowledge(
            concept,
            correct
        )

        # ----------------------------------
        # 5. EVALUATE RESPONSE
        # ----------------------------------

        if correct:
            effectiveness = "POSITIVE_RESPONSE"
        else:
            effectiveness = "INEFFECTIVE"

        # ----------------------------------
        # 6. RECORD INTERVENTION
        # ----------------------------------

        learner.record_intervention(
            concept=concept,
            action=action,
            effectiveness=effectiveness
        )

        # ----------------------------------
        # OUTPUT
        # ----------------------------------

        print("\n------------------------------------------")
        print("Cycle:", cycle)
        print("Mastery before:", learner_state["mastery"])
        print("Error count:", learner_state["error_count"])
        print(
            "Repeated error:",
            learner_state["repeated_error"]
        )
        print(
            "Previous effectiveness:",
            previous_effectiveness
        )
        print("Selected action:", action)
        print("Learner response:", correct)
        print("Effectiveness:", effectiveness)
        print("Mastery after:", updated_mastery)

    # ----------------------------------
    # FINAL STATE
    # ----------------------------------

    print("\n==========================================")
    print("           FINAL LEARNER STATE")
    print("==========================================")

    print("Knowledge:")
    print(learner.knowledge)

    print("\nError history:")
    print(learner.reasoning["error_history"])

    print("\nIntervention history:")
    print(
        learner.reasoning["intervention_history"]
    )

    print("\nState history:")
    print(learner.state_history)


if __name__ == "__main__":
    run_experiment()
