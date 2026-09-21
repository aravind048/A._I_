INITIAL_MASTERY = 0.40
LEARN_PROBABILITY = 0.20
GUESS_PROBABILITY = 0.20
SLIP_PROBABILITY = 0.10


def update_mastery(
    current_mastery,
    correct,
    learn_probability=LEARN_PROBABILITY,
    guess_probability=GUESS_PROBABILITY,
    slip_probability=SLIP_PROBABILITY
):

    if correct:

        probability_correct = (
            current_mastery * (1 - slip_probability)
            + (1 - current_mastery) * guess_probability
        )

        posterior = (
            current_mastery * (1 - slip_probability)
            / probability_correct
        )

    else:

        probability_wrong = (
            current_mastery * slip_probability
            + (1 - current_mastery) * (1 - guess_probability)
        )

        posterior = (
            current_mastery * slip_probability
            / probability_wrong
        )

    updated_mastery = (
        posterior
        + (1 - posterior) * learn_probability
    )

    return updated_mastery

if __name__ == "__main__":

    mastery = INITIAL_MASTERY

    print("Initial mastery:", mastery)

    mastery = update_mastery(
        mastery,
        True
    )

    print(
        "Mastery after correct response:",
        mastery
    )

    mastery = update_mastery(
        mastery,
        False
    )

    print(
        "Mastery after incorrect response:",
        mastery
    )