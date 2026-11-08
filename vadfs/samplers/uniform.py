"""Choose the midpoint of each of K equal bins of the candidate sequence."""


def select(count, budget_count, seed):
    return [((2 * index + 1) * count) // (2 * budget_count)
            for index in range(budget_count)]
