"""Sample evenly from candidate zero at intervals of 1 / budget ratio."""


def select(count, budget_count, seed, *, ratio):
    numerator, denominator = ratio.as_integer_ratio()
    return [index * denominator // numerator for index in range(budget_count)]
