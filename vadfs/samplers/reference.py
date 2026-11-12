"""All candidates; contains no labels, images, or model calls."""


def select(count, budget_count, seed, *, ratio=None):
    if count != budget_count:
        raise ValueError('Reference requires a 100% budget')
    return list(range(count))
