"""Local RNG, sampling without replacement, chronological output."""

import random


def select(count, budget_count, seed, *, ratio=None):
    return sorted(random.Random(seed).sample(range(count), budget_count))
