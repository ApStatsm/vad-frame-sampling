"""Sampler boundary: only candidate positions, budget and a sampler seed enter."""

import hashlib
import math

from . import random, reference, uniform


METHODS = {'reference': reference.select, 'uniform': uniform.select, 'random': random.select}


def budget_count(count, ratio):
    if type(count) is not int or count < 1:
        raise ValueError('Candidate count must be a positive integer')
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError('Budget must be a finite number in (0, 1]')
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('Budget must be a finite number in (0, 1]')
    return min(count, max(1, math.ceil(ratio * count)))


def video_seed(seed, dataset, filename):
    """Stable across processes, machines, Python hash seeds and video ordering."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Sampler seed must be an integer in [0, 2**32)')
    payload = f'{seed}\0{dataset}\0{filename}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')


def choose(candidates, method, ratio, seed):
    """Return exactly K unique original-frame indices in increasing order."""
    frames = list(candidates)
    if any(type(value) is not int or value < 0 for value in frames):
        raise ValueError('Candidates must be nonnegative integer frame indices')
    if any(left >= right for left, right in zip(frames, frames[1:])):
        raise ValueError('Candidates must be unique and increasing')
    count = budget_count(len(frames), ratio)
    if method not in METHODS:
        raise ValueError(f'Unsupported sampler: {method}; choose from {tuple(METHODS)}')
    positions = METHODS[method](len(frames), count, seed)
    if (len(positions) != count or len(set(positions)) != count
            or positions != sorted(positions)
            or any(not 0 <= value < len(frames) for value in positions)):
        raise ValueError('Sampler violated budget, uniqueness, order or bounds')
    return [frames[index] for index in positions]
