"""Choose one original frame uniformly inside each nonoverlapping window."""

import math
import random


def window_stride(ratio):
    inverse = 1 / ratio
    if not math.isfinite(inverse) or ratio != 1 / round(inverse):
        raise ValueError('Random windows require a budget reciprocal to an integer block stride')
    return round(inverse)


def select(count, budget_count, seed, *, ratio):
    """Return the window anchors in reference-candidate coordinates."""
    return list(range(0, count, window_stride(ratio)))


def select_frames(anchors, window_frames, frame_count, reference_tokens, seed):
    """Clip the last window to real frames and to the official reference coverage."""
    stop = min(frame_count, reference_tokens * 16)
    if not anchors or anchors[-1] >= stop:
        raise ValueError('Random window starts outside the decoded evaluation coverage')
    rng = random.Random(seed)
    return [rng.randrange(start, min(start + window_frames, stop)) for start in anchors]
