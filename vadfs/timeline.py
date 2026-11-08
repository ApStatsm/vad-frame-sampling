"""Restore missing reference-token scores, then retain upstream block evaluation.

Interpolation uses original frame coordinates on the reference candidate grid.
Each reconstructed reference token is subsequently repeated over its original
16-frame evaluation block. This preserves the old Full result at budget=1 for
every supported reconstruction method. Annotation and labels never enter here.
"""

import numpy as np


METHODS = ('nearest', 'linear', 'previous')


def candidates(token_count, block_frames=16, frame_offset=0):
    if type(token_count) is not int or token_count < 1:
        raise ValueError('Reference token count must be a positive integer')
    if type(block_frames) is not int or block_frames < 1:
        raise ValueError('Invalid block length')
    if type(frame_offset) is not int or not 0 <= frame_offset < block_frames:
        raise ValueError('Frame offset must lie inside the block')
    return [index * block_frames + frame_offset for index in range(token_count)]


def restore_tokens(scores, selected_frames, reference_frames, method='nearest'):
    values = np.asarray(scores)
    selected = np.asarray(selected_frames)
    reference = np.asarray(reference_frames)
    if method not in METHODS:
        raise ValueError(f'Unknown restoration method: {method}')
    for indices in (selected, reference):
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(np.diff(indices.astype(np.int64)) <= 0)):
            raise ValueError('Frame coordinates must be nonempty, integer and increasing')
        if np.any(indices > np.iinfo(np.int64).max):
            raise ValueError('Frame coordinate exceeds int64')
    selected = selected.astype(np.int64)
    reference = reference.astype(np.int64)
    if (values.ndim not in (1, 2) or len(values) != len(selected)
            or values.dtype.kind != 'f' or not np.isfinite(values).all()):
        raise ValueError('Expected finite floating scores with one row per selected frame')
    membership = np.searchsorted(reference, selected)
    if np.any(membership >= len(reference)) or not np.array_equal(reference[membership], selected):
        raise ValueError('Selected frames must belong to the reference candidate grid')
    if np.array_equal(selected, reference):
        return values.copy()
    if method == 'previous':
        positions = np.searchsorted(selected, reference, side='right') - 1
        return values[np.maximum(positions, 0)]
    if method == 'nearest':
        right = np.minimum(np.searchsorted(selected, reference), len(selected) - 1)
        left = np.maximum(right - 1, 0)
        # Tie goes to the earlier sample; endpoints use the closest observation.
        positions = np.where(reference - selected[left] <= selected[right] - reference, left, right)
        return values[positions]
    matrix = values[:, None] if values.ndim == 1 else values
    restored = np.column_stack([np.interp(reference, selected, matrix[:, column])
                                for column in range(matrix.shape[1])]).astype(values.dtype)
    return restored[:, 0] if values.ndim == 1 else restored


def to_frames(reference_token_scores, block_frames=16):
    if type(block_frames) is not int or block_frames < 1:
        raise ValueError('Invalid block length')
    return np.repeat(reference_token_scores, block_frames, axis=0)
