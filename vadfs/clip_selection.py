"""Fit-free CLIP selection representation; original model inputs stay untouched."""

import hashlib
import time

import numpy as np

from .samplers import OPTIMAL_METHODS, budget_count, choose
from .samplers.optimal import select


PROJECTION_SPEC = {'kind': 'clip_fixed_projection', 'seed': 234, 'dimensions': 8,
                   'normalization': 'row_l2_selection_only',
                   'generator': 'numpy.PCG64.standard_normal',
                   'orthogonalization': 'reduced_qr_positive_R_diagonal',
                   'scale': 'sqrt(512/d)', 'dtype': 'float64'}


def validate_representation(value):
    if value != PROJECTION_SPEC:
        raise ValueError('CLIP optimal representation must match the fixed method.md projection specification')


def fixed_projection():
    rng = np.random.Generator(np.random.PCG64(PROJECTION_SPEC['seed']))
    q, r = np.linalg.qr(rng.standard_normal((512, 8)), mode='reduced')
    diagonal = np.diag(r)
    if not np.isfinite(q).all() or np.any(diagonal == 0):
        raise ValueError('Invalid fixed QR projection')
    q *= np.where(diagonal < 0, -1., 1.)
    q = np.asarray(q, dtype='<f8', order='C')
    digest = hashlib.sha256(q.tobytes(order='C')).hexdigest()
    return q, {'specification': dict(PROJECTION_SPEC), 'q_sha256': digest,
               'q_shape': [512, 8], 'byte_order': 'little-endian float64 C order',
               'numpy': np.__version__, 'data_fitted': False}


def project_rows(features, count, q):
    x = np.array(features, dtype=np.float64, copy=True)
    if x.ndim != 2 or x.shape[1] != 512 or not np.isfinite(x).all():
        raise ValueError('Expected finite CLIP features [N,512]')
    if type(count) is not int or not 1 <= count <= len(x):
        raise ValueError('Invalid CLIP selection K')
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (512, 8) or not np.isfinite(q).all():
        raise ValueError('Expected finite fixed Q [512,8]')
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError('Zero or nonfinite CLIP feature norm')
    d = min(8, count)
    z = (x / norms[:, None]) @ (q[:, :d] * np.sqrt(512 / d))
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite projected CLIP features')
    return z


def select_cached_rows(features, grid, method, ratio, seed, q=None, options=None):
    """Return original feature rows, their positions, and selection diagnostics.

    No decoder, CLIP encoder, GT, PCA, SVD, normalization of the VAD input, or
    data-dependent dimension adaptation is invoked by this boundary.
    """
    if method not in ('reference', 'uniform', 'random', *OPTIMAL_METHODS):
        raise ValueError('Unsupported cached-grid method')
    if any(a >= b for a, b in zip(grid, grid[1:])):
        raise ValueError('Reference grid must be strictly increasing')
    if (features.ndim != 2 or features.shape != (len(grid), 512)
            or features.dtype.kind != 'f' or not np.isfinite(features).all()):
        raise ValueError('Invalid full CLIP feature cache')
    count = budget_count(len(grid), ratio)
    if count == len(grid):
        positions = list(range(len(grid)))
        detail = {'full_budget_bypass': True, 'reason': 'K_equals_N', 'dimension': None}
    elif method in OPTIMAL_METHODS:
        if q is None:
            raise ValueError('Fixed projection is required for budgeted CLIP optimal selection')
        tick = time.perf_counter()
        z = project_rows(features, count, q)
        transform_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        positions, detail = select(z, count, method, ratio, options, rank_adapt=False)
        detail['timing_seconds'] = {'clip_selection_projection': transform_seconds,
                                    'optimal_selection': time.perf_counter() - tick}
    else:
        selected = choose(grid, method, ratio, seed)
        positions = np.searchsorted(np.asarray(grid), selected).tolist()
        detail = {'full_budget_bypass': False}
    if (len(positions) != count or positions != sorted(set(positions))
            or any(not 0 <= index < len(grid) for index in positions)):
        raise ValueError('CLIP sampler violated budget/order/uniqueness/bounds')
    # Advanced indexing copies the original dtype/values; z never reaches VAD.
    return features[positions], positions, detail
