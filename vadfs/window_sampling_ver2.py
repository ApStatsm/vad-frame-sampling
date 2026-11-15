"""Raw-window constrained design selection. No model, labels or GT inputs."""

import hashlib
import random
import time

import cv2
import numpy as np

from .samplers import budget_count
from .samplers.optimal import objectives
from .samplers.random_window import window_stride

METHODS_VER2 = ('uniform', 'random_window', 'd_optimal', 'a_optimal', 'e_optimal')
SPEC_VER2 = {'version': 2, 'thumbnail': [32, 32], 'resize': 'INTER_AREA',
             'features': 'RGB means,2*RGB population std,gray edge,adjacent gray difference',
             'projection_seed': 234, 'dimensions': 8, 'ridge': 1.0, 'max_sweeps': 5,
             'objective_rtol': 1e-12, 'e_rtol': 1e-10, 'anchor': 'window_start'}


def windows_ver2(reference_tokens, frame_count, ratio):
    if type(reference_tokens) is not int or reference_tokens < 1 or type(frame_count) is not int or frame_count < 1:
        raise ValueError('Positive integer reference/frame counts required')
    if ratio not in (1., .5, .25, .125):
        raise ValueError('Unsupported ver2 budget')
    q = window_stride(ratio)
    stop = min(frame_count, 16 * reference_tokens)
    anchors = list(range(0, 16 * reference_tokens, 16 * q))
    bounds = [(a, min(a + 16 * q, stop)) for a in anchors]
    if len(bounds) != budget_count(reference_tokens, ratio) or any(a >= b for a, b in bounds):
        raise ValueError('Empty window or mismatched K')
    return bounds


def feature_row_ver2(bgr, previous=None):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.
    gray = image @ np.array([.299, .587, .114])
    edge = (np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2
    motion = 0. if previous is None else np.abs(gray - previous).mean()
    row = np.r_[image.mean(axis=(0, 1)), 2 * image.std(axis=(0, 1)), edge, motion]
    return row, gray


def collect_ver2(path, reference_tokens):
    """First pass: OpenCV thumbnails for all in-coverage original frames."""
    begin = time.perf_counter()
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    timings = {'first_pass_decode': 0., 'cheap_features': 0.}
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open {path}')
        header = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if (not np.isfinite(header) or header < 1 or abs(header - round(header)) > 1e-3
                or not np.isfinite(fps) or fps <= 0):
            raise ValueError('Invalid first-pass metadata')
        timings['first_pass_decode'] += time.perf_counter() - begin
        rows, previous, count = [], None, 0
        while True:
            tick = time.perf_counter()
            ok, bgr = cap.read()
            timings['first_pass_decode'] += time.perf_counter() - tick
            if not ok:
                break
            if count < 16 * reference_tokens:
                tick = time.perf_counter()
                row, previous = feature_row_ver2(bgr, previous)
                rows.append(row)
                timings['cheap_features'] += time.perf_counter() - tick
            count += 1
        if count != round(header) or reference_tokens not in (count // 16, (count + 15) // 16):
            raise ValueError('First-pass decode/header/reference mismatch')
    finally:
        tick = time.perf_counter()
        cap.release()
        timings['first_pass_decode'] += time.perf_counter() - tick
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (min(count, 16 * reference_tokens), 8) or not np.isfinite(values).all():
        raise ValueError('Invalid thumbnail matrix')
    return values, {'decoded_frames': count, 'fps': fps, 'timing_seconds': timings,
                    'first_pass_backend': 'OpenCV CAP_FFMPEG BGR thumbnails'}


def projection_ver2():
    rng = np.random.Generator(np.random.PCG64(234))
    q, r = np.linalg.qr(rng.standard_normal((8, 8)), mode='reduced')
    if np.any(np.diag(r) == 0):
        raise ValueError('Degenerate projection')
    q *= np.where(np.diag(r) < 0, -1., 1.)
    return np.asarray(q, dtype='<f8', order='C')


def project_ver2(rows, count, q):
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 8 or not np.isfinite(x).all():
        raise ValueError('Expected finite [F,8] thumbnail rows')
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (8, 8) or not np.isfinite(q).all() or not np.allclose(q.T @ q, np.eye(8), atol=1e-12):
        raise ValueError('Invalid fixed projection')
    norm = np.linalg.norm(x, axis=1)
    normalized = np.divide(x, norm[:, None], out=np.zeros_like(x), where=norm[:, None] != 0)
    d = min(8, count)
    return normalized @ q[:, :d] * np.sqrt(8 / d)


def select_ver2(rows, bounds, method, seed=0, q=None):
    """Return one original index per window and deterministic diagnostics."""
    if method not in METHODS_VER2 or not bounds or bounds[0][0] != 0:
        raise ValueError('Invalid method/windows')
    if any(a >= b or a < 0 or (j and bounds[j-1][1] != a) for j, (a, b) in enumerate(bounds)):
        raise ValueError('Windows must partition contiguous coverage')
    chosen = [a for a, _ in bounds]
    detail = {'exchanges': 0, 'auxiliary_exchanges': 0, 'evaluated_candidates': 0,
              'changed_windows_per_sweep': [], 'termination': 'baseline', 'specification': SPEC_VER2}
    if method == 'random_window':
        rng = random.Random(seed)
        chosen = [rng.randrange(a, b) for a, b in bounds]
    elif method != 'uniform':
        if rows is None or len(rows) != bounds[-1][1]:
            raise ValueError('One thumbnail row per in-coverage frame required')
        z = project_ver2(rows, len(chosen), projection_ver2() if q is None else q)
        eye = np.eye(z.shape[1])
        def matrix(indices):
            return eye + z[indices].T @ z[indices]
        current = matrix(chosen)
        detail.update(dimension=z.shape[1], zero_rows=int(np.count_nonzero(~np.any(z, axis=1))),
                      initial_objectives=objectives(current), initial_rank=int(np.linalg.matrix_rank(z[chosen])))
        detail['termination'] = 'sweep_limit'
        for _ in range(SPEC_VER2['max_sweeps']):
            changed = 0
            for j, (a, b) in enumerate(bounds):
                old = chosen[j]
                origin = objectives(current)
                best, best_values, best_primary = old, origin, False
                etol = 1e-10 * max(1., float(np.linalg.eigvalsh(current)[-1]))
                dtol = 1e-12 * max(1., abs(origin['logdet']))
                atol = 1e-12 * max(1., abs(origin['inverse_trace']))
                for candidate in range(a, b):
                    if candidate == old:
                        continue
                    trial = current - np.outer(z[old], z[old]) + np.outer(z[candidate], z[candidate])
                    values = objectives(trial)
                    detail['evaluated_candidates'] += 1
                    if method == 'd_optimal':
                        eligible = values['logdet'] > origin['logdet'] + dtol
                        better = values['logdet'] > best_values['logdet'] + dtol
                        tied = abs(values['logdet'] - best_values['logdet']) <= dtol
                        primary = True
                    elif method == 'a_optimal':
                        eligible = values['inverse_trace'] < origin['inverse_trace'] - atol
                        better = values['inverse_trace'] < best_values['inverse_trace'] - atol
                        tied = abs(values['inverse_trace'] - best_values['inverse_trace']) <= atol
                        primary = True
                    else:
                        m, o, bm = values['minimum_eigenvalue'], origin['minimum_eigenvalue'], best_values['minimum_eigenvalue']
                        primary = m > o + etol
                        eligible = primary or (abs(m-o) <= etol and values['logdet'] > origin['logdet'] + dtol)
                        better = ((primary and not best_primary) or (primary == best_primary and
                                  (m > bm + etol or (abs(m-bm) <= etol and values['logdet'] > best_values['logdet'] + dtol))))
                        tied = primary == best_primary and abs(m-bm) <= etol and abs(values['logdet']-best_values['logdet']) <= dtol
                    if eligible and (best == old or better or (tied and candidate < best)):
                        best, best_values, best_primary = candidate, values, primary
                if best != old:
                    chosen[j] = best
                    current = matrix(chosen)
                    actual = objectives(current)
                    if ((method == 'd_optimal' and actual['logdet'] < origin['logdet'] - dtol)
                            or (method == 'a_optimal' and actual['inverse_trace'] > origin['inverse_trace'] + atol)
                            or (method == 'e_optimal' and actual['minimum_eigenvalue'] < origin['minimum_eigenvalue'] - etol)):
                        raise ValueError('Rebuilt objective violated monotonicity')
                    detail['exchanges'] += 1
                    detail['auxiliary_exchanges'] += int(method == 'e_optimal' and not best_primary)
                    changed += 1
            current = matrix(chosen)
            detail['changed_windows_per_sweep'].append(changed)
            if changed == 0:
                detail['termination'] = 'no_improving_exchange'
                break
        detail.update(final_objectives=objectives(current), final_rank=int(np.linalg.matrix_rank(z[chosen])),
                      minimum_at_ridge=bool(abs(np.linalg.eigvalsh(current)[0]-1) <= 1e-10 * max(1.,np.linalg.eigvalsh(current)[-1])),
                      uniform_overlap=sum(i == a for i, (a, _) in zip(chosen, bounds)) / len(chosen))
        if not np.any(z):
            detail.update(fallback='zero_projection_uniform')
    if len(chosen) != len(bounds) or any(not a <= i < b for i, (a, b) in zip(chosen, bounds)):
        raise ValueError('Window constraint violated')
    return chosen, detail


def projection_hash_ver2(q):
    return hashlib.sha256(np.asarray(q, dtype='<f8', order='C').tobytes()).hexdigest()
