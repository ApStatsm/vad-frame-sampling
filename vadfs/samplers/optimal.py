"""Deterministic D/A greedy and bounded E exchange on label-free design rows."""

import numpy as np

from .uniform import select as uniform


OPTIMAL_METHODS = ('d_optimal', 'a_optimal', 'e_optimal')
DEFAULTS = {'ridge': 1.0, 'rank_rtol': 1e-8, 'gain_rtol': 1e-12,
            'e_rtol': 1e-10, 'logdet_rtol': 1e-12,
            'e_max_pairs': 4096, 'e_max_rounds': 10, 'e_pool_size': 64}


def settings(options):
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown optimal settings: {sorted(unknown)}')
    result = {**DEFAULTS, **options}
    for key in ('ridge', 'rank_rtol', 'gain_rtol', 'e_rtol', 'logdet_rtol'):
        value = result[key]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('e_max_pairs', 'e_max_rounds', 'e_pool_size'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if result['e_pool_size'] ** 2 > result['e_max_pairs']:
        raise ValueError('E pool square must fit the pair limit')
    return result


def objectives(matrix):
    chol = np.linalg.cholesky(matrix)
    inverse_chol = np.linalg.solve(chol, np.eye(len(matrix)))
    return {'logdet': float(2 * np.log(np.diag(chol)).sum()),
            'inverse_trace': float(np.square(inverse_chol).sum()),
            'minimum_eigenvalue': float(np.linalg.eigvalsh(matrix)[0])}


def gram_rank(rows):
    spectrum = np.linalg.eigvalsh(rows.T @ rows)
    return int(np.count_nonzero(spectrum > 1e-10 * max(0., spectrum[-1])))


def gains(matrix, rows, method):
    chol = np.linalg.cholesky(matrix)
    u = np.linalg.solve(chol.T, np.linalg.solve(chol, rows.T)).T
    q = np.einsum('ij,ij->i', rows, u)
    if not np.isfinite(u).all() or np.any(q < -1e-10):
        raise ValueError('Invalid optimal solve')
    q = np.maximum(q, 0)
    return (np.log1p(q) if method == 'd_optimal'
            else np.einsum('ij,ij->i', u, u) / (1 + q))


def select(rows, count, method, ratio, options=None, *, rank_adapt=True):
    config = settings(options or {})
    z = np.asarray(rows, dtype=np.float64)
    if (method not in OPTIMAL_METHODS or z.ndim != 2 or not 1 <= z.shape[1] <= 8
            or not np.isfinite(z).all() or type(count) is not int or not 1 <= count <= len(z)):
        raise ValueError('Optimal selection needs finite [N,d<=8] rows and 1 <= K <= N')
    if count == len(z):
        return list(range(count)), {'full_budget_bypass': True}
    if rank_adapt:
        _, singular, vt = np.linalg.svd(z, full_matrices=False)
        rank = int(np.count_nonzero(singular > config['rank_rtol'] * singular[0])) if singular[0] else 0
        dimension = min(z.shape[1], count, rank)
    else:
        if z.shape[1] > count:
            raise ValueError('Fit-free selection requires d <= K')
        rank = gram_rank(z)
        dimension = z.shape[1]
    diagnostic = {'rank': rank, 'dimension': dimension, 'full_budget_bypass': False,
                  'rank_adaptation': rank_adapt,
                  'rank_diagnostic': 'SVD relative singular tolerance' if rank_adapt else 'Gram eigenvalue relative tolerance 1e-10; diagnostic only',
                  'fallback': None, 'evaluated_pairs': 0, 'exchanges': 0, 'auxiliary_exchanges': 0}
    if (rank_adapt and dimension == 0) or (not rank_adapt and not np.any(z)):
        diagnostic.update(fallback='zero_rank_uniform' if rank_adapt else 'zero_projection_uniform', termination='zero_rank')
        return uniform(len(z), count, 0, ratio=ratio), diagnostic
    if rank_adapt and dimension < z.shape[1]:
        z = z @ vt[:dimension].T
    identity = config['ridge'] * np.eye(dimension)
    matrix = identity.copy()
    if method != 'e_optimal':
        diagnostic['initial_objectives'] = objectives(matrix)
        remaining = np.ones(len(z), dtype=bool)
        chosen = []
        for _ in range(count):
            indices = np.flatnonzero(remaining)
            values = gains(matrix, z[indices], method)
            best = values.max()
            tolerance = config['gain_rtol'] * max(1.0, abs(best))
            index = int(indices[np.flatnonzero(values >= best - tolerance)[0]])
            chosen.append(index)
            remaining[index] = False
            # Rebuild from the selected set to avoid accumulated rank-one error.
            matrix = identity + z[chosen].T @ z[chosen]
        diagnostic['termination'] = 'budget_filled'
    else:
        chosen = uniform(len(z), count, 0, ratio=ratio)
        matrix = identity + z[chosen].T @ z[chosen]
        diagnostic['initial_objectives'] = objectives(matrix)
        diagnostic['initial_selected_rank'] = gram_rank(z[chosen])
        diagnostic['termination'] = 'round_limit'
        for _ in range(config['e_max_rounds']):
            selected = sorted(chosen)
            unselected = sorted(set(range(len(z))) - set(chosen))
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            tolerance = config['e_rtol'] * max(1.0, float(eigenvalues[-1]))
            old_min = float(eigenvalues[0])
            old_log = objectives(matrix)['logdet']
            logtol = config['logdet_rtol'] * max(1.0, abs(old_log))
            if len(selected) * len(unselected) > config['e_max_pairs']:
                projections = np.square(z @ eigenvectors[:, 0])
                selected = sorted(selected, key=lambda i: (projections[i], i))[:config['e_pool_size']]
                unselected = sorted(unselected, key=lambda i: (-projections[i], i))[:config['e_pool_size']]
            # Lexicographic (removed index, inserted index) final tie break.
            best = None
            for remove in sorted(selected):
                for insert in sorted(unselected):
                    trial = matrix - np.outer(z[remove], z[remove]) + np.outer(z[insert], z[insert])
                    minimum = float(np.linalg.eigvalsh(trial)[0])
                    sign, logdet = np.linalg.slogdet(trial)
                    logdet = float(logdet)
                    if sign <= 0 or not np.isfinite(minimum) or not np.isfinite(logdet):
                        raise ValueError('Invalid E exchange objective')
                    diagnostic['evaluated_pairs'] += 1
                    primary = minimum > old_min + tolerance
                    auxiliary = abs(minimum - old_min) <= tolerance and logdet > old_log + logtol
                    if not (primary or auxiliary):
                        continue
                    if (best is None or (primary and not best[4])
                            or (primary == best[4] and (minimum > best[2] + tolerance
                                or (abs(minimum - best[2]) <= tolerance and logdet > best[3] + logtol)))):
                        best = (remove, insert, minimum, logdet, primary)
            if best is None:
                diagnostic['termination'] = 'no_improving_exchange'
                break
            remove, insert, _, _, primary = best
            chosen.remove(remove)
            chosen.append(insert)
            matrix = identity + z[chosen].T @ z[chosen]
            diagnostic['exchanges'] += 1
            diagnostic['auxiliary_exchanges'] += int(not primary)
    diagnostic['final_objectives'] = objectives(matrix)
    diagnostic['selected_rank'] = gram_rank(z[chosen])
    if method == 'e_optimal':
        diagnostic['minimum_at_ridge'] = bool(abs(diagnostic['final_objectives']['minimum_eigenvalue']
            - config['ridge']) <= config['e_rtol'] * max(1., float(np.linalg.eigvalsh(matrix)[-1])))
    positions = sorted(chosen)
    if len(set(positions)) != count or any(not 0 <= i < len(z) for i in positions):
        raise ValueError('Optimal sampler violated budget/bounds')
    return positions, diagnostic
