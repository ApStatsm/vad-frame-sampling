"""Geometry-only cached-grid experiment. Never accepts features or labels."""

import random

METHODS_VER3 = ('uniform_ver3', 'paired_ver3', 'relocated_ver3')
SPEC_VER3 = {'budget': 0.125, 'group_slots': 8, 'replace_slots': [3, 4],
             'pair_gap_reference_rows': 1, 'incomplete_group': 'unchanged',
             'random_seeds': [0, 1, 2, 3, 4], 'score_anchor': 'actual_selected_frame'}


def select_ver3(n, method, seed):
    if type(n) is not int or n < 1 or method not in METHODS_VER3:
        raise ValueError('Invalid ver3 candidate count or method')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Invalid ver3 seed')
    uniform = list(range(0, n, 8))
    selected = uniform.copy()
    rng = random.Random(seed)
    changes = []
    if method != 'uniform_ver3':
        for start in range(0, len(uniform) - 7, 8):
            left, right = uniform[start + 3:start + 5]
            if method == 'paired_ver3':
                new = [left + 3, left + 4]
            else:
                # Both moved; same interval/support and changed slots as paired.
                # Exclude adjacent pairs to provide a dispersed relocation control.
                choices = [(a, b) for a in range(left + 1, right)
                           for b in range(a + 2, right)]
                new = list(rng.choice(choices))
            selected[start + 3:start + 5] = new
            changes.append({'slot_start': start, 'old_rows': [left, right], 'new_rows': new})
    if (len(selected) != (n + 7) // 8 or selected != sorted(set(selected))
            or selected[0] != uniform[0] or selected[-1] != uniform[-1]
            or selected[-1] >= n):
        raise ValueError('ver3 violated K, endpoints, uniqueness, order or bounds')
    return selected, {'protocol_ver3': SPEC_VER3, 'changes': changes,
                      'groups_changed': len(changes), 'changed_slots': 2 * len(changes),
                      'uniform_rows': uniform, 'no_complete_group': len(uniform) < 8}
