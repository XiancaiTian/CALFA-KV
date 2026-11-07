"""RoPE-pair constraints and two-pass finite-rate grouping policy."""

from itertools import combinations

import random

SEEDS = (210745926, 210745927)

def validate(permutation):
    if len(permutation) != 8:
        raise ValueError('expected 8 KV heads')
    for row in permutation:
        if len(row) != 128 or sorted(row) != list(range(128)):
            raise ValueError('each head must be a permutation of 0..127')
        if any(row[i + 1] - row[i] != 64 for i in range(0, 128, 2)):
            raise ValueError('native PERM_ROPE requires adjacent low/high RoPE pairs')
        for group in range(16):
            values = set(row[8 * group:8 * group + 8])
            if len(values) != 8 or {value ^ 64 for value in values} != values:
                raise ValueError('a RoPE pair was split across codebooks')
    return True

def split():
    outer_fit = [w for w in range(64) if w % 4 != 3]
    outer_validation = [w for w in range(64) if w % 4 == 3]
    inner_fit = [w for i, w in enumerate(outer_fit) if i % 4 != 3]
    inner_selection = [w for i, w in enumerate(outer_fit) if i % 4 == 3]
    return dict(inner_fit=inner_fit, inner_selection=inner_selection,
                outer_fit=outer_fit, outer_validation=outer_validation)

def channels(pairs):
    low = sorted(pairs)
    assert len(low) == 4 and len(set(low)) == 4 and all(0 <= p < 64 for p in low)
    return [channel for p in low for channel in (p, p + 64)]

def partitions(first, second):
    """35 unordered partitions of 8 pairs into 4+4, retaining the lowest pair on left."""
    assert len(first) == len(second) == 8
    low = sorted(set(x % 64 for x in first + second))
    assert len(low) == 8
    result = []
    for tail in combinations(low[1:], 3):
        left = [low[0], *tail]
        right = sorted(set(low).difference(left))
        result.append((channels(left), channels(right)))
    assert len(result) == 35
    return result

def blocks(layer, head, pass_index=0):
    """First use disjoint group pairs, then shift the order by one group."""
    if pass_index not in (0, 1):
        raise ValueError("pass_index must be 0 or 1")
    ids = list(range(16))
    random.Random(20261021 + 100 * layer + head).shuffle(ids)
    if pass_index == 1:
        ids = ids[1:] + ids[:1]
    return list(zip(ids[::2], ids[1::2]))


def decide(before, after, margin):
    """Two seeds; balanced relative MSE/Fisher gain, neither component >0.5% worse."""
    assert len(before) == len(after) == 2
    ratios = []
    for old, new in zip(before, after):
        if min(old['sse'], old['fisher_sse']) <= 0:
            return False, None
        ordinary = new['sse'] / old['sse']
        fisher = new['fisher_sse'] / old['fisher_sse']
        if max(ordinary, fisher) > 1.005:
            return False, None
        ratios.append((ordinary + fisher) / 2)
    return all(r <= 1 - margin for r in ratios), sum(ratios) / 2
