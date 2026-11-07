"""Frozen legal pair-swap nomination and P/C acceptance; no GPU dependency."""
import itertools
import random

import numpy as np


def validate_permutation(permutation):
    p = np.asarray(permutation, dtype=np.int64)
    assert p.shape == (128,) and sorted(p.tolist()) == list(range(128))
    assert np.all(p[::2] < 64) and np.array_equal(p[1::2], p[::2] + 64)
    return p


def swap_pairs(permutation, move):
    p = validate_permutation(permutation).copy()
    first, second, i, j = move
    assert 0 <= first < second < 16 and 0 <= i < 4 and 0 <= j < 4
    a, b = first * 8 + i * 2, second * 8 + j * 2
    p[a:a + 2], p[b:b + 2] = p[b:b + 2].copy(), p[a:a + 2].copy()
    return validate_permutation(p)


def nominate(permutation, source_correlation, residual_correlation, seed):
    """Two source-correlation, two residual-correlation, two random proposals.

    Scores are within-group sums of off-diagonal squared correlations. They
    nominate candidates only; they do not accept a P or estimate achieved rate.
    """
    p = validate_permutation(permutation)
    matrices = [np.asarray(x, dtype=np.float64) ** 2
                for x in (source_correlation, residual_correlation)]
    assert all(m.shape == (128, 128) and np.isfinite(m).all() for m in matrices)
    for matrix in matrices:
        np.fill_diagonal(matrix, 0)

    def score(matrix, columns):
        return float(matrix[np.ix_(columns, columns)].sum() / 2)

    current = [[score(m, p[g * 8:(g + 1) * 8]) for g in range(16)]
               for m in matrices]
    records = []
    for first, second in itertools.combinations(range(16), 2):
        for i, j in itertools.product(range(4), repeat=2):
            move = (first, second, i, j)
            trial = swap_pairs(p, move)
            gains = [sum(score(m, trial[g * 8:(g + 1) * 8]) - old[g]
                         for g in (first, second))
                     for m, old in zip(matrices, current)]
            records.append(dict(move=move, source_gain=gains[0],
                                residual_gain=gains[1]))
    assert len(records) == 1920
    result, selected = [], set()
    for key in ("source_gain", "residual_gain"):
        ordered = sorted(records, key=lambda r: (-r[key], r["move"]))
        for row in [r for r in ordered if r["move"] not in selected][:2]:
            result.append(dict(row, nomination=key))
            selected.add(row["move"])
    remaining = [r for r in records if r["move"] not in selected]
    for row in random.Random(seed).sample(remaining, 2):
        result.append(dict(row, nomination="deterministic_exploration"))
    assert len(result) == len({r["move"] for r in result}) == 6
    return result


def acceptance(parent, control, candidate):
    """Candidate beats current P/C AND matched C-only retrain by >=0.5%.

    Source-stratum KL and fixed-V output SSE cannot exceed either comparator
    by >1%. All comparisons use unrounded local calibration measurements.
    """
    checks = {}
    for name, reference in (("parent", parent), ("control", control)):
        checks[name + "_kl_gain"] = (
            reference["kl"] > 1e-12
            and candidate["kl"] <= reference["kl"] * .995
        )
        for key in ("gpqa_kl", "code_kl", "output_relative_sse"):
            checks[name + "_" + key + "_guard"] = (
                candidate[key] <= reference[key] * 1.01 + 1e-12
            )
    return dict(accepted=all(checks.values()), checks=checks)


def self_checks():
    p = np.stack((np.arange(64), np.arange(64) + 64), -1).flatten()
    rng = np.random.default_rng(901)
    x = rng.normal(size=(400, 128))
    correlation = np.corrcoef(x, rowvar=False)
    proposals = nominate(p, correlation, correlation, 23)
    assert proposals == nominate(p, correlation, correlation, 23)
    for row in proposals:
        changed = swap_pairs(p, row["move"])
        assert np.count_nonzero(changed != p) == 4
        assert np.array_equal(swap_pairs(changed, row["move"]), p)
    base = dict(kl=1., gpqa_kl=1., code_kl=1., output_relative_sse=1.)
    good = dict(base, kl=.994)
    assert acceptance(base, base, good)["accepted"]
    assert not acceptance(base, dict(base, kl=.99), good)["accepted"]
    assert not acceptance(base, base, dict(good, code_kl=1.011))["accepted"]
    assert not acceptance(base, base, dict(good, kl=.996))["accepted"]
    return dict(passed=True, legal_and_involutive_pair_swaps=True,
                six_unique_deterministic_proposals=True,
                parent_and_matched_control_both_required=True,
                source_and_output_guards=True)
