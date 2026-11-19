"""Prepared Baseline sequences and analytical stride/access profiles."""
from __future__ import annotations

import numpy as np
import pytest

from baseline import (
    baseline_access_profile,
    prepare_baseline_sequence,
    run_baseline_sequence_inplace,
)
from common import random_state


@pytest.mark.parametrize("n,targets", [
    (2, ()), (2, (0,)), (2, (0, 1)),
    (4, (0, 3)), (6, (2, 5, 2)),
    (8, (0, 1, 2, 3)), (12, (0, 5, 10)),
])
def test_prepared_matches_unprepared_baseline(n, targets):
    source = random_state(n, seed=100 + n)
    prepared = prepare_baseline_sequence(n, targets)
    one, two = source.copy(), source.copy()
    prepared.run_inplace(one)
    run_baseline_sequence_inplace(two, targets)
    np.testing.assert_allclose(one, two, atol=1e-12, rtol=1e-12)
    assert prepared.targets == targets


def test_prepared_plan_snapshots_mutable_input_and_is_repeatable():
    targets = [0, 3, 1]
    prepared = prepare_baseline_sequence(4, targets)
    targets[0] = 2
    assert prepared.targets == (0, 3, 1)

    source = random_state(4, seed=123)
    first, second = source.copy(), source.copy()
    prepared.run_inplace(first)
    prepared.run_inplace(second)
    np.testing.assert_array_equal(first, second)


def test_plan_rejects_wrong_state_size_without_mutating_input():
    plan = prepare_baseline_sequence(5, (0, 4))
    state = random_state(4)
    untouched = state.copy()
    with pytest.raises(ValueError, match="state length"):
        plan.run_inplace(state)
    np.testing.assert_array_equal(state, untouched)


@pytest.mark.parametrize("n,targets,exc", [
    (1, (0,), ValueError), (True, (0,), TypeError),
    (4, (0, 4), ValueError), (4, (0, False), TypeError),
    (4, (0, 1.0), TypeError), (4, "01", TypeError),
])
def test_plan_invalid_parameters(n, targets, exc):
    with pytest.raises(exc):
        prepare_baseline_sequence(n, targets)


def test_profile_pair_strides_and_sample_pairs():
    profile = baseline_access_profile(20, (0, 5, 10))
    assert profile["state_bytes"] == 1 << 24
    assert [r["pair_stride_amplitudes"] for r in profile["per_gate"]] == [1, 32, 1024]
    assert [r["pair_stride_bytes"] for r in profile["per_gate"]] == [16, 512, 16384]
    assert profile["per_gate"][0]["sample_index_pairs"] == [
        (0, 1), (2, 3), (4, 5), (6, 7),
    ]
    assert profile["per_gate"][1]["sample_index_pairs"] == [
        (0, 32), (1, 33), (2, 34), (3, 35),
    ]
    assert profile["per_gate"][2]["sample_index_pairs"] == [
        (0, 1024), (1, 1025), (2, 1026), (3, 1027),
    ]
    assert profile["totals"]["total_logical_bytes"] == 3 * (1 << 25)


def test_profile_preserves_repetition_and_empty_sequence():
    p = baseline_access_profile(3, (1, 1))
    assert len(p["per_gate"]) == 2
    assert [r["target"] for r in p["per_gate"]] == [1, 1]
    assert p["per_gate"][0]["pairs_processed"] == 4
    assert p["totals"]["amplitude_reads"] == 16
    empty = baseline_access_profile(3, ())
    assert empty["per_gate"] == []
    assert empty["totals"]["total_logical_bytes"] == 0


def test_profile_rejects_invalid_target():
    with pytest.raises(ValueError):
        baseline_access_profile(4, (0, 5))
