"""Baseline interface regression checks using independent tensor-axis math."""
from __future__ import annotations

import numpy as np
import pytest

from baseline import (
    prepare_baseline_sequence,
    run_baseline_inplace,
    run_baseline_sequence_inplace,
)
from baseline_reference import h_sequence_numpy_reference
from common import random_state
from verify_baseline_a import CASES, verify_case


@pytest.mark.parametrize("n,targets", CASES)
def test_acceptance_scenarios(n, targets):
    result = verify_case(n, targets)
    assert result["max_absolute_error"] < 1e-12
    assert result["norm_error"] < 1e-12


@pytest.mark.parametrize("n,targets", [
    (4, [0]), (5, [4]), (5, [2, 4, 1]),
    (6, [0, 0, 5, 5, 1]), (7, [6, 4, 2, 0]),
])
def test_independent_tensor_oracle_with_non_normalized_complex_input(n, targets):
    generator = np.random.default_rng(7919 + n)
    original = np.ascontiguousarray(
        generator.standard_normal(1 << n) + 1j * generator.standard_normal(1 << n),
        dtype=np.complex128,
    )
    snapshot = original.copy()
    expected = h_sequence_numpy_reference(original, targets)
    assert expected is not original
    actual = original.copy()
    run_baseline_sequence_inplace(actual, targets)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(original, snapshot)


@pytest.mark.parametrize("q1,q2", [
    (0, 1), (0, 5), (4, 1), (5, 0),
])
def test_legacy_two_gate_matches_sequence_and_tensor_oracle(q1, q2):
    source = random_state(6, seed=2026 + q1 * 7 + q2)
    expected = h_sequence_numpy_reference(source, [q1, q2])
    legacy = source.copy()
    seq = source.copy()
    prepared = source.copy()

    run_baseline_inplace(legacy, q1, q2)
    run_baseline_sequence_inplace(seq, [q1, q2])
    prepare_baseline_sequence(6, [q1, q2]).run_inplace(prepared)
    np.testing.assert_allclose(legacy, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(seq, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(prepared, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bad_state", [
    lambda: np.ones((2, 2), dtype=np.complex128),
    lambda: np.ones(4, dtype=np.complex64),
    lambda: np.ones(3, dtype=np.complex128),
    lambda: np.ones(8, dtype=np.complex128)[::2],
])
def test_reference_and_baseline_reject_invalid_state(bad_state):
    invalid = bad_state()
    with pytest.raises(ValueError):
        h_sequence_numpy_reference(invalid, [0, 1])
    with pytest.raises(ValueError):
        run_baseline_sequence_inplace(invalid, [0, 1])


def test_readonly_state_is_rejected_by_reference_baseline_and_plan():
    data = random_state(4)
    data.setflags(write=False)
    with pytest.raises(ValueError):
        h_sequence_numpy_reference(data, [0, 1])
    with pytest.raises(ValueError):
        run_baseline_inplace(data)
    with pytest.raises(ValueError):
        prepare_baseline_sequence(4, [0, 1]).run_inplace(data)


def test_empty_sequence_returns_independent_copy_from_numpy_oracle():
    source = random_state(5)
    expected = h_sequence_numpy_reference(source, [])
    assert expected is not source
    np.testing.assert_array_equal(expected, source)


def test_repeated_h_on_same_target_is_executed_not_skipped():
    source = random_state(6, seed=351)
    actual = source.copy()
    run_baseline_sequence_inplace(actual, [3, 3])
    np.testing.assert_allclose(actual, source, rtol=1e-12, atol=1e-12)
    # No optimization/cancellation allowed by Baseline's task contract.
    assert prepare_baseline_sequence(6, [3, 3]).targets == (3, 3)
