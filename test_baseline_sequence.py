"""Independent reference for the Baseline H-only sequence (no fusion)."""
from __future__ import annotations

import numpy as np
import pytest

from baseline import (
    baseline_logical_memory_accesses,
    run_baseline_inplace,
    run_baseline_sequence_inplace,
)
from common import random_state


H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
I = np.eye(2, dtype=np.complex128)


def independent_h_sequence(initial: np.ndarray, targets: tuple[int, ...]) -> np.ndarray:
    """Build full one-gate matrices independently; apply in specified order."""
    n = initial.size.bit_length() - 1
    state = initial.copy()
    for target in targets:
        operator = np.array([[1]], dtype=np.complex128)
        for qubit in range(n - 1, -1, -1):
            operator = np.kron(operator, H if qubit == target else I)
        state = operator @ state
    return state


@pytest.mark.parametrize("n,targets", [
    (2, ()), (2, (0,)), (2, (0, 1)), (2, (0, 0)),
    (3, (2, 0, 1)), (4, (0, 2, 1, 3)),
    (5, (4, 0, 4, 1, 0)), (6, (0, 5, 2, 1, 4, 5)),
])
def test_h_sequence_matches_independent_dense_matrix(n, targets):
    original = random_state(n, seed=2026 + n)
    actual = original.copy()
    result = run_baseline_sequence_inplace(actual, targets)
    assert result is None
    np.testing.assert_allclose(
        actual, independent_h_sequence(original, targets), rtol=1e-12, atol=1e-12
    )


def test_sequence_list_and_tuple_are_equivalent():
    initial = random_state(5)
    a, b = initial.copy(), initial.copy()
    run_baseline_sequence_inplace(a, [0, 3, 1])
    run_baseline_sequence_inplace(b, (0, 3, 1))
    np.testing.assert_array_equal(a, b)


def test_two_gate_default_is_compatible():
    initial = random_state(8)
    a, b = initial.copy(), initial.copy()
    run_baseline_inplace(a)
    run_baseline_sequence_inplace(b, (0, 1))
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("targets,expected_error", [
    ((0, -1), ValueError), ((0, 4), ValueError), ((1, False), TypeError),
    ((2.0,), TypeError), ((np.int64(1),), TypeError),
    ("01", TypeError), ((0, None), TypeError),
])
def test_invalid_targets_do_not_modify_state(targets, expected_error):
    initial = random_state(4)
    state = initial.copy()
    with pytest.raises(expected_error):
        run_baseline_sequence_inplace(state, targets)
    np.testing.assert_array_equal(state, initial)


def test_logical_memory_count():
    assert baseline_logical_memory_accesses(2, 2) == {
        "amplitude_reads": 8,
        "amplitude_writes": 8,
        "logical_read_bytes": 128,
        "logical_write_bytes": 128,
        "total_logical_bytes": 256,
    }
    assert baseline_logical_memory_accesses(20, 0)["total_logical_bytes"] == 0
    assert baseline_logical_memory_accesses(20, 4)["total_logical_bytes"] == 134217728


@pytest.mark.parametrize("args,exc", [
    ((1, 2), ValueError), ((3, -1), ValueError),
    ((True, 2), TypeError), ((4, 1.0), TypeError),
])
def test_logical_model_rejects_invalid_args(args, exc):
    with pytest.raises(exc):
        baseline_logical_memory_accesses(*args)
