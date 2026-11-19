"""Independent matrix-based checks for customizable two-H baseline."""
from __future__ import annotations
import numpy as np
import pytest
from baseline import run_baseline_inplace
from common import random_state
from fusion import run_fusion_inplace

H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
I = np.eye(2, dtype=np.complex128)


def matrix_oracle(n: int, q1: int, q2: int) -> np.ndarray:
    """Independent full-matrix oracle for small n (least-significant-bit order)."""
    op = np.array([[1.0]], dtype=np.complex128)
    for q in range(n - 1, -1, -1):
        op = np.kron(op, H if q in (q1, q2) else I)
    return op


@pytest.mark.parametrize("n,q1,q2", [
    (2, 0, 1), (3, 0, 2), (4, 0, 3), (4, 1, 2),
    (5, 0, 4), (5, 4, 1), (6, 2, 5),
])
def test_two_h_matches_independent_matrix(n: int, q1: int, q2: int) -> None:
    initial = random_state(n, seed=21 + n)
    actual = initial.copy()
    out = run_baseline_inplace(actual, q1=q1, q2=q2)
    assert out is None
    np.testing.assert_allclose(actual, matrix_oracle(n, q1, q2) @ initial, rtol=1e-12, atol=1e-12)


def test_original_default_matches_fusion() -> None:
    original = random_state(8, seed=2026)
    a, b, c = original.copy(), original.copy(), original.copy()
    run_baseline_inplace(a)
    run_baseline_inplace(b, q1=0, q2=1)
    run_fusion_inplace(c)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a, c, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("q1,q2,exception", [
    (-1, 1, ValueError), (0, 4, ValueError), (3, 3, ValueError),
    (False, 1, TypeError), (0, 1.0, TypeError), (np.int64(0), 1, TypeError),
])
def test_invalid_target_indices_leave_input_unchanged(q1, q2, exception) -> None:
    state = random_state(4, 42)
    original = state.copy()
    with pytest.raises(exception):
        run_baseline_inplace(state, q1=q1, q2=q2)
    np.testing.assert_array_equal(state, original)
