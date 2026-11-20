"""D-module cross-PR acceptance tests. Copy into the repository ROOT.

Required actual PR APIs:
  A PR #4: baseline.run_baseline_inplace(state, q1=0, q2=1)
  B PR #5: fusion.run_fusion_inplace(state, q1=0, q2=1)

No timing or cache-miss claims are made by this test module.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from baseline import run_baseline_inplace
from fusion import run_fusion_inplace
from d_oracle import dense_expected, expected_amplitude, sample_output_indices

METHODS = (run_baseline_inplace, run_fusion_inplace)
ATOL = 1e-12
RTOL = 1e-12


def make_state(n: int, seed: int, normalized: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = np.ascontiguousarray(
        rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n),
        dtype=np.complex128,
    )
    if normalized:
        state /= np.linalg.norm(state)
    return state


def run(method, original: np.ndarray, q1: int, q2: int) -> np.ndarray:
    actual = original.copy()
    result = method(actual, q1=q1, q2=q2)
    assert result is None, "both public APIs are specified as in-place / return None"
    return actual


@pytest.mark.parametrize("n,q1,q2", [
    (n, q1, q2)
    for n in (2, 3, 4, 5)
    for q1, q2 in permutations(range(n), 2)
])
def test_small_systems_match_independent_dense_matrix(n, q1, q2):
    """All ordered distinct target pairs, two seeds including complex phase."""
    for seed in (1, 2026):
        original = make_state(n, seed)
        snapshot = original.copy()
        expected = dense_expected(original, q1, q2)
        baseline = run(run_baseline_inplace, original, q1, q2)
        fusion = run(run_fusion_inplace, original, q1, q2)
        np.testing.assert_allclose(baseline, expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(fusion, expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(fusion, baseline, rtol=RTOL, atol=ATOL)
        np.testing.assert_array_equal(original, snapshot)


@pytest.mark.parametrize("n,q1,q2", [(2,0,1), (3,0,2), (4,1,3)])
def test_every_computational_basis_state_and_phase_sign(n, q1, q2):
    """Exhaustive basis vectors; exposes LSB/MSB and negative-sign mistakes."""
    dimension = 1 << n
    unitary = None  # build once, not for each basis vector
    from d_oracle import two_h_matrix
    unitary = two_h_matrix(n, q1, q2)
    for basis in range(dimension):
        initial = np.zeros(dimension, dtype=np.complex128)
        initial[basis] = 1
        expected = unitary[:, basis]
        for method in METHODS:
            actual = run(method, initial, q1, q2)
            np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n,q1,q2", [
    (8,0,7), (9,2,8), (10,0,9), (12,0,5), (12,11,0)
])
def test_medium_size_against_sparse_independent_formula(n, q1, q2):
    """No huge Kronecker matrix: independently check 128+ selected amplitudes."""
    original = make_state(n, 3300 + n)
    baseline = run(run_baseline_inplace, original, q1, q2)
    fusion = run(run_fusion_inplace, original, q1, q2)
    np.testing.assert_allclose(baseline, fusion, rtol=RTOL, atol=ATOL)
    for output_index in sample_output_indices(n, q1, q2, count=128):
        expected = expected_amplitude(original, int(output_index), q1, q2)
        np.testing.assert_allclose(baseline[output_index], expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(fusion[output_index], expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("method", METHODS)
def test_non_normalized_complex_inputs_are_linear(method):
    """Input normalization is NOT required by common.validate_state_vector."""
    x, y = make_state(6, 781, normalized=False), make_state(6, 782, normalized=False)
    a, b = 1.3 - 0.4j, -0.2 + 0.8j
    lhs = run(method, np.ascontiguousarray(a*x + b*y), 5, 0)
    rhs = a*run(method, x, 5, 0) + b*run(method, y, 5, 0)
    np.testing.assert_allclose(lhs, rhs, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("method", METHODS)
def test_inverse_and_reverse_order(method):
    original = make_state(9, 5509)
    forward = run(method, original, 0, 8)
    reverse = run(method, original, 8, 0)
    np.testing.assert_allclose(forward, reverse, rtol=RTOL, atol=ATOL)
    recovered = run(method, forward, 0, 8)
    np.testing.assert_allclose(recovered, original, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.linalg.norm(forward), np.linalg.norm(original),
                               rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("method", METHODS)
def test_returns_none_and_modifies_the_original_buffer(method):
    source = make_state(4, 893)
    actual = source.copy()
    pointer_before = actual.ctypes.data
    result = method(actual, q1=0, q2=3)
    assert result is None
    assert actual.ctypes.data == pointer_before
    np.testing.assert_allclose(actual, dense_expected(source, 0, 3), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("bad_q1,bad_q2,exception", [
    (-1, 1, ValueError), (0, 4, ValueError), (1, 1, ValueError),
    (True, 1, TypeError), (0, False, TypeError),
    (1.0, 2, TypeError), (np.int64(0), 2, TypeError),
])
@pytest.mark.parametrize("method", METHODS)
def test_invalid_targets_rejected_without_mutation(method, bad_q1, bad_q2, exception):
    state = make_state(4, 671)
    snapshot = state.copy()
    with pytest.raises(exception):
        method(state, q1=bad_q1, q2=bad_q2)
    np.testing.assert_array_equal(state, snapshot)


@pytest.mark.parametrize("case", ["wrong_dtype", "wrong_shape", "wrong_size",
                                  "read_only", "non_contiguous", "too_small"])
@pytest.mark.parametrize("method", METHODS)
def test_invalid_state_rejected_before_mutation(method, case):
    if case == "wrong_dtype":
        state = np.arange(8, dtype=np.complex64)
    elif case == "wrong_shape":
        state = np.zeros((2, 4), dtype=np.complex128)
    elif case == "wrong_size":
        state = np.ones(6, dtype=np.complex128)
    elif case == "read_only":
        state = np.arange(8, dtype=np.complex128)
        state.setflags(write=False)
    elif case == "non_contiguous":
        state = np.arange(16, dtype=np.complex128)[::2]
    else:
        state = np.zeros(2, dtype=np.complex128)
    snapshot = state.copy()
    with pytest.raises(ValueError):
        method(state, q1=0, q2=1)
    np.testing.assert_array_equal(state, snapshot)
