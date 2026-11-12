"""Tests are deterministic and run without GPU or a quantum simulator."""

import numpy as np
import pytest

from baseline import run_baseline_inplace
from common import random_state
from fusion import run_fusion_inplace


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 8, 10])
@pytest.mark.parametrize("seed", [0, 17, 2026])
def test_equivalence_on_random_states(n_qubits: int, seed: int) -> None:
    original = random_state(n_qubits, seed)
    baseline = original.copy()
    fused = original.copy()
    run_baseline_inplace(baseline)
    run_fusion_inplace(fused)
    np.testing.assert_allclose(baseline, fused, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(fused), 1.0, atol=1e-12)


def test_basis_state_zero() -> None:
    for n in [2, 3, 5]:
        original = np.zeros(1 << n, dtype=np.complex128)
        original[0] = 1.0
        baseline, fused = original.copy(), original.copy()
        run_baseline_inplace(baseline)
        run_fusion_inplace(fused)
        np.testing.assert_allclose(baseline, fused, atol=1e-12)


def test_two_qubit_numpy_matrix_oracle() -> None:
    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
    oracle = np.kron(h, h)
    original = random_state(2, 123)
    expected = oracle @ original
    for implementation in [run_baseline_inplace, run_fusion_inplace]:
        actual = original.copy()
        implementation(actual)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_two_applications_restore_original() -> None:
    original = random_state(6, 99)
    for implementation in [run_baseline_inplace, run_fusion_inplace]:
        actual = original.copy()
        implementation(actual)
        implementation(actual)
        np.testing.assert_allclose(actual, original, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("implementation", [run_baseline_inplace, run_fusion_inplace])
def test_reject_invalid_inputs(implementation) -> None:
    with pytest.raises(ValueError):
        implementation(np.zeros(3, dtype=np.complex128))
    with pytest.raises(ValueError):
        implementation(np.zeros(4, dtype=np.complex64))
    with pytest.raises(ValueError):
        implementation(np.zeros(2, dtype=np.complex128))
