"""Check configurable fusion targets against a full NumPy operator."""

from itertools import permutations

import numpy as np
import pytest

from common import random_state
from fusion import run_fusion_inplace


@pytest.mark.parametrize(
    "n_qubits,q1,q2",
    [(n, q1, q2) for n in (3, 4) for q1, q2 in permutations(range(n), 2)],
)
def test_arbitrary_targets_match_tensor_product(n_qubits: int, q1: int, q2: int) -> None:
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    identity = np.eye(2)
    operator = np.eye(1)
    for qubit in reversed(range(n_qubits)):
        operator = np.kron(operator, hadamard if qubit in (q1, q2) else identity)

    original = random_state(n_qubits, seed=42)
    actual = original.copy()
    run_fusion_inplace(actual, q1=q1, q2=q2)

    np.testing.assert_allclose(actual, operator @ original, rtol=1e-12, atol=1e-12)


def test_nonadjacent_targets_keep_untouched_bit_and_signs() -> None:
    state = np.zeros(8, dtype=np.complex128)
    state[3] = 1.0  # |011>, with its middle qubit set.
    expected = np.zeros_like(state)
    expected[[2, 3, 6, 7]] = [0.5, -0.5, 0.5, -0.5]

    run_fusion_inplace(state, q1=0, q2=2)

    np.testing.assert_array_equal(state, expected)


@pytest.mark.parametrize(
    "q1,q2,error",
    [
        (-1, 0, ValueError),
        (0, 3, ValueError),
        (1, 1, ValueError),
        (0, True, TypeError),
        (1.0, 2, TypeError),
    ],
)
def test_invalid_targets_leave_input_unchanged(q1, q2, error) -> None:
    original = random_state(3)
    state = original.copy()

    with pytest.raises(error):
        run_fusion_inplace(state, q1=q1, q2=q2)

    np.testing.assert_array_equal(state, original)
