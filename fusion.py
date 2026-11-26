"""Fuse two Hadamard gates on distinct target qubits."""

from __future__ import annotations

import numpy as np
from numba import njit

from common import validate_state_vector


@njit(cache=True)
def _fusion_kernel(state: np.ndarray, q1: int, q2: int) -> None:
    # The gates commute. Use the lower target as the low bit of the local basis.
    low_stride = 1 << min(q1, q2)
    high_stride = 1 << max(q1, q2)

    # Enumerate only base indices where both target bits are zero.
    for block in range(0, state.size, 2 * high_stride):
        for subgroup in range(0, high_stride, 2 * low_stride):
            for offset in range(low_stride):
                i00 = block + subgroup + offset
                i01 = i00 + low_stride
                i10 = i00 + high_stride
                i11 = i10 + low_stride

                # Load the whole group before applying H tensor H in place.
                a, b, c, d = state[i00], state[i01], state[i10], state[i11]
                state[i00] = (a + b + c + d) * 0.5
                state[i01] = (a - b + c - d) * 0.5
                state[i10] = (a + b - c - d) * 0.5
                state[i11] = (a - b - c + d) * 0.5


def run_fusion_inplace(state: np.ndarray, q1: int = 0, q2: int = 1) -> None:
    """Apply two H gates in place, with qubit 0 as the least-significant bit."""
    validate_state_vector(state)
    n_qubits = state.size.bit_length() - 1
    if type(q1) is not int or type(q2) is not int:
        raise TypeError("q1 and q2 must be Python integer qubit indices (not bool)")
    if not (0 <= q1 < n_qubits and 0 <= q2 < n_qubits):
        raise ValueError(f"qubit indices must satisfy 0 <= q1,q2 < {n_qubits}")
    if q1 == q2:
        raise ValueError("q1 and q2 must refer to different qubits")

    _fusion_kernel(state, q1, q2)
