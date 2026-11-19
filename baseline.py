"""Sequential Hadamard-gate reference, with configurable target qubits.

Indexing is little-endian: qubit 0 is the least-significant state index bit.
Each gate makes a separate full-array pass. The default H0 -> H1 path is
preserved for reproducibility with the original experiment A baseline.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from common import validate_state_vector


@njit(cache=True)
def _apply_h_inplace(state: np.ndarray, target: int) -> None:
    stride = 1 << target
    span = stride << 1
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for base in range(0, state.size, span):
        for offset in range(stride):
            i = base + offset
            j = i + stride
            a = state[i]
            b = state[j]
            state[i] = (a + b) * inv_sqrt2
            state[j] = (a - b) * inv_sqrt2


@njit(cache=True)
def _baseline_kernel(state: np.ndarray) -> None:
    """Original specialized H0 -> H1 kernel; retained for existing benchmarks."""
    _apply_h_inplace(state, 0)
    _apply_h_inplace(state, 1)


@njit(cache=True)
def _baseline_targets_kernel(state: np.ndarray, q1: int, q2: int) -> None:
    """Unfused H_{q1}, then H_{q2}; no intermediate gate fusion."""
    _apply_h_inplace(state, q1)
    _apply_h_inplace(state, q2)


def run_baseline_inplace(state: np.ndarray, q1: int = 0, q2: int = 1) -> None:
    """Apply two H gates in-place; targets must be distinct valid qubit indices.

    Both operations use complex128 and two independent sweeps over the input.
    Original one-argument calls retain exactly the original Numba kernel.
    """
    validate_state_vector(state)
    n_qubits = state.size.bit_length() - 1
    if type(q1) is not int or type(q2) is not int:
        raise TypeError("q1 and q2 must be Python integer qubit indices (not bool)")
    if not (0 <= q1 < n_qubits and 0 <= q2 < n_qubits):
        raise ValueError(f"qubit indices must satisfy 0 <= q1,q2 < {n_qubits}")
    if q1 == q2:
        raise ValueError("q1 and q2 must refer to different qubits")

    if q1 == 0 and q2 == 1:
        _baseline_kernel(state)
    else:
        _baseline_targets_kernel(state, q1, q2)
