"""Unfused reference: apply H on qubit 0, then on qubit 1.

Qubit 0 is the least-significant bit of the state-vector index.
Each H gate traverses the full state vector in a separate pass.
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
    _apply_h_inplace(state, 0)
    _apply_h_inplace(state, 1)


def run_baseline_inplace(state: np.ndarray) -> None:
    """Apply H_0 followed by H_1 to a validated state vector in-place."""
    validate_state_vector(state)
    _baseline_kernel(state)
