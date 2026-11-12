"""Fused H_0 + H_1: process four adjacent amplitudes per pass."""

from __future__ import annotations

import numpy as np
from numba import njit

from common import validate_state_vector


@njit(cache=True)
def _fusion_kernel(state: np.ndarray) -> None:
    # LSB indexing: i,i+1,i+2,i+3 represent local basis |00>,|01>,|10>,|11>.
    # This is (H tensor H), formed without materializing a 4x4 matrix.
    for i in range(0, state.size, 4):
        a = state[i]
        b = state[i + 1]
        c = state[i + 2]
        d = state[i + 3]
        state[i] = (a + b + c + d) * 0.5
        state[i + 1] = (a - b + c - d) * 0.5
        state[i + 2] = (a + b - c - d) * 0.5
        state[i + 3] = (a - b - c + d) * 0.5


def run_fusion_inplace(state: np.ndarray) -> None:
    """Apply the equivalent fused H_0/H_1 operation in-place."""
    validate_state_vector(state)
    _fusion_kernel(state)
