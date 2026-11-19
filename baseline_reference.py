"""Independent NumPy-only mathematical oracle for the H-only Baseline.

For validation, NOT a timing baseline: uses tensor-axis manipulation rather
than the bit-stride, base/offset loops used by the Numba implementation.
Qubit 0 maps to the last tensor axis (state index least-significant bit).
"""

from __future__ import annotations

import numpy as np

from baseline import prepare_baseline_sequence
from common import validate_state_vector


def h_sequence_numpy_reference(
    state: np.ndarray, targets: tuple[int, ...] | list[int]
) -> np.ndarray:
    """Return a fresh array after applying H targets in their stated order.

    Does not modify *state*. Repeated targets, including H0 followed by H0,
    are executed rather than simplified; an empty list returns a copy.
    """
    validate_state_vector(state)
    n_qubits = state.size.bit_length() - 1
    plan = prepare_baseline_sequence(n_qubits, targets)
    result = state.copy()

    for target in plan.targets:
        # Tensor axes (from left to right) correspond to q_(n-1), ..., q_0.
        moved = np.moveaxis(result.reshape((2,) * n_qubits), n_qubits - 1 - target, 0)
        # Copy both halves before overwriting either; otherwise NumPy views
        # could alias state data and corrupt the second output half.
        zero = moved[0].copy()
        one = moved[1].copy()
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        moved[0] = (zero + one) * inv_sqrt2
        moved[1] = (zero - one) * inv_sqrt2

    return result
