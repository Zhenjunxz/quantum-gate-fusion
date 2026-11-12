"""Shared input validation and reproducible state generation."""

from __future__ import annotations

import numpy as np


def validate_state_vector(state: np.ndarray) -> None:
    """Require an in-place, 1-D complex128 state of at least two qubits."""
    if not isinstance(state, np.ndarray):
        raise TypeError("state must be a numpy.ndarray")
    if state.ndim != 1 or state.dtype != np.complex128:
        raise ValueError("state must be a 1-D complex128 array")
    n = state.size
    if n < 4 or n & (n - 1):
        raise ValueError("state length must be 2**n with n >= 2")
    if not state.flags.c_contiguous or not state.flags.writeable:
        raise ValueError("state must be contiguous and writable")


def random_state(n_qubits: int, seed: int = 42) -> np.ndarray:
    """Generate a normalized pure state with deterministic random amplitudes."""
    if n_qubits < 2:
        raise ValueError("n_qubits must be >= 2")
    rng = np.random.default_rng(seed)
    size = 1 << n_qubits
    # Explicit dtype to make performance comparisons and memory accounting fair.
    state = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    state = np.ascontiguousarray(state, dtype=np.complex128)
    state /= np.linalg.norm(state)
    return state
