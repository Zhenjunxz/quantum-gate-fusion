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

@njit(cache=True)
def _baseline_sequence_kernel(state: np.ndarray, targets: np.ndarray) -> None:
    """Apply each H gate in sequence with a separate full-state sweep."""
    for target in targets:
        _apply_h_inplace(state, target)


def run_baseline_sequence_inplace(state: np.ndarray, targets: tuple[int, ...] | list[int]) -> None:
    """Execute any *H-only* gate sequence without fusion, in-place.

    Examples: targets=(0, 1), (0, 5), or (0, 1, 2, 3).
    Repeated targets are intentional and allowed: (0, 0) implements H0 twice.
    An empty sequence is the identity operation. Targets are checked in full
    before the state is modified. Qubit 0 is the least-significant index bit.

    This is an optional sequence interface for Baseline, not a change to the
    two-gate q1/q2 interface pending integration with the other team modules.
    """
    validate_state_vector(state)
    if not isinstance(targets, (list, tuple)):
        raise TypeError("targets must be a list or tuple of Python integer indices")
    n_qubits = state.size.bit_length() - 1
    for target in targets:
        if type(target) is not int:
            raise TypeError("every target must be a Python int (not bool)")
        if not 0 <= target < n_qubits:
            raise ValueError(f"target {target} must satisfy 0 <= target < {n_qubits}")
    if len(targets) == 0:
        return
    # Conversion occurs outside the JIT kernel; benchmark code must define
    # whether it times this public wrapper or only the compiled computation.
    compiled_targets = np.asarray(targets, dtype=np.int64)
    _baseline_sequence_kernel(state, compiled_targets)


def baseline_logical_memory_accesses(n_qubits: int, gate_count: int) -> dict[str, int]:
    """Count abstract state-amplitude reads/writes for an unfused H sequence.

    Per H gate: 2**n amplitude reads and 2**n amplitude writes.
    Assumes complex128 (16 bytes/amplitude), matching this project.
    Counts are *not* measured CPU cache misses or DRAM traffic.
    """
    if type(n_qubits) is not int or type(gate_count) is not int:
        raise TypeError("n_qubits and gate_count must be Python integers")
    if n_qubits < 2 or gate_count < 0:
        raise ValueError("require n_qubits >= 2 and gate_count >= 0")
    amplitudes = 1 << n_qubits
    count = amplitudes * gate_count
    bytes_per_amplitude = np.dtype(np.complex128).itemsize
    return {
        "amplitude_reads": count,
        "amplitude_writes": count,
        "logical_read_bytes": count * bytes_per_amplitude,
        "logical_write_bytes": count * bytes_per_amplitude,
        "total_logical_bytes": 2 * count * bytes_per_amplitude,
    }
