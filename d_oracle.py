"""Independent mathematical oracles for two Hadamard gates on distinct qubits.

Conventions
-----------
* State index is little-endian: qubit 0 is the least-significant bit.
* The full matrix oracle is for SMALL systems, not performance benchmarking.
* The sparse analytic oracle uses the Walsh-Hadamard sign formula. It can
  check selected output amplitudes without constructing a 2**n x 2**n matrix.
* Neither oracle imports baseline.py or fusion.py.
"""
from __future__ import annotations

import numpy as np

_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
_I = np.eye(2, dtype=np.complex128)


def two_h_matrix(n_qubits: int, q1: int, q2: int) -> np.ndarray:
    """Return the full 2**n square unitary; use only for small n (<= 7)."""
    if not (2 <= n_qubits <= 7):
        raise ValueError("dense oracle restricted to 2 <= n_qubits <= 7")
    if not (0 <= q1 < n_qubits and 0 <= q2 < n_qubits and q1 != q2):
        raise ValueError("q1, q2 must be distinct valid targets")
    result = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    # Leftmost tensor factor is the highest-numbered qubit.
    for qubit in range(n_qubits - 1, -1, -1):
        result = np.kron(result, _H if qubit in (q1, q2) else _I)
    return result


def dense_expected(source: np.ndarray, q1: int, q2: int) -> np.ndarray:
    """Compute a complete independent reference by matrix-vector multiply."""
    n = len(source).bit_length() - 1
    return two_h_matrix(n, q1, q2) @ source


def expected_amplitude(source: np.ndarray, index: int, q1: int, q2: int) -> complex:
    """Calculate ONE output amplitude without referring to implementation loops.

    Let y be the output index, x an input index that differs from y only in
    the two target bits. The coefficient is (-1)^(x_q1*y_q1+x_q2*y_q2)/2.
    This is the two-bit Walsh-Hadamard transform derived from H tensor H.
    """
    n = len(source).bit_length() - 1
    if not (0 <= index < len(source)):
        raise ValueError("output index out of range")
    if not (0 <= q1 < n and 0 <= q2 < n and q1 != q2):
        raise ValueError("invalid target qubits")
    bit1 = 1 << q1
    bit2 = 1 << q2
    base = index & ~(bit1 | bit2)
    y1 = int(bool(index & bit1))
    y2 = int(bool(index & bit2))
    total = 0.0 + 0.0j
    for x1 in (0, 1):
        for x2 in (0, 1):
            x = base | (bit1 if x1 else 0) | (bit2 if x2 else 0)
            sign = -1 if (x1 * y1 + x2 * y2) % 2 else 1
            total += sign * source[x]
    return total * 0.5


def sample_output_indices(n_qubits: int, q1: int, q2: int,
                          count: int = 128, seed: int = 2026) -> np.ndarray:
    """Include target-bit corner cases and deterministic random output indices."""
    size = 1 << n_qubits
    b1, b2 = 1 << q1, 1 << q2
    anchors = {
        0, 1, size - 1, size - 2,
        b1, b2, b1 | b2,
        (size - 1) ^ b1, (size - 1) ^ b2,
        (size - 1) ^ (b1 | b2),
    }
    anchors.update(range(min(size, 16)))
    rng = np.random.default_rng(seed)
    while len(anchors) < min(size, count):
        anchors.add(int(rng.integers(0, size)))
    return np.array(sorted(anchors), dtype=np.int64)
