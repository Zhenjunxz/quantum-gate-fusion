"""Run A-only Baseline acceptance checks without Fusion or CPU timing.

Run from repository root:
    python verify_baseline_a.py

This checks the in-place Numba Baseline against the independent tensor-axis
NumPy oracle, both for the legacy two-gate interface and general H sequences.
It does NOT provide benchmark results or cache measurements.
"""
from __future__ import annotations

import numpy as np

from baseline import (
    baseline_access_profile,
    baseline_logical_memory_accesses,
    prepare_baseline_sequence,
    run_baseline_inplace,
    run_baseline_sequence_inplace,
)
from baseline_reference import h_sequence_numpy_reference
from common import random_state


CASES: tuple[tuple[int, tuple[int, ...]], ...] = (
    (2, (0, 1)),
    (6, (0, 5)),
    (8, (7, 3, 0)),
    (12, (0, 1, 5, 10)),
    (16, (15, 0, 15, 1)),
    (5, (0, 0)),
    (5, ()),
)


def verify_case(n_qubits: int, targets: tuple[int, ...]) -> dict:
    """Check A's numerical result against an algorithmically different oracle."""
    source = random_state(n_qubits, seed=2026 + n_qubits + len(targets))
    snapshot = source.copy()
    actual = source.copy()
    expected = h_sequence_numpy_reference(source, list(targets))

    # Validate all three A execution routes without changing Fusion code.
    run_baseline_sequence_inplace(actual, list(targets))
    plan = prepare_baseline_sequence(n_qubits, list(targets))
    prepared = source.copy()
    plan.run_inplace(prepared)

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(prepared, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(source, snapshot)
    assert np.isfinite(actual).all()
    if len(targets) == 2 and targets[0] != targets[1]:
        two_gate = source.copy()
        run_baseline_inplace(two_gate, q1=targets[0], q2=targets[1])
        np.testing.assert_allclose(two_gate, expected, atol=1e-12, rtol=1e-12)

    profile = baseline_access_profile(n_qubits, list(targets))
    logical = baseline_logical_memory_accesses(n_qubits, len(targets))
    assert profile["totals"] == logical
    assert len(profile["per_gate"]) == len(targets)
    assert logical["amplitude_reads"] == (1 << n_qubits) * len(targets)
    max_error = float(np.max(np.abs(actual - expected)))
    norm_error = float(abs(np.linalg.norm(actual) - np.linalg.norm(source)))
    return {
        "n_qubits": n_qubits,
        "targets": targets,
        "max_absolute_error": max_error,
        "norm_error": norm_error,
        "logical_read_write_bytes": logical["total_logical_bytes"],
    }


def main() -> None:
    print("A-only Baseline acceptance (NumPy tensor oracle; not timing)")
    for n, targets in CASES:
        result = verify_case(n, targets)
        print(
            f"PASS n={n:>2} targets={targets!s:<18} "
            f"max_err={result['max_absolute_error']:.3e} "
            f"norm_err={result['norm_error']:.3e} "
            f"logical_bytes={result['logical_read_write_bytes']}"
        )
    print(f"All {len(CASES)} Baseline acceptance scenarios passed.")


if __name__ == "__main__":
    main()
