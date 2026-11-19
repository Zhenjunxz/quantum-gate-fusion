"""Independent structural and physical-invariant checks for A's Baseline.

Tests the H|0>, H|1> equation directly and proves by enumeration that each
single-H sweep pairs each state-vector index exactly once. No Fusion or
runtime measurement is involved.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from baseline import (
    baseline_logical_access_trace,
    baseline_logical_memory_accesses,
    run_baseline_sequence_inplace,
)
from common import random_state
from verify_baseline_a import CASES, main


@pytest.mark.parametrize("n", range(2, 9))
def test_all_qubit_positions_on_representative_basis_states(n: int) -> None:
    dimension = 1 << n
    inv_sqrt2 = 1 / np.sqrt(2.0)
    for q in range(n):
        bit = 1 << q
        basis_indices = {0, dimension - 1, bit, dimension - 1 - bit}
        for basis in basis_indices:
            actual = np.zeros(dimension, dtype=np.complex128)
            actual[basis] = 1.0
            expected = np.zeros_like(actual)
            zero = basis & ~bit
            one = zero | bit
            expected[zero] = inv_sqrt2
            expected[one] = (-inv_sqrt2 if basis & bit else inv_sqrt2)

            run_baseline_sequence_inplace(actual, [q])
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("n", range(2, 9))
def test_single_h_sweep_exactly_partitions_all_amplitudes(n: int) -> None:
    dimension = 1 << n
    for q in range(n):
        bit = 1 << q
        trace = baseline_logical_access_trace(n, [q], max_pairs=dimension // 2)
        assert not trace["truncated"]
        reads = []
        writes = []
        for event in trace["trace"]:
            i, j = event["read_indices"]
            assert i & bit == 0
            assert j == i | bit
            assert (i ^ j) == bit
            assert event["write_indices"] == (i, j)
            reads.extend((i, j))
            writes.extend(event["write_indices"])
        assert sorted(reads) == list(range(dimension))
        assert sorted(writes) == list(range(dimension))
        model = baseline_logical_memory_accesses(n, 1)
        assert model["amplitude_reads"] == dimension
        assert model["amplitude_writes"] == dimension


def test_h_on_distinct_qubits_commutes_and_preserves_norm() -> None:
    source = random_state(9, seed=78)
    left = source.copy()
    right = source.copy()
    run_baseline_sequence_inplace(left, [0, 8, 4])
    run_baseline_sequence_inplace(right, [4, 0, 8])
    np.testing.assert_allclose(left, right, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(left), np.linalg.norm(source), atol=1e-12)


@pytest.mark.parametrize("q", [0, 2, 5])
def test_h_squared_is_identity_with_no_gate_elimination(q: int) -> None:
    source = random_state(6, seed=2026 + q)
    actual = source.copy()
    run_baseline_sequence_inplace(actual, [q, q])
    np.testing.assert_allclose(actual, source, atol=1e-12, rtol=1e-12)
    assert baseline_logical_memory_accesses(6, 2)["amplitude_reads"] == 128


def test_json_acceptance_report_is_valid_and_not_performance_data(
    monkeypatch, tmp_path, capsys
) -> None:
    destination = tmp_path / "baseline_a" / "acceptance.json"
    monkeypatch.setattr(sys, "argv", ["verify_baseline_a.py", "--output", str(destination)])
    main()
    output = capsys.readouterr().out
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert "All 7 Baseline acceptance scenarios passed." in output
    assert report["status"] == "passed"
    assert report["schema_version"] == 1
    assert report["case_count"] == len(CASES) == len(report["cases"])
    assert report["cases"][0]["targets"] == [0, 1]
    assert "python" in report["environment"]
    assert "numba" in report["environment"]
    assert "not" in report["warning"].lower()
    assert all(entry["max_absolute_error"] < 1e-12 for entry in report["cases"])


def test_json_report_default_cli_does_not_write_files(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify_baseline_a.py"])
    main()
    assert "All 7 Baseline acceptance scenarios passed." in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
