"""Small, independently checkable cases for the idealized Baseline LRU model."""
from __future__ import annotations

import pytest

from baseline_cache_model import simulate_baseline_ideal_lru as simulate


def test_one_line_covers_full_2_qubit_state_and_stays_warm_between_gates():
    data = simulate(2, [0, 1], 64, line_bytes=64)
    assert data["logical_events"] == 16
    assert data["totals"]["read_misses"] == 1
    assert data["totals"]["read_hits"] == 7
    assert data["totals"]["write_misses"] == 0
    assert data["totals"]["write_hits"] == 8
    assert data["per_gate"][0]["read_misses"] == 1
    assert data["per_gate"][1]["read_misses"] == 0
    assert data["totals"]["dirty_evictions"] == 0
    assert data["final_dirty_resident_lines"] == 1


def test_gate_position_affects_ideal_lru_if_capacity_is_small():
    # 3 qubits = 8 amplitudes, spanning 2 x 64-byte cache lines.
    # H0 keeps each amplitude pair on one line, but H2 alternates lines.
    contiguous = simulate(3, [0], 64, line_bytes=64)
    scattered = simulate(3, [2], 64, line_bytes=64)
    assert contiguous["totals"]["read_misses"] == 2
    assert contiguous["totals"]["write_misses"] == 0
    assert scattered["totals"]["read_misses"] == 8
    assert scattered["totals"]["write_misses"] == 8
    assert scattered["totals"]["dirty_evictions"] > 0

    # With both lines resident, both traversal orders require 2 cold line fills.
    for target in (0, 2):
        full_capacity = simulate(3, [target], 128, line_bytes=64)
        assert full_capacity["totals"]["read_misses"] == 2
        assert full_capacity["totals"]["write_misses"] == 0
        assert full_capacity["totals"]["evictions"] == 0


def test_repeated_gate_is_not_cancelled_and_cache_state_persists():
    result = simulate(3, [0, 0], 128, line_bytes=64)
    assert result["logical_events"] == 32
    assert len(result["per_gate"]) == 2
    assert result["per_gate"][0]["read_misses"] == 2
    assert result["per_gate"][1]["read_misses"] == 0


def test_empty_sequence_has_zero_events():
    data = simulate(20, [], 64)
    assert data["logical_events"] == 0
    assert data["per_gate"] == []
    assert all(value == 0 for value in data["totals"].values())
    assert data["final_resident_lines"] == 0


def test_alignment_can_change_modelled_accesses():
    first = simulate(3, [0], 64, line_bytes=64, base_offset_bytes=0)
    offset = simulate(3, [0], 64, line_bytes=64, base_offset_bytes=16)
    assert first["totals"]["read_misses"] == 2
    assert offset["totals"]["read_misses"] >= first["totals"]["read_misses"]


@pytest.mark.parametrize("params,expected_exception", [
    ({"line_bytes": 48}, ValueError),
    ({"line_bytes": 16}, ValueError),
    ({"line_bytes": True}, ValueError),
    ({"cache_capacity_bytes": 32}, ValueError),
    ({"cache_capacity_bytes": 96}, ValueError),
    ({"cache_capacity_bytes": False}, TypeError),
    ({"base_offset_bytes": 13}, ValueError),
    ({"base_offset_bytes": -16}, ValueError),
    ({"base_offset_bytes": False}, TypeError),
    ({"max_events": True}, TypeError),
    ({"max_events": -1}, ValueError),
])
def test_invalid_cache_model_parameters(params, expected_exception):
    arguments = {"cache_capacity_bytes": 64}
    arguments.update(params)
    with pytest.raises(expected_exception):
        simulate(3, [0], **arguments)


def test_oversized_job_refused_before_simulating():
    with pytest.raises(ValueError, match="exceeds"):
        simulate(20, [0, 5], 64, max_events=10000)


@pytest.mark.parametrize("n,targets,lines", [
    (2, (0,), 1),
    (3, (0, 2), 1),
    (4, (1, 3), 2),
    (5, (0, 4, 2), 4),
    (6, (5, 0, 3, 5), 8),
])
def test_read_write_accounting_and_eviction_invariants(n, targets, lines):
    profile = simulate(n, targets, cache_capacity_bytes=64 * lines)
    total = profile["totals"]
    assert total["read_hits"] + total["read_misses"] == (1 << n) * len(targets)
    assert total["write_hits"] + total["write_misses"] == (1 << n) * len(targets)
    assert total["evictions"] >= total["dirty_evictions"] >= 0
    assert profile["final_resident_lines"] <= lines
    for gate in profile["per_gate"]:
        assert gate["read_hits"] + gate["read_misses"] == 1 << n
        assert gate["write_hits"] + gate["write_misses"] == 1 << n
