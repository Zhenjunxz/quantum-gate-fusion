"""Idealized LRU model for the *unfused* H-gate Baseline's logical accesses.

This is a teaching/analysis model, NOT CPU hardware-counter measurement.
It assumes a fully associative, write-back, write-allocate LRU cache with
configurable line size and capacity, initially empty. It follows the source
loop's logical load/load/store/store pair order, *not* an optimized JIT
machine-code trace. No prefetching, vectorization, coherence, or multilevel
cache hierarchy is represented.
"""

from __future__ import annotations

from collections import OrderedDict

from baseline import prepare_baseline_sequence


def simulate_baseline_ideal_lru(
    n_qubits: int,
    targets: tuple[int, ...] | list[int],
    cache_capacity_bytes: int,
    *,
    line_bytes: int = 64,
    base_offset_bytes: int = 0,
    max_events: int = 1_000_000,
) -> dict:
    """Simulate a bounded logical Baseline sequence in an ideal LRU cache.

    Each H pair first reads amplitudes i and j, then writes both. Each
    complex128 amplitude occupies 16 bytes. Cache starts cold and persists
    across H-gate passes. Results are *model events*, not PMU cache misses.

    The number of modeled memory events is 2 * 2**n * len(targets).
    Refuse an oversized workload before traversing it; use
    baseline_cache_line_pair_profile for very large n instead.
    """
    plan = prepare_baseline_sequence(n_qubits, targets)

    if type(line_bytes) is not int or line_bytes < 32 or (line_bytes & (line_bytes - 1)):
        raise ValueError("line_bytes must be a power-of-two Python int >= 32")
    if type(cache_capacity_bytes) is not int:
        raise TypeError("cache_capacity_bytes must be a Python int (not bool)")
    if cache_capacity_bytes < line_bytes or cache_capacity_bytes % line_bytes:
        raise ValueError("cache_capacity_bytes must be >= line_bytes and a multiple of it")
    if type(base_offset_bytes) is not int:
        raise TypeError("base_offset_bytes must be a Python int (not bool)")
    if not (0 <= base_offset_bytes < line_bytes and base_offset_bytes % 16 == 0):
        raise ValueError("base_offset_bytes must be aligned to 16 bytes and < line_bytes")
    if type(max_events) is not int:
        raise TypeError("max_events must be a Python int (not bool)")
    if not 1 <= max_events <= 10_000_000:
        raise ValueError("max_events must satisfy 1 <= max_events <= 10000000")

    n_amplitudes = 1 << n_qubits
    total_events = 2 * n_amplitudes * len(plan.targets)
    if total_events > max_events:
        raise ValueError(
            f"requested {total_events} logical events exceeds max_events={max_events}; "
            "use baseline_cache_line_pair_profile for large states"
        )

    cache_lines = cache_capacity_bytes // line_bytes
    # OrderedDict maps cache-line number -> dirty flag, in LRU -> MRU order.
    resident: OrderedDict[int, bool] = OrderedDict()

    counters = (
        "read_hits", "read_misses", "write_hits", "write_misses",
        "evictions", "dirty_evictions",
    )
    totals = dict.fromkeys(counters, 0)
    per_gate = []

    def access(amplitude_index: int, is_write: bool, gate: dict[str, int]) -> None:
        line = (base_offset_bytes + 16 * amplitude_index) // line_bytes
        if line in resident:
            was_dirty = resident.pop(line)
            resident[line] = was_dirty or is_write
            key = "write_hits" if is_write else "read_hits"
        else:
            key = "write_misses" if is_write else "read_misses"
            if len(resident) == cache_lines:
                _old_line, was_dirty = resident.popitem(last=False)
                gate["evictions"] += 1
                totals["evictions"] += 1
                if was_dirty:
                    gate["dirty_evictions"] += 1
                    totals["dirty_evictions"] += 1
            # A write miss allocates a line; any write marks it dirty.
            resident[line] = is_write
        gate[key] += 1
        totals[key] += 1

    for gate_index, target in enumerate(plan.targets):
        gate = {
            "gate_index": gate_index,
            "target": target,
            **dict.fromkeys(counters, 0),
        }
        stride = 1 << target
        span = stride << 1
        for base in range(0, n_amplitudes, span):
            for offset in range(stride):
                i, j = base + offset, base + offset + stride
                access(i, False, gate)
                access(j, False, gate)
                access(i, True, gate)
                access(j, True, gate)
        per_gate.append(gate)

    assert totals["read_hits"] + totals["read_misses"] == n_amplitudes * len(plan.targets)
    assert totals["write_hits"] + totals["write_misses"] == n_amplitudes * len(plan.targets)

    return {
        "model": "fully_associative_LRU_cold_write_back_write_allocate",
        "n_qubits": n_qubits,
        "targets": plan.targets,
        "line_bytes_assumed": line_bytes,
        "capacity_bytes_assumed": cache_capacity_bytes,
        "base_offset_bytes_assumed": base_offset_bytes,
        "logical_events": total_events,
        "per_gate": per_gate,
        "totals": totals,
        "final_dirty_resident_lines": sum(resident.values()),
        "final_resident_lines": len(resident),
        "disclaimer": (
            "Model-only logical cache events; NOT CPU PMU misses, "
            "DRAM traffic, or benchmark measurements."
        ),
    }
