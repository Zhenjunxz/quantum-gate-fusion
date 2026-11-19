"""Independent explicit reference for Baseline's hypothetical set-associative LRU.

The reference enumerates individual address events via *bit tests*, not
Baseline's base/offset loops, and tracks each set with a simple Python list
rather than the implementation's OrderedDict. Small exhaustive combinations
are intentional; no real CPU timing is inferred.
"""
from __future__ import annotations

import itertools

import pytest

from baseline_cache_model import simulate_baseline_ideal_lru


COUNTERS = (
    "read_hits", "read_misses", "write_hits", "write_misses",
    "evictions", "dirty_evictions",
)


def independent_cache_reference(n, targets, capacity, line_bytes, ways, alignment):
    """Simple list-based oracle, independent from the implementation's LRU."""
    set_count = capacity // line_bytes // ways
    resident = [[] for _ in range(set_count)]
    totals = dict.fromkeys(COUNTERS, 0)
    per_gate = []

    for gate_index, target in enumerate(targets):
        counts = {"gate_index": gate_index, "target": target, **dict.fromkeys(COUNTERS, 0)}
        bit = 1 << target
        for i in range(1 << n):
            if i & bit:
                continue
            j = i + bit
            for amplitude, is_write in (
                (i, False), (j, False), (i, True), (j, True)
            ):
                line = (alignment + 16 * amplitude) // line_bytes
                set_index = line % set_count
                contents = resident[set_index]  # least-recent -> most-recent
                matches = [pos for pos, (tag, _) in enumerate(contents) if tag == line]
                if matches:
                    index = matches[0]
                    _, dirty = contents.pop(index)
                    contents.append((line, dirty or is_write))
                    key = "write_hits" if is_write else "read_hits"
                else:
                    key = "write_misses" if is_write else "read_misses"
                    if len(contents) == ways:
                        _, dirty = contents.pop(0)
                        counts["evictions"] += 1
                        totals["evictions"] += 1
                        if dirty:
                            counts["dirty_evictions"] += 1
                            totals["dirty_evictions"] += 1
                    contents.append((line, is_write))
                counts[key] += 1
                totals[key] += 1
        per_gate.append(counts)

    return totals, per_gate, sum(len(s) for s in resident), sum(
        sum(int(dirty) for _, dirty in s) for s in resident
    )


@pytest.mark.parametrize("n,targets,line,capacity,ways,alignment", [
    (n, targets, line, capacity, ways, alignment)
    for n in (2, 3, 4, 5)
    for targets in ((0,), (n - 1,), (0, n - 1), (n - 1, 0, n - 1))
    for line in (32, 64, 128)
    for capacity in (line, 2 * line, 4 * line)
    for ways in (1, 2, 4)
    if ways <= capacity // line and (capacity // line) % ways == 0
    for alignment in (0, 16)
])
def test_set_associative_vs_independent_reference(
    n, targets, line, capacity, ways, alignment
):
    expected_total, expected_gates, expected_resident, expected_dirty = (
        independent_cache_reference(n, targets, capacity, line, ways, alignment)
    )
    actual = simulate_baseline_ideal_lru(
        n,
        targets,
        capacity,
        line_bytes=line,
        associativity_ways=ways,
        base_offset_bytes=alignment,
        max_events=100_000,
    )
    assert actual["totals"] == expected_total
    assert actual["per_gate"] == expected_gates
    assert actual["final_resident_lines"] == expected_resident
    assert actual["final_dirty_resident_lines"] == expected_dirty
    assert actual["set_count_assumed"] == capacity // line // ways
    assert actual["associativity_ways_assumed"] == ways


def test_direct_mapped_conflict_vs_fully_associative_same_capacity():
    # Four 64 B lines, cache capacity two lines. H3 pairs lines 0/2 and 1/3.
    # Under two-set direct mapping, 0 and 2 conflict; two-way fully
    # associative can retain the paired lines simultaneously.
    direct = simulate_baseline_ideal_lru(
        4, [3], 128, line_bytes=64, associativity_ways=1
    )
    fully = simulate_baseline_ideal_lru(
        4, [3], 128, line_bytes=64, associativity_ways=2
    )
    assert direct["totals"]["read_misses"] == 16
    assert direct["totals"]["write_misses"] == 16
    assert fully["totals"]["read_misses"] == 4
    assert fully["totals"]["write_misses"] == 0
    assert direct["set_count_assumed"] == 2
    assert fully["set_count_assumed"] == 1
    assert direct["logical_events"] == fully["logical_events"] == 32


def test_omitted_associativity_preserves_old_fully_associative_results():
    for n, targets, cache, line in (
        (3, (0, 2), 128, 64),
        (5, (4, 1, 3), 256, 64),
        (4, (0, 3, 0), 128, 32),
    ):
        default = simulate_baseline_ideal_lru(n, targets, cache, line_bytes=line)
        explicitly_full = simulate_baseline_ideal_lru(
            n,
            targets,
            cache,
            line_bytes=line,
            associativity_ways=cache // line,
        )
        assert default == explicitly_full
        assert default["set_count_assumed"] == 1


@pytest.mark.parametrize("ways,error", [
    (0, ValueError), (-1, ValueError), (3, ValueError),
    (5, ValueError), (True, TypeError), (1.0, TypeError),
])
def test_invalid_associativity_rejected(ways, error):
    with pytest.raises(error):
        simulate_baseline_ideal_lru(
            4, [0], 128, line_bytes=64, associativity_ways=ways
        )


def test_empty_sequence_associativity_works_without_any_events():
    output = simulate_baseline_ideal_lru(20, [], 128, associativity_ways=1)
    assert all(v == 0 for v in output["totals"].values())
    assert output["set_count_assumed"] == 2
    assert output["final_resident_lines"] == 0


def test_oversized_request_remains_bounded_with_associativity():
    with pytest.raises(ValueError, match="exceeds"):
        simulate_baseline_ideal_lru(
            20, [0, 5], 128, associativity_ways=1, max_events=1000
        )
