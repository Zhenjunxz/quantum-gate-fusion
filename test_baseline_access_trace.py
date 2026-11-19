"""Compare analytical Baseline access/line profiles with brute-force loops."""
from __future__ import annotations

import itertools
import pytest

from baseline import (
    baseline_logical_access_trace,
    baseline_cache_line_pair_profile,
)


def reference_pairs(n, target):
    stride = 1 << target
    for base in range(0, 1 << n, 2 * stride):
        for offset in range(stride):
            yield (base + offset, base + offset + stride)


def reference_same_line(n, target, line, alignment):
    return sum(
        (alignment + 16 * i) // line == (alignment + 16 * j) // line
        for i, j in reference_pairs(n, target)
    )


@pytest.mark.parametrize("n,targets", [
    (2, (0,)), (3, (1, 2)), (4, (2, 0, 3)), (5, ()),
    (6, (0, 5, 2)), (8, (0, 0)),
])
def test_access_trace_matches_exact_kernel_pair_order(n, targets):
    expected = []
    for gate_index, target in enumerate(targets):
        for pair_index, (i, j) in enumerate(reference_pairs(n, target)):
            expected.append((gate_index, target, pair_index, (i, j)))
    actual = baseline_logical_access_trace(n, targets, max_pairs=min(256, len(expected)))
    assert actual["total_pairs"] == len(expected)
    assert actual["sampled_pairs"] == min(256, len(expected))
    assert actual["truncated"] is (actual["sampled_pairs"] < len(expected))
    for event, (gate_index, target, pair_index, pair) in zip(actual["trace"], expected):
        assert (event["gate_index"], event["target"], event["pair_index_in_gate"]) == (
            gate_index, target, pair_index
        )
        assert event["read_indices"] == pair
        assert event["write_indices"] == pair
        assert event["read_byte_offsets"] == tuple(16 * k for k in pair)
        assert event["write_byte_offsets"] == tuple(16 * k for k in pair)


def test_trace_large_n_is_bounded_and_has_correct_prefix():
    result = baseline_logical_access_trace(24, [0, 10], max_pairs=3)
    assert len(result["trace"]) == 3
    assert result["total_pairs"] == 1 << 24
    assert [entry["read_indices"] for entry in result["trace"]] == [
        (0, 1), (2, 3), (4, 5),
    ]


@pytest.mark.parametrize("n,line", itertools.product([2, 3, 4, 5, 8], [32, 64, 128]))
@pytest.mark.parametrize("alignment_fraction", [0, 1, 2, 3])
def test_line_pair_count_matches_bruteforce(n, line, alignment_fraction):
    offset = (alignment_fraction * 16) % line
    targets = list(range(n))
    report = baseline_cache_line_pair_profile(n, targets, line, offset)
    for entry in report["per_gate"]:
        same = reference_same_line(n, entry["target"], line, offset)
        assert entry["same_line_pairs"] == same
        assert entry["different_line_pairs"] == (1 << (n - 1)) - same
        assert entry["same_line_pairs"] + entry["different_line_pairs"] == entry["pair_count"]
        expected_lines = (offset + 16 * (1 << n) - 1) // line + 1
        assert entry["distinct_state_lines_touched"] == expected_lines


def test_same_footprint_different_pair_geometry():
    report = baseline_cache_line_pair_profile(20, [0, 5, 10], line_bytes=64, base_offset_bytes=0)
    records = report["per_gate"]
    assert [x["same_line_pairs"] for x in records] == [1 << 19, 0, 0]
    assert [x["distinct_state_lines_touched"] for x in records] == [1 << 18] * 3
    assert report["note"].endswith("NOT hardware cache misses.")


def test_base_alignment_changes_geometry_not_logical_bytes():
    aligned = baseline_cache_line_pair_profile(8, [0], 64, 0)["per_gate"][0]
    shifted = baseline_cache_line_pair_profile(8, [0], 64, 16)["per_gate"][0]
    assert aligned["same_line_pairs"] == 128
    assert shifted["same_line_pairs"] == 64


@pytest.mark.parametrize("kwargs,error", [
    ({"line_bytes": 48}, ValueError),
    ({"line_bytes": 16}, ValueError),
    ({"line_bytes": True}, ValueError),
    ({"base_offset_bytes": -16}, ValueError),
    ({"base_offset_bytes": 15}, ValueError),
    ({"base_offset_bytes": 64}, ValueError),
    ({"base_offset_bytes": False}, TypeError),
])
def test_line_model_rejects_unsupported_configs(kwargs, error):
    with pytest.raises(error):
        baseline_cache_line_pair_profile(4, [0, 1], **kwargs)


@pytest.mark.parametrize("limit,error", [
    (-1, ValueError), (4097, ValueError), (True, TypeError),
])
def test_trace_limit_guard(limit, error):
    with pytest.raises(error):
        baseline_logical_access_trace(4, [0], max_pairs=limit)


def test_empty_trace_and_zero_limit():
    result = baseline_logical_access_trace(4, [], max_pairs=16)
    assert result["trace"] == [] and not result["truncated"]
    zero = baseline_logical_access_trace(4, [0], max_pairs=0)
    assert zero["trace"] == [] and zero["truncated"]
