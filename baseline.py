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


class BaselineSequencePlan:
    """A validated, reusable H-only sequence for the unfused reference.

    Build once and call run_inplace many times without recreating the target
    array. It still validates each *state* before an in-place operation.
    The plan is optional; existing Baseline entry points remain supported.
    """

    def __init__(self, n_qubits: int, targets: tuple[int, ...] | list[int]) -> None:
        if type(n_qubits) is not int:
            raise TypeError("n_qubits must be a Python int (not bool)")
        if n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")
        if not isinstance(targets, (list, tuple)):
            raise TypeError("targets must be a list or tuple of Python integer indices")
        checked = []
        for target in targets:
            if type(target) is not int:
                raise TypeError("every target must be a Python int (not bool)")
            if not 0 <= target < n_qubits:
                raise ValueError(f"target {target} must satisfy 0 <= target < {n_qubits}")
            checked.append(target)

        # Snapshot the caller's list, so subsequent edits cannot change the plan.
        self.n_qubits = n_qubits
        self.targets = tuple(checked)
        self._target_array = np.asarray(self.targets, dtype=np.int64)
        self._target_array.setflags(write=False)

    def run_inplace(self, state: np.ndarray) -> None:
        """Perform separate H passes; never fuse adjacent or repeating gates."""
        validate_state_vector(state)
        if state.size != (1 << self.n_qubits):
            raise ValueError(
                f"state length must be 2**{self.n_qubits} for this prepared plan"
            )
        if self.targets:
            _baseline_sequence_kernel(state, self._target_array)


def prepare_baseline_sequence(
    n_qubits: int, targets: tuple[int, ...] | list[int]
) -> BaselineSequencePlan:
    """Validate and prepare an H-only Baseline sequence for repeated execution."""
    return BaselineSequencePlan(n_qubits, targets)


def baseline_access_profile(
    n_qubits: int, targets: tuple[int, ...] | list[int]
) -> dict:
    """Describe algorithm-level access patterns for each *unfused* H gate.

    Sample index pairs reflect the exact baseline loop ordering. Pair strides
    are array index differences, not observed cache/DRAM transfer counts.
    """
    plan = prepare_baseline_sequence(n_qubits, targets)
    n_amplitudes = 1 << n_qubits
    bytes_per_amplitude = np.dtype(np.complex128).itemsize
    per_gate = []

    for gate_index, target in enumerate(plan.targets):
        stride = 1 << target
        span = stride << 1
        # The same formula as the baseline's nested loops, but only for
        # the first few pairs so no large trace is materialized.
        samples = []
        for pair_number in range(min(4, n_amplitudes // 2)):
            block, offset = divmod(pair_number, stride)
            left = block * span + offset
            samples.append((left, left + stride))

        per_gate.append({
            "gate_index": gate_index,
            "target": target,
            "pair_stride_amplitudes": stride,
            "pair_stride_bytes": stride * bytes_per_amplitude,
            "span_amplitudes": span,
            "pairs_processed": n_amplitudes // 2,
            "logical_read_bytes": n_amplitudes * bytes_per_amplitude,
            "logical_write_bytes": n_amplitudes * bytes_per_amplitude,
            "sample_index_pairs": samples,
        })

    return {
        "n_qubits": n_qubits,
        "n_amplitudes": n_amplitudes,
        "state_bytes": n_amplitudes * bytes_per_amplitude,
        "gate_targets": plan.targets,
        "per_gate": per_gate,
        "totals": baseline_logical_memory_accesses(n_qubits, len(plan.targets)),
    }


def baseline_logical_access_trace(
    n_qubits: int,
    targets: tuple[int, ...] | list[int],
    max_pairs: int = 16,
) -> dict:
    """Bounded prefix of the *algorithmic* read/write order, not a CPU trace.

    Each pair describes two amplitude loads followed by two amplitude stores
    in the nested-loop order of _apply_h_inplace. Actual compiler, cache,
    prefetcher, and DRAM behavior are intentionally not modeled here.

    Unlike a full trace, memory use is O(max_pairs); safe for large n.
    """
    plan = prepare_baseline_sequence(n_qubits, targets)
    if type(max_pairs) is not int:
        raise TypeError("max_pairs must be a Python int (not bool)")
    if not 0 <= max_pairs <= 4096:
        raise ValueError("max_pairs must be between 0 and 4096")
    amplitude_pairs_per_gate = (1 << n_qubits) // 2
    total_pairs = amplitude_pairs_per_gate * len(plan.targets)
    count = min(max_pairs, total_pairs)
    trace = []
    for sequence_pair in range(count):
        gate_index, local_pair = divmod(sequence_pair, amplitude_pairs_per_gate)
        target = plan.targets[gate_index]
        stride = 1 << target
        block, offset = divmod(local_pair, stride)
        i = block * (2 * stride) + offset
        j = i + stride
        trace.append({
            "gate_index": gate_index,
            "target": target,
            "pair_index_in_gate": local_pair,
            "read_indices": (i, j),
            "write_indices": (i, j),
            "read_byte_offsets": (16 * i, 16 * j),
            "write_byte_offsets": (16 * i, 16 * j),
        })
    return {
        "n_qubits": n_qubits,
        "targets": plan.targets,
        "total_pairs": total_pairs,
        "sampled_pairs": count,
        "truncated": count < total_pairs,
        "trace": trace,
    }


def baseline_cache_line_pair_profile(
    n_qubits: int,
    targets: tuple[int, ...] | list[int],
    line_bytes: int = 64,
    base_offset_bytes: int = 0,
) -> dict:
    """Exact *address geometry* of H-gate amplitude pairs under a line model.

    Only answers whether the two amplitudes in a pair lie in the same
    hypothetical cache line, given configurable line size and array alignment.
    Does NOT estimate cache misses, DRAM transactions, or execution time.

    Uses periodic counting rather than materializing 2**n amplitudes.
    Assumes 16-byte complex128 amplitudes and power-of-two line sizes.
    """
    plan = prepare_baseline_sequence(n_qubits, targets)
    if type(line_bytes) is not int or line_bytes < 32 or line_bytes & (line_bytes - 1):
        raise ValueError("line_bytes must be a power-of-two Python int >= 32")
    if type(base_offset_bytes) is not int:
        raise TypeError("base_offset_bytes must be a Python int (not bool)")
    if not (0 <= base_offset_bytes < line_bytes and base_offset_bytes % 16 == 0):
        raise ValueError("base_offset_bytes must be aligned to 16 bytes and < line_bytes")

    n_amplitudes = 1 << n_qubits
    n_pairs = n_amplitudes // 2
    state_bytes = n_amplitudes * 16
    distinct_lines = (base_offset_bytes + state_bytes - 1) // line_bytes + 1
    per_gate = []

    for gate_index, target in enumerate(plan.targets):
        stride = 1 << target
        stride_bytes = 16 * stride
        if stride_bytes >= line_bytes:
            same = 0
        else:
            # For power-of-two stride and line size, pair-line membership
            # repeats every line_bytes // 32 consecutive pair iterations.
            period_pairs = line_bytes // 32
            def same_line(pair_number: int) -> bool:
                block, offset = divmod(pair_number, stride)
                left_index = block * 2 * stride + offset
                left_byte = base_offset_bytes + 16 * left_index
                right_byte = left_byte + stride_bytes
                return left_byte // line_bytes == right_byte // line_bytes

            full_period_same = sum(same_line(p) for p in range(period_pairs))
            cycles, remainder = divmod(n_pairs, period_pairs)
            same = cycles * full_period_same
            same += sum(same_line(p) for p in range(remainder))

        per_gate.append({
            "gate_index": gate_index,
            "target": target,
            "pair_stride_bytes": stride_bytes,
            "pair_count": n_pairs,
            "same_line_pairs": same,
            "different_line_pairs": n_pairs - same,
            "different_line_fraction": (n_pairs - same) / n_pairs,
            "distinct_state_lines_touched": distinct_lines,
        })

    return {
        "n_qubits": n_qubits,
        "targets": plan.targets,
        "cache_line_bytes_assumed": line_bytes,
        "array_base_offset_bytes_assumed": base_offset_bytes,
        "per_gate": per_gate,
        "note": "Analytical cache-line address geometry; NOT hardware cache misses.",
    }
