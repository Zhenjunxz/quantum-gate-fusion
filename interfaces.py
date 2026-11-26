"""
C 模块：统一接口定义与基准测试规范。
所有模块（baseline / fusion / benchmark）必须遵守此处的类型别名、函数签名与计时边界约定。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Tuple
import numpy as np

# ---------- 统一类型别名 ----------
Gate = Tuple[str, List[int]]                # 单个门，如 ("H", [0])
GateSequence = List[Gate]                   # 门序列，如 [("H", [0]), ("H", [1])]
FusionGroups = List[List[int]]              # 融合分组，如 [[0, 1]] 表示第 0、1 个门融合

# ---------- 统一结果结构 ----------
@dataclass
class BenchmarkResult:
    backend_name: str
    n_qubits: int
    timing_scope: Literal["kernel", "public"]
    mean_ms: float
    std_ms: float
    median_ms: float
    max_abs_error: float

# ---------- 统一 Benchmark 入口签名 ----------
def run_benchmark(
    backend_fn: Callable[[np.ndarray], None],
    n_qubits: int,
    gate_sequence: GateSequence,
    fusion_groups: FusionGroups,
    timing_scope: Literal["kernel", "public"] = "kernel",
    warmup_iterations: int = 10,
    repetitions: int = 100,
) -> BenchmarkResult:
    """
    单 backend 的统一 Python API；成对实验通过共享计时原语交替执行。
    计时边界铁律:
    1. 输入数组复制/构造必须在 timer.start() 之前完成。
    2. 预热循环在计时循环之前，预热数据不得进入统计。
    3. kernel 模式下，参数校验与结果封装在计时区间外。
    4. public 模式下，参数校验在计时区间内，但输入复制仍在计时区间外。
    5. 计时使用 time.perf_counter_ns()。
    6. CSV 写入与统计汇总在全部计时循环结束后统一执行。
    """
    # Lazy import avoids a circular dependency with benchmark.py.
    from benchmark import run_benchmark as implementation
    return implementation(backend_fn, n_qubits, gate_sequence, fusion_groups,
                          timing_scope, warmup_iterations, repetitions)


def validate_case(n_qubits, gate_sequence, fusion_groups) -> dict:
    """Normalize C's canonical two-H contract; also accept E's legacy scalar targets.

    This project's Fusion implementation supports two distinct H gates only.
    Reject unsupported circuits/groups rather than silently changing the task.
    """
    if type(n_qubits) is not int or not 2 <= n_qubits <= 24:
        raise ValueError("n_qubits must be a Python int between 2 and 24")
    if not isinstance(gate_sequence, (list, tuple)) or len(gate_sequence) != 2:
        raise ValueError("gate_sequence must contain exactly two H gates")
    targets = []
    for gate in gate_sequence:
        if (not isinstance(gate, (list, tuple)) or len(gate) != 2
                or not isinstance(gate[0], str) or gate[0] != "H"):
            raise ValueError("Each gate must be ('H', [target])")
        target = gate[1]
        if isinstance(target, (list, tuple)):
            if len(target) != 1:
                raise ValueError("Each H gate must have exactly one target")
            target = target[0]
        if type(target) is not int or not 0 <= target < n_qubits:
            raise ValueError("Gate targets must be Python ints in [0, n_qubits)")
        targets.append(target)
    if targets[0] == targets[1]:
        raise ValueError("The two target qubits must be distinct")
    if (not isinstance(fusion_groups, (list, tuple)) or len(fusion_groups) != 1
            or not isinstance(fusion_groups[0], (list, tuple))
            or len(fusion_groups[0]) != 2
            or any(type(i) is not int for i in fusion_groups[0])
            or list(fusion_groups[0]) != [0, 1]):
        raise ValueError("fusion_groups must be [[0, 1]], indexing the two gates")
    return {"n_qubits": n_qubits, "targets": tuple(targets),
            "gate_sequence": [("H", [q]) for q in targets], "fusion_groups": [[0, 1]]}

