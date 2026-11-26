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
    所有性能对比实验的唯一入口。
    计时边界铁律:
    1. 输入数组复制/构造必须在 timer.start() 之前完成。
    2. 预热循环在计时循环之前，预热数据不得进入统计。
    3. kernel 模式下，参数校验与结果封装在计时区间外。
    4. public 模式下，参数校验在计时区间内，但输入复制仍在计时区间外。
    5. 计时使用 time.perf_counter_ns()。
    6. CSV 写入与统计汇总在全部计时循环结束后统一执行。
    """
    raise NotImplementedError("E 模块的 benchmark.py 负责实现此接口。")
