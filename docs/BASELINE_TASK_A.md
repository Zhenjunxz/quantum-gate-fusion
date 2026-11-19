# 代码任务 A：Baseline（逐门执行）

当前 Baseline 只实现 **Hadamard (H) 门**；它为 Gate Fusion 对照实验提供不融合的参考计算，不开发通用量子模拟器。

## 1. 现有接口（保持兼容）

```python
from baseline import run_baseline_inplace
run_baseline_inplace(state)                 # 原有 H0 -> H1
run_baseline_inplace(state, q1=0, q2=5)   # 两个不同目标位置
```

`q1/q2` 仍是小组 [Issue #3](https://github.com/Zhenjunxz/quantum-gate-fusion/issues/3) 的**待确认接口**；本次修改不把它认定为最终小组统一接口。

## 2. A 模块新增：受限 H 门序列接口

```python
from baseline import run_baseline_sequence_inplace
run_baseline_sequence_inplace(state, targets=[0, 1])
run_baseline_sequence_inplace(state, targets=[0, 5])
run_baseline_sequence_inplace(state, targets=[0, 1, 2, 3])
run_baseline_sequence_inplace(state, targets=[0, 0])  # H0 再 H0：逐门执行两遍
```

- `state` 必须是可写、连续的一维 `np.complex128` 数组，长度为 `2**n` 且 `n >= 2`。
- `targets` 是由 **Python int** 组成的 `list` 或 `tuple`，每项都满足 `0 <= target < n`；参数会在原地修改前完整验证。空序列相当于恒等变换。
- 允许重复同一目标（如 `H0,H0`），因为 Baseline 不能悄悄删除门：虽然数学上 `H²=I`，但实验中若删除门，访存任务已经改变。
- qubit 0 对应 state 索引的最低有效位。单门配对振幅下标相差 `2**target`。
- 每个门都通过 `_apply_h_inplace` 独立遍历并写回**整个状态数组**，不实施 Gate Fusion。
- 本接口只属于 A 模块草案；它**不是** B 模块必须立即支持的接口。若今后要做融合规模实验，应由 A/B/C 共同决定门序列的数据格式和合法融合分组。

### Python 包装层与数值内核

`run_baseline_sequence_inplace` 会校验输入并把门位置列表转成 `int64` 数组，随后调用 Numba 的 `_baseline_sequence_kernel`。转换和输入检查位于编译内核之外；负责 Benchmark 的组员应统一定义计时区间，不能随意拿两种不同的计时边界对比。

## 3. A 模块新增：理论逻辑读写量

```python
from baseline import baseline_logical_memory_accesses
baseline_logical_memory_accesses(n_qubits=20, gate_count=4)
```

长度为 `N=2**n` 的状态向量，每个振幅使用 `complex128`（16 字节）。Baseline 对每个 H 门完整遍历一次：

- 逻辑振幅读取总次数：`gate_count * N`
- 逻辑振幅写回总次数：`gate_count * N`
- 理论逻辑总读写字节：`2 * gate_count * N * 16`

函数返回 `amplitude_reads`、`amplitude_writes`、`logical_read_bytes`、`logical_write_bytes` 和 `total_logical_bytes`。**它不测量 Cache Miss，也不代表真实 DRAM 传输字节。**

## 4. A 模块验收

```bash
python -m pytest -q
```

测试涵盖：固定两门接口兼容、随机复数态、独立的完整矩阵参照、多个目标位置、重复门、空序列、输入校验以及逻辑访存公式。

完成本模块不等于完成实验 A/B，也不等于 Fusion、Benchmark 或 Cache 测量完成。后续要等 B、C、D、E 的代码与测试集成后才能形成正式实验对照。
