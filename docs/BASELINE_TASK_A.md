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

## 5. A 模块深化：复用已准备的门序列

在相同门序列上重复执行时，可先准备一次目标位置数组，避免每次从 Python 列表重复转换（这是 A 模块的能力，**不规定** E 的计时协议）：

```python
from baseline import prepare_baseline_sequence
plan = prepare_baseline_sequence(n_qubits=20, targets=[0, 5, 10])
plan.run_inplace(state)  # 与逐门 H0 -> H5 -> H10 等价；仍是 unfused
```

- 创建时立即检查 `n_qubits` 和全部目标位置，并复制为固定的目标序列。调用方之后修改原列表，不影响计划。
- 执行时检查 state 类型、形状和 `2**n_qubits` 的规模要求，然后一次进入 Numba 的逐门循环。
- `plan.run_inplace` 仍有 Python 状态校验；真正的计算是 `_baseline_sequence_kernel`。**没有宣称该接口一定比旧接口快**，要按测量协议决定比较什么。
- 允许空序列和重复门；数学上能抵消的门依然逐个执行。

## 6. A 模块深化：可解释的访问步长/读写模型

```python
from baseline import baseline_access_profile
profile = baseline_access_profile(20, [0, 5, 10])
for gate in profile["per_gate"]:
    print(gate["target"], gate["pair_stride_bytes"], gate["sample_index_pairs"])
```

对 **complex128，16 字节/振幅**，不同目标位的理论配对间距：

| H 目标 qubit | 配对下标之差 | 配对振幅位置的字节偏移 | 最初四对位置 |
|---|---:|---:|---|
| H0 | 1 | 16 B | (0,1), (2,3), (4,5), (6,7) |
| H5 | 32 | 512 B | (0,32), (1,33), (2,34), (3,35) |
| H10 | 1024 | 16,384 B | (0,1024), (1,1025), (2,1026), (3,1027) |

每个 H 门都处理 `2**(n-1)` 对振幅，各读写 `2**n` 个振幅。**改变 H 门位置不改变每门逻辑读写总量，但改变了配对访问模式。**

这里的字节偏移是两个配对振幅的地址间距，**不是**从 DRAM 传输的字节数；该模型不计算 Cache Miss、访存带宽和预测加速比。缓存行是否命中仍需实际测量或进一步证据。

本次 A 模块新增了 `test_baseline_plan.py`，覆盖计划复用、列表快照、错误状态不修改、实际配对位置和理论统计。

## 7. A 模块再深化：受控的逻辑访问轨迹

```python
from baseline import baseline_logical_access_trace
trace = baseline_logical_access_trace(n_qubits=4, targets=[0, 2], max_pairs=8)
for event in trace["trace"]:
    print(event["gate_index"], event["read_indices"], event["write_indices"])
```

- 按照 Baseline 双层循环的**逻辑执行顺序**生成振幅索引：每对先读两个振幅，再写回两个振幅。
- `max_pairs` 是整个序列的最大采样对数，限制为 0–4096，避免为 `n=20+` 生成巨大的逐次访问记录。
- 保留 `gate_index`、目标位置、门内 pair 编号、逻辑读写索引和**相对数组起点**的字节偏移。
- 这是由代码循环结构**推导**的逻辑轨迹，不是仪器捕获的 CPU 指令地址顺序。JIT 优化、prefetch、Cache 和 DRAM 细节可能不同。

## 8. A 模块再深化：缓存行配对地址几何（不是 Cache Miss）

```python
from baseline import baseline_cache_line_pair_profile
report = baseline_cache_line_pair_profile(
    n_qubits=20,
    targets=[0, 5, 10],
    line_bytes=64,           # 假设值；不代表当前 CPU 的真实 Cache Line
    base_offset_bytes=0,     # 假设 state[0] 在缓存行边界
)
for gate in report["per_gate"]:
    print(gate["target"], gate["same_line_pairs"],
          gate["different_line_pairs"], gate["distinct_state_lines_touched"])
```

将状态向量看成连续的 16 B/振幅数组，指定一个**假设**缓存行大小（2 的整数次幂且至少 32 B）和 16 B 对齐的数组起始偏移。程序**精确计算地址几何意义上**每对振幅是否位于同一行，而不是运行模拟 CPU Cache。

例如假定 64 B 行、起始 64 B 对齐，`n=20`：

| Gate | 两振幅间字节距离 | 同一行的配对数 | 不同行的配对数 | 整体涉及的不同缓存行数 |
|---|---:|---:|---:|---:|
| H0 | 16 B | 524,288 | 0 | 262,144 |
| H5 | 512 B | 0 | 524,288 | 262,144 |
| H10 | 16,384 B | 0 | 524,288 | 262,144 |

**关键解释**：
- 不同门的位置可能改变**一对振幅是否属于同一个缓存行**；改变字节间距并不意味着“每门多读了缓存行”，因为一轮完整遍历仍会触及状态向量的全部振幅。
- 同行配对与**缓存命中率**不是一回事：即便一对跨两行，也可能两行都已在 Cache 里。是否发生 Cache Miss 取决于访问时序、替换策略、预取、并发和实际机器。
- 指定的 `line_bytes` 和 `base_offset_bytes` 是模型假设。若讨论具体机器，需与真实硬件参数核对；本函数不会测量 CPU Cache 或 DRAM，也不能预测加速比。
- 计算同一行配对数使用循环结构的**周期性公式**，因此即使 `n=20+` 也不需要枚举全部振幅（时间复杂度主要取决于目标门数和假设行大小）。

新增 `test_baseline_access_trace.py`，用小规模的**独立双层循环穷举**核对逻辑索引顺序和同行配对计数，覆盖 32/64/128 B 不同假设行大小、起始偏移、错误输入和大规模有界采样。

**以上全部是 A 模块的分析辅助接口，不改变数值计算内核，也不修改 B/C/D/E 的模块。**

## 9. A 模块进一步深化：有限容量的理想 LRU Cache 模型

此前的缓存行配对模型只回答“两个振幅是否处于同一行”；本节的**理想化软件模型**进一步研究：若整个序列在一个**假设的有限容量缓存**上按照 Baseline 源码中的顺序执行，访问先后是否会造成不同的*模型缺失次数*？

```python
from baseline_cache_model import simulate_baseline_ideal_lru

report = simulate_baseline_ideal_lru(
    n_qubits=3,
    targets=[0],
    cache_capacity_bytes=64,
    line_bytes=64,
)
print(report["totals"])
```

### 模型假设（重要）

- **完全相联、LRU 替换、write-back、write-allocate，初始为空缓存**，不存在多级 Cache、硬件预取、SIMD/JIT 重排、乱序执行或其他进程的干扰。
- 将 Baseline 代码中每对振幅的逻辑执行描述为 *read(i), read(j), write(i), write(j)*。一振幅 16 B，不跨行；数组起点的对齐偏移可配置。
- Cache 在同一段门序列的不同 H 门之间**持续保留**，每一门仍按源码中的双层循环处理所有振幅，不融合也不消除重复门。
- `cache_capacity_bytes` 必须是 `line_bytes` 的正整数倍；行大小必须是假设的 2 的幂且至少 32 B。函数返回每个门的 read/write hit、read/write miss、evictions、dirty_evictions，以及汇总和最终驻留行数。
- **模型结果不是实际 CPU PMU 的 Cache Miss，也不是实测 DRAM 读写或加速比。** 真实 CPU Cache 大小、关联度、行大小、写策略、预取和编译器行为可能不同；使用本模型只能检验某种*机制解释在给定假设下是否可能*，不能证实硬件行为。

### 直观例子（3 qubits、8 个振幅、假设 64 B 行）

8 个 `complex128` 振幅占 128 B，即两个 Cache Lines。仅保留 **1 行（64 B）**时：
- `H0`：访问成对的相邻振幅，先读完/写完一行，再移动到下一行。模型为 **2 次读缺失、0 次写缺失**。
- `H2`：每对振幅跨越这两行；每对按“读 i、读 j、写 i、写 j”的源码顺序执行，不断替换唯一的缓存行。模型为 **8 次读缺失、8 次写缺失**。
- 如果容量改为 **2 行（128 B）**：在同一假设下，`H0` 和 `H2` 都只需要 **2 次冷启动读缺失**，之后可驻留全部状态。

这说明：**理论读写量相同，但“访存顺序 × 缓存容量”的交互可能造成不同的模型命中率**。这是实验 B 可以检验的*候选机制*，绝不能把这一模型数字冒充真实芯片测量。

### 规模边界

模型逐次跟踪逻辑访存事件，事件总量为 `2 * 2**n * gate_count`。默认最多 **1,000,000 事件**（可调但有上限），超过时提前抛异常，避免对 `n=20+` 的数百万事件做高开销 Python 循环。大规模时继续使用 `baseline_cache_line_pair_profile` 的 O(目标门数 × 行大小) 地址几何模型，**不要用该 LRU 模型生成正式性能数据**。

新增独立的 `test_baseline_cache_model.py`，覆盖手工可核对的 1/2 缓存行情形、H0/H2 的对比、跨门热态、对齐、事件计数守恒、容量和规模边界。

## 10. A 模块深化：组相联度敏感性（理想模型，不是实际 Cache）

仅通过“同一缓存行/跨缓存行”或完全相联 LRU 来解释性能，可能忽略 **Cache Set 冲突（conflict behavior）**。因此 \`baseline_cache_model.py\` 的原函数新增可选参数 \`associativity_ways\`，兼容原调用：

```python
from baseline_cache_model import simulate_baseline_ideal_lru

# 两种容量相同（128 B）、缓存行大小相同（64 B）的模型
full = simulate_baseline_ideal_lru(
    4, [3], 128, line_bytes=64, associativity_ways=2
)
direct = simulate_baseline_ideal_lru(
    4, [3], 128, line_bytes=64, associativity_ways=1
)
print(full["totals"], direct["totals"])
```

定义：\`cache_capacity_bytes / line_bytes\` 是模型缓存行总数，\`associativity_ways\` 是每个 Set 的路数，Set 数量 = 总行数/路数，\`set_index = line_number % number_of_sets\`。每个 Set **单独按 LRU** 替换，写策略仍是假定 write-back、write-allocate，初始 Cache 为空。若省略 \`associativity_ways\`，保留上一版的**完全相联**行为。

**可手算反例：** 4 qubits 状态向量 256 B，共四条 64 B 缓存行，H3 的配对访问会交替触及第 0/2 行、第 1/3 行。模型 Cache 容量固定 128 B（2 条行）：

| 模型 | 组数 × 路数 | 逻辑读取缺失（模型） | 逻辑写入缺失（模型） |
|---|---|---:|---:|
| 直接映射 | 2 sets × 1 way | 16 | 16 |
| 完全相联 | 1 set × 2 ways | 4 | 0 |

这表明**同样容量、同样门序列、同样算法级逻辑读写量，也可能因组冲突而在假设模型中表现不同。** 但这里使用的是简单模运算索引的教学模型，并非真实 CPU 的缓存组织描述；实际机器可能有不同的索引/哈希方式、不同关联度、多级 Cache 和硬件预取。不能用此模型数字冒充硬件 Cache Miss，也不能据此断言现实中哪个门位置必然更快。

**准确性保障：** 新增 \`test_baseline_associativity.py\`，用与 Baseline 双层循环不同的“逐个下标位测试”法列举成对振幅，并使用普通 Python 列表独立模拟每个 Set 的 LRU。测试覆盖多种 n、目标位、缓存行大小、容量、对齐、路数、跨门访问及非法配置。其作用是验证**模型计算与设定的语义一致**，不是验证模型预测真实硬件。

此功能只增加 A 模块分析能力，不修改 Baseline 数值内核或其他人负责的 Fusion/Benchmark。

## 11. A 模块交付验收：独立 NumPy 参照 + 一键检查

仅增加 A 的**数值可靠性验收**，不再扩展 Cache 模型。除了原有 pytest 外，可以直接运行：

```bash
python -m pytest -q
python verify_baseline_a.py
```

新增两个 A 工具：

- `baseline_reference.py::h_sequence_numpy_reference(state,targets)`：使用 **NumPy reshape + moveaxis** 按张量轴实现 H 门，而不是复用 Baseline 的 bit-stride / base-offset 双层循环。函数返回新数组，原始 state 不变，适合交叉验算；**不作为性能对照的 Baseline**。
- `verify_baseline_a.py`：默认对 `n=2,6,8,12,16` 以及重复门、空序列进行数学参照检查；同时核对双门兼容接口、受限 H-only 序列接口、prepared plan 与逻辑读写公式的一致性。只输出误差、逻辑字节数，不计时。

它可以成为 C/D 组员集成前的最小验收手段，但**C 是否采用哪些接口，仍由组内共同决定**。原 README/main 及 B/C/D/E 文件未修改。

特别注意：
1. oracle 与 Baseline 使用相同的数据类型和 Hadamard 数学公式，但不同的**数组索引实现**，因此能发现常见的 endianness、stride、循环边界错误；不代表覆盖所有潜在软件错误。
2. `np.testing.assert_allclose(..., atol=1e-12, rtol=1e-12)` 是当前项目的数值容差，不表示两种浮点计算的结果逐位相同。
3. `verify_baseline_a.py` 和 NumPy oracle 不属于 E 的正式计时方案；其运行时间和内存开销**不应计入 Baseline/Fusion 加速比**。
