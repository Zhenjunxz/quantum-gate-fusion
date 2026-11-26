# E 模块使用说明

此版已接入 A 的 PR #4（bca34bf）与 B 的 PR #5（7a8216f），并根据 11 月 24 日 C 的意见更新。完整集成包可以解压到新目录使用；不要用桌面的旧 baseline.py/fusion.py 搭配新版 benchmark.py。没有修改 GitHub 上的 PR。

## 已完成

- 实验 A 默认 n=12、14、16、18、20、22，固定 H0/H1。
- 实验 B 固定 n=20，自动测 H0/H1、H0/H5、H0/H10，已完成对接和试跑。
- 预热参数名为 warmup_iterations，默认 10 次，每个版本、每组配置分别预热，预热不调用计时器、不计入统计。正式重复默认 20 次。
- 每次计时前复制相同初始状态；复制、正确性检查、首次编译不计时。交替两个版本的运行顺序。计时前也核对所测内核与各自公开函数的输出。
- 使用 time.perf_counter()，保存秒数并换算成毫秒、纳秒。纳秒列是单位换算，不表示计时精度达到 1 纳秒。
- 默认 --timing-scope kernel，直接调用计算内核，移除原来的 partial 适配包装。复制和参数解包都发生在开始计时前；公开函数的输入检查不计入，但仍有 Python 调用与 Numba 调度开销。Baseline 对所有位置统一使用通用内核。
- 可选 --timing-scope public，测完整公开函数调用，包含输入检查。该模式中 Baseline 的 (0,1) 使用其原有专用路径，已记录在环境文件。两种模式的数据必须分开分析。
- 保存原始时间、执行顺序、中位数、Q1、Q3、IQR、最小值、最大值、加速比和正确性误差。四分位区间是运行波动，不是置信区间。
- 保存运行环境、源文件 SHA256、可获得时的 Git 提交号及测量规则。
- 所有参数在任何计时前校验，拒绝越界、重复目标、非法门/分组、bool 伪整数及非法重复/预热次数。两种模式都先验证合法输入，public 模式在每次计时内仍执行其正常输入检查。
- 每次运行新建一个 UTC 时间命名的目录，不覆盖之前数据。所有配置的全部计时循环结束后才写两个 CSV，测量过程中不穿插 CSV 或环境文件写入。环境文件在开始和结束/失败时写入，标明是否完整完成；中途失败不会保存已测部分的 CSV。
- 可用时输出实验 A/B 的运行时间和加速比图。

## 命令

在项目目录、激活项目原有 Python 环境后执行：

```bash
# 小规模试跑，只检查流程
python benchmark.py --experiment A --ns 4 6 --repeats 3 --warmup-iterations 10

# 实验 A 的候选正式配置（统一测量协议后重新运行）
python benchmark.py --experiment A --ns 12 14 16 18 20 22 --repeats 20 --warmup-iterations 10

# 实验 B（可运行，正式测量参数仍待小组确定）
python benchmark.py --experiment B --fixed-n 20 --pairs 0,1 0,5 0,10

# 两组实验连续执行
python benchmark.py --experiment both

# 完整公开函数调用模式，对比时不要与内核模式混合
python benchmark.py --experiment both --timing-scope public
```

## 输出

- raw_timings.csv：一行是一种方法的一次运行，含实验类型、n、门位置、gate_sequence、fusion_groups、轮次、轮内顺序，以及秒/毫秒/换算纳秒时间。
- summary.csv：一行是一组参数；speedup 为 Baseline 中位时间 / Fusion 中位时间。
- environment.json：实验参数、环境、测量规则、代码指纹和运行状态。
- experiment_A.png / experiment_B.png：运行时间（中位数与四分位区间）和加速比图，需 Matplotlib。

## 统一入口与输入格式

```python
run_benchmark(
    n_qubits,
    gate_sequence,
    fusion_groups,
    *,
    repeats=20,
    warmup_iterations=10,
    seed=42,
    output=Path("results"),
    timing_scope="kernel",
)
```

调用示例：

```python
from benchmark import run_benchmark

run_benchmark(
    n_qubits=20,
    gate_sequence=[("H", 0), ("H", 5)],
    fusion_groups=[[0, 1]],
    warmup_iterations=10,
)
```

gate_sequence 按执行顺序列出两个 H 门；fusion_groups 的 0、1 是门序列的位置，不是 qubit 编号。返回结果目录 Path。当前能力只覆盖本项目实验所需的两个不同目标 H 门，融合分组必须为 [[0,1]]；不支持的多门、其他门或其他分组会显式拒绝，不能静默忽略。

C 仅提供了参数名称，尚未给出内部数据格式，因此这里采用并明确记录上述简单约定，供全组对齐。批量实验使用 run_experiments；它复用同一校验和测量流程，等全部配置测完才写 CSV。命令行仍支持 --ns、--pairs 快速生成实验配置，--warmups 作为旧命令的兼容别名保留，默认值也已改为 10。

## 计时边界与仍需确认的事项

1. A/B 公开函数均使用 run_xxx_inplace(state, q1=0, q2=1)，E 的统一参数通过 validate_case 转成 q1/q2 后对接。
2. kernel 模式直接调用内部 _baseline_targets_kernel 与 _fusion_kernel。用 Python 的 perf_counter 包围 Numba 函数调用，无法完全排除 Python 调用、Numba 调度和计时器本身的开销，不能承诺“零函数调用开销”。若必须严格测原生计算体内部时间，需要另行实现与验证原生计时方案，不能仅靠移动 Python 计时器解决，也不应随意扣除一个固定时间。
3. 正式重复次数、机器和输入准备方式仍需全组统一。预热已按 C 的要求默认设为 10，但稳定性仍应在正式机器上检查。
4. 已根据这两份内核核对：Baseline 两次完整读写、Fusion 一次完整读写，对应 4N×16 与 2N×16 字节；这不是硬件内存流量。内核后续改变时需重新核对。

每次计时前复制数组会影响 Cache，当前记录为 Cache 状态未受控。该方案既不保证冷 Cache，也不保证整个数组都在热 Cache 中。不要把观察到的加速直接当成 Cache Miss 减少的证据。

本程序中的输出一致性检查不能替代 D 的独立参照正确性测试。正式采集应在集成和正确性验收后进行，新旧计时范围的数据不要混用。

## 本次验证

此前已对九组计划配置进行联调。11 月 24 日修改后增加 test_benchmark.py，验证统一入口、默认预热 10 次且不进入统计、非法参数在计时前拒绝、复制在计时外、所有测量先于 CSV 写入、时间单位换算、交替顺序，以及 kernel/public 模式非相邻门的输出。

执行 python -m pytest -q：857 项通过（A/B 原有 832 项，E 新增 25 项），原有测试有一条 pytest 弃用提示。功能试跑与测试均不能作为正式性能数据。


## 2026-11-26 集成更新（以此节和 docs/INTEGRATION.md 为准）

C 的单 backend API 现为 `run_benchmark(backend_fn, n_qubits, gate_sequence, fusion_groups, timing_scope='kernel', warmup_iterations=10, repetitions=100)`，返回 `BenchmarkResult`；用 `prepare_backend` 绑定 A/B。本文上方原 `run_benchmark` 双方法调用请改成 `run_paired_benchmark`。CLI 继续支持 `--repeats`，默认 100；显式 20/30 等参数保持有效。门格式规范为 `('H', [q])`，旧标量仍接受，CSV 统一输出新格式。计时改为 `perf_counter_ns()` 差值后换算。原试跑和历史测试记录仅描述 E 原 PR，不是集成版的验证结果。
