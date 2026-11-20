# 经典量子态向量模拟：Gate Fusion × CPU 访存

**课程**：计算机组成原理 Course Project（小组协作）  
**研究问题**：两个连续 Hadamard 门的融合，如何改变算法层面的状态数组访问次数？这种减少何时转化为 CPU 上的执行时间变化？

> 当前版本是可验证的**最小实验原型**，不是完整的量子模拟器，也不预设融合一定更快。性能结论必须以小组成员在自己机器上获取的真实数据为准。

## 1. 研究对象与实现

- 状态表示：长度 `2**n` 的一维 `numpy.complex128` 数组；第 0 号量子位是数组下标的**最低有效位**。
- `baseline.py`：两个 H 门按顺序执行（先 `H_0`，再 `H_1`），分别遍历整个数组。
- `fusion.py`：`run_fusion_inplace(state, q1=0, q2=1)` 支持两个不同的目标位。按未操作的位将状态分为四振幅组，每组读入一次、应用 `H ⊗ H`、写回一次。默认仍执行 `H(0)` 和 `H(1)`。
- 两种实现都由 **Numba `@njit(cache=True)`** 编译，输入类型、精度、数学任务一致，且都原地修改输入。

对局部振幅 `(a,b,c,d)`，融合结果为

```text
a' = (a+b+c+d)/2
b' = (a-b+c-d)/2
c' = (a+b-c-d)/2
d' = (a-b-c+d)/2
```

### 理论访问量（非实测 DRAM 流量）

假设 `N=2**n` 个复数，每个复数 16 字节：

| 方法 | 每个振幅的逻辑读次数 | 逻辑写次数 | 算法层面的读写字节数 |
|---|---:|---:|---:|
| Baseline | 2 | 2 | `4 * N * 16` |
| Fusion | 1 | 1 | `2 * N * 16` |

此处逻辑数组访问量不等于 Cache Miss、DRAM 实际传输或 CPU 指令数。两种实现的算术执行路径可能也不完全相同，不能把所有时间差都归因于主存。

## 学习资料

- **[PQsim 源码入门学习指南](docs/PQSIM_STUDY_GUIDE.md)**：阅读顺序、关键源码解释、H Gate 振幅索引、Baseline/Fusion 的对应关系和小组检查题。

## 2. 安装与验证

推荐 Python 3.11，macOS / Linux / Windows 均可执行（需有可用的 Numba 平台支持）。

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q
```

测试包括：

1. 多种随机归一化复数状态，对比 Baseline / Fusion；
2. 零基态输入；
3. `n=2` 时 NumPy `kron(H,H)` 独立数学参照；
4. H 变换执行两次应恢复原状态；
5. 无效输入类型、长度检查。
6. 不同目标位及反序目标的独立矩阵参照、非法目标拒绝且输入保持原样。

## 3. 运行初步性能实验

先用小规模快速检查：

```bash
python benchmark.py --ns 12 14 16 --repeats 5
```

再做范围更大的实验（视电脑内存情况调整）：

```bash
python benchmark.py --ns 12 14 16 18 20 22 --repeats 7
```

**输出（本机运行后生成，不包含任何预设的实验数据）**：

- `results/raw_timings.csv`：每次测量的原始数据，单位 ns / ms
- `results/summary.csv`：每个规模的运行时间中位数、加速比、理论访问字节数、最大数值误差
- `results/environment.json`：软硬件环境、随机种子与实验设置
- `results/runtime.png`：运行时间折线图

计时注意事项：

- 首次 Numba 编译、测试验证、状态数组生成与每轮输入复制**均不计入**计时。
- 两个方法交替先后次序，以减弱顺序造成的系统性偏差；仍需避免后台程序/热节流干扰。
- 每次方法调用都在原始状态的独立副本上执行；内核比较是**单次两门计算的执行时间**，而非复制和生成输入的总时间。
- 初步结果可以用于探查异常，正式结果建议增加重复次数、多轮独立实验和更多的规模点。
- 复制数组虽然在计时区间外，但可能改变 CPU Cache 的热状态；当前数据并不单独证明 DRAM 流量变化。若要区分 Cache 和主存，需要补充合适的硬件计数器或专门设计的控制实验。
- 默认忽略 `results/`，避免把个人设备上未经核对的实验结果误认为全组已经共同验证的结论。

## 目前已归档的实验

- **[Mac 首次性能试跑（2026-11-12）](experiments/mac_run_01/README.md)**：保存 6 个规模、每种方法 15 次的原始计时与中位数汇总、运行环境及可预览折线图。该记录属于初步探索，不代表最终性能结论。
- 小组成员独立复现与讨论：[GitHub Issue #1](https://github.com/Zhenjunxz/quantum-gate-fusion/issues/1)。

## 4. 文件结构

```text
quantum-gate-fusion/
├── baseline.py            # 逐门 H_0 再 H_1
├── fusion.py              # 四振幅融合 (H ⊗ H)
├── common.py              # 输入验证 / 随机状态生成
├── test_correctness.py    # pytest 正确性测试
├── benchmark.py           # 计时、CSV、折线图和环境记录
├── requirements.txt       # Python 依赖
├── .github/workflows/tests.yml  # GitHub Actions 自动运行正确性测试
├── .gitignore
└── README.md
```

## 5. 本周（截至 2026-11-15）的建议验收

- [ ] 全组能解释 State Vector、H Gate 以及逐门/融合的数学等价性。
- [ ] 每位成员至少独立运行 `python -m pytest -q` 并记录结果。
- [ ] 查看 `baseline.py` 与 `fusion.py` 的数组下标和运算，确认没有错误比较。
- [ ] 用 `--ns 12 14 16 --repeats 5` 试运行，不把试运行数字当作最终研究结论。
- [ ] 在 Issues / PR 中记录发现的问题及修复原因。

**12 月初目标**：可复现代码、原始实验数据、可视化、计组课程知识解释、报告与 Pre。课程正式提交格式及精确日期以任课教师发布为准。

## 6. 小组协作建议

- 任何成员要修改实验方法时，先在 GitHub Issue 里描述目标和影响，再提交 Pull Request。
- 性能数据要附上环境、参数、原始结果，并由其他成员独立复核。
- 所有代码可以借助 AI 起草，但组员必须理解其运行过程、测试方法及实验限制。
