# PQsim 源码入门：从 State Vector 到 Gate Fusion

> 本文根据小组分享的 PQsim 学习讨论整理，并核对了 [PQsim 官方 GitHub 源码](https://github.com/quantex/PQsim)。它是**学习指南**，不是已完成的实验结果，也不要求组员完整复现 PQsim。

## 0. 为什么学习 PQsim？

[**PQsim**](https://github.com/quantex/PQsim) 是面向教学的经典量子线路模拟器，使用 State Vector（态向量）表示量子态。它的关键好处是：用较短的 Python / NumPy / Numba 代码，展示量子门如何通过数组索引执行运算。

**我们要学的不是 PQsim 的全部功能，而是这一条链条：**

\`状态向量如何存储 → H 门如何定位并修改振幅 → 两个 H 门为什么会多次遍历数组 → Gate Fusion 怎样实现局部数据复用\`。

当前课程研究焦点是 **CPU 内存访问组织与运行性能**，不是实现完整量子计算机或证明任何特定硬件上的加速。

## 1. 推荐读的源码（已核对实际路径）

| 阅读顺序 | PQsim 文件 | 看什么 | 对应本仓库 |
|---|---|---|---|
| 1 | [README.md](https://github.com/quantex/PQsim/blob/master/README.md) | 教学目标、statevector、Numba 等基本概念 | 本仓库 \`README.md\` |
| 2 | [pqsim/ui.py](https://github.com/quantex/PQsim/blob/master/pqsim/ui.py) | \`qsim.run()\` 如何创建零初始态 \`|00...0⟩\` | \`common.py\` |
| 3 | [pqsim/numbaQC.py](https://github.com/quantex/PQsim/blob/master/pqsim/numbaQC.py) | \`h(n, qb0, vec)\` 如何根据目标量子位找到振幅对 | \`baseline.py\` |
| 4 | [examples/simple_sim.py](https://github.com/quantex/PQsim/blob/master/examples/simple_sim.py) | 调用入口和量子线路由门序列组成的形式 | \`test_correctness.py\` |
| 选读 | [pqsim/numpyQC.py](https://github.com/quantex/PQsim/blob/master/pqsim/numpyQC.py) | 另一种后端的数组/向量化表达方式 | 对照理解，不直接用于本实验 |

**注意**：PQsim 的 \`examples/simple_sim.py\` 原代码存在命名不一致，例如 \`import numpy as numpy\` 后使用 \`np\`，以及 \`parms\` / \`params\` 混用；它还依赖 Qiskit 相关代码。**不要直接复制该示例并假定可以运行**。入门时直接运行本仓库已整理的代码更稳妥。

## 2. 学会四件事

### 2.1 State Vector 如何占用内存？

\`n\` 个量子位的纯态可以写成

\`\`\`text
|ψ⟩ = Σ_i α_i |i⟩,  i = 0, 1, ..., 2^n-1
\`\`\`

\`α_i\` 是复数振幅。经典模拟器需要保存 \`2^n\` 个振幅。例如 \`n=3\` 时是 **\`2^3=8\`** 个复数（不是 \`23\`）。

PQsim 在 \`pqsim/ui.py\` 中使用：

\`\`\`python
vec = np.zeros(2**nq, dtype=complex)
vec[0] = 1.
\`\`\`

它创建 \`|00...0⟩\`。本仓库的 \`common.py\` 则另外提供归一化随机复数态用于测试，避免只检查零初始态而遗漏错误。

本仓库统一采用 \`complex128\`（每个振幅 16 字节），所以状态数组大小为 \`16 * 2^n\` 字节，不包括 Python/NumPy 的额外对象开销。例如 \`n=20\` 时，振幅数组是 16 MiB。

**学习检查**：为什么量子位每增加 1，数组大小翻倍？为什么每个元素是复数？

### 2.2 H Gate 如何访问数组？

Hadamard 门矩阵：

\`\`\`text
H = (1/√2) * [[1, 1],
              [1,-1]]
\`\`\`

对一对振幅 \`(a, b)\`：

\`\`\`text
a' = (a+b)/√2
b' = (a-b)/√2
\`\`\`

本仓库与 PQsim 使用**最低有效位对应第 0 量子位**的下标约定。作用于目标量子位 \`q\` 时，成对振幅下标相差 \`2^q\`。

PQsim \`pqsim/numbaQC.py\` 的 \`h()\` 中：

\`\`\`python
stripe = 2**qb0
stride = 2*stripe
# 每次循环选择 j 和 k=j+stripe 这一对振幅，再计算并写回
\`\`\`

它使用了 \`@njit(parallel=True)\` 和 \`prange\`。本仓库 \`baseline.py\` 使用**串行的 Numba JIT 内核**来减少实验干扰；两者写法和并行设置并不完全相同，**不能拿 PQsim 的耗时直接和我们的 Fusion 耗时比较来归因于门融合**。

**学习检查**：当 \`q=0\`、\`q=1\`、\`q=3\` 时，哪几组振幅会成为一对？

### 2.3 Baseline：两个门按顺序执行

本项目固定研究同一个数学任务：对状态先执行 \`H_0\`，再执行 \`H_1\`。

\`baseline.py\` 的 \`_baseline_kernel()\` 调用两次单量子位 H 内核，因此有两轮完整的状态数组遍历：

\`\`\`text
第 1 轮：读振幅 → H_0 → 写振幅
第 2 轮：读振幅 → H_1 → 写振幅
\`\`\`

在**算法层面的读写次数模型**中，每个振幅被读取两次、写回两次。但这不等于数据一定从 DRAM 读了两次：第二轮可能从 CPU Cache 命中。

**学习检查**：为什么“两个门”通常意味着两次遍历？哪些情况下第二次访问可能仍在 Cache 中？

### 2.4 Fusion：一次读取四个振幅，局部完成两门

\`fusion.py\` 将相邻的四个振幅 \`(a,b,c,d)\` 作为一个局部计算单元，实施 \`H⊗H\`：

\`\`\`text
a' = (a+b+c+d)/2
b' = (a-b+c-d)/2
c' = (a+b-c-d)/2
d' = (a-b-c+d)/2
\`\`\`

在这份代码中，Fusion 处理四振幅只需局部读一次、写一次。这是**特定两门、特定下标安排**的简化例子；一般的 Gate Fusion 对任意门位、门序列不一定如此简单。

算法层面，Fusion 从“两轮全数组读写”变成“一轮处理”。但它是否加速还要看内核的指令、计算开销、Cache、数据规模、编译器优化等，必须实测。

**学习检查**：为什么不能只是把两个原来的函数写在同一个函数里，就声称完成了访存融合？为什么两个版本需要做正确性对照？

## 3. 直接把学习和实验连起来

| 第一步：全组共同理解 | 第二步：自己动手验证 | 应留下的证据 |
|---|---|---|
| 读 PQsim README 和 \`ui.py\` | 查看 \`common.py\`，自己计算 \`n=3/10/20\` 的数组大小 | 简短的理解记录 |
| 读 PQsim \`numbaQC.py\` 的 \`h()\` | 逐行解释 \`baseline.py\` 的索引 | 说明 \`q=0\` 和 \`q=1\` 的振幅对 |
| 读 \`fusion.py\` | 检查局部四振幅公式和数组写回位置 | 数学等价性解释 |
| 对照 \`test_correctness.py\` | 在自己电脑运行 \`python -m pytest -q\` | 实际测试输出和环境 |
| 最后才看 \`benchmark.py\` | 快速试跑 \`python benchmark.py --ns 12 14 16 --repeats 5\` | 原始 CSV、环境记录；不要提前认定结果 |

建议先完成前四行，性能实验是下一阶段的重点。全组可以分模块推进，但人人应理解总体研究问题、测试的意义和结论边界。

## 4. 要回答的核心研究问题

1. **正确性**：Baseline 和 Fusion 是否产生等价的状态向量？（\`test_correctness.py\`）
2. **理论访问量**：在本项目模型下，逻辑振幅读写次数如何变化？
3. **实际耗时**：不同 \`n\` 下谁更快、加速比多少？（\`benchmark.py\`）
4. **机制解释**：什么时候两轮访问仍可在 Cache 中完成？何时可能受到带宽或指令开销影响？
5. **可选硬件证据**：能否在可用的平台测量可靠的硬件计数器？这不是第一阶段必做项。

**严禁把“逻辑读写次数减半”直接写成“DRAM 流量减半”或“Cache Miss 减半”；也不要在拿到实测结果前声称 Fusion 更快。**

## 5. 暂时不用做什么

- 不需要读完 PQsim 的所有文件，甚至不需要安装它才可以学习这个项目。
- 不需要先学完整套量子计算或 Shor/Grover 算法。
- 不需要在当前阶段改用 C++、SIMD、GPU、多线程优化。
- 不需要把 Google [qsim](https://github.com/quantumlib/qsim) 作为必须读完的代码库；它可作为后续扩展参考。
- 不需要为了增加项目规模而实现任意门集合或完整模拟器。

## 6. 来源与核对说明

- 小组分享的原始讨论：提出从 PQsim 认识态向量、H 门、逐门执行和融合执行的学习路径。
- 已核对 [PQsim GitHub](https://github.com/quantex/PQsim) 的 README、\`pqsim/ui.py\`、\`pqsim/numbaQC.py\`、\`examples/simple_sim.py\`。
- 本文将原对话里的“第一天/明天”调整为与小组现阶段项目进度一致的学习任务；没有将讨论中的设想记为已完成的实验。
