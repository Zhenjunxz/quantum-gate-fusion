# D 模块：Baseline / Fusion 独立正确性验证

## 1. 模块目标

本模块负责验证两个 H 门的 **Baseline** 与 **Gate Fusion** 实现在相同输入下是否数值等价、是否符合独立数学参考结果，以及是否正确处理量子位目标和非法输入。**不进行性能计时、Cache Miss 或内存带宽测量。**

验证范围：`complex128` 状态向量；两个不同目标量子位上的 H 门；q0 为最低有效位；原地更新接口：

```python
run_baseline_inplace(state, q1=0, q2=1)
run_fusion_inplace(state, q1=0, q2=1)
```

## 2. 新增文件

| 文件 | 功能 |
| --- | --- |
| `d_oracle.py` | 小规模使用 `np.kron` 构造独立完整矩阵参考；较大规模使用独立振幅公式抽样核对。 |
| `test_d_independent.py` | pytest 正确性测试，覆盖基态、随机复数态、目标位组合、数值性质、原地语义及非法参数。 |
| `verify_d.py` | 运行快速/完整配置验收，生成含环境、源码 SHA-256、误差指标的 JSON 报告。 |
| `README_D.md` | 本说明。 |

## 3. 验证方法

对同一份输入状态 `state` 分别复制后运行 Baseline 与 Fusion，并构造独立数学参考：

\[
\psi_A = H_{q_2}H_{q_1}\psi, \quad
\psi_B = U_{\mathrm{fusion}}\psi, \quad
\psi_{\mathrm{ref}} = U_{\mathrm{exact}}\psi.
\]

要求三者在 `rtol=1e-12`、`atol=1e-12` 的容差内一致。小规模参考通过 Kronecker 积构造完整矩阵；大规模参考通过四项 Hadamard 振幅公式抽查选定的输出元素，同时进行 A/B 全数组比较。大规模**不是**独立参考的全数组逐项验证。

其他检查包括：

- 计算基态与随机复数态、不同目标位置（包括反向输入顺序）。
- 线性性、两次相同联合 H 变换可恢复原态、范数保持。
- 非法目标、错误类型/形状/内存布局的输入被拒绝，且拒绝前不修改原数组。
- 原地修改与函数返回 `None` 的接口约定。

## 4. Windows 运行方式

进入包含 `baseline.py`、`fusion.py`、`common.py` 以及上述 D 文件的**仓库根目录**，使用已配置的虚拟环境：

```cmd
.\.venv\Scripts\python.exe -m pytest -q test_d_independent.py
.\.venv\Scripts\python.exe verify_d.py --mode quick --output results/d_quick.json
.\.venv\Scripts\python.exe verify_d.py --mode full --output results/d_full.json
.\.venv\Scripts\python.exe -m pytest -q
```

如需保存完整 pytest 输出：

```cmd
.\.venv\Scripts\python.exe -m pytest -q > results\pytest_full.txt 2>&1
```

## 5. 本地执行结果（2026-11-20）

以下结果来自开发者本地当前集成文件夹，**不等于 GitHub 合并后 CI 已通过**：

| 命令 / 项目 | 本地结果 |
| --- | --- |
| `pytest -q test_d_independent.py` | 80 passed |
| `verify_d.py --mode quick` | 8/8 passed |
| `verify_d.py --mode full` | 9/9 passed（含一项重复配置，共 8 种不同参数） |
| `pytest -q` | 888 passed, 1 warning |

完整验收涵盖 `n=12,14,16,18,20,22`（目标 `(0,1)`），以及 `n=20` 的目标 `(0,1)`、`(0,5)`、`(0,10)`。最大 A/B 振幅差异位于浮点舍入误差量级。原始逐配置数值请以实际运行生成的 `results/d_full.json` 为准，不在这里转抄伪造。

当前 warning 是 `test_baseline_access_trace.py` 的 pytest 参数化弃用警告，并非计算正确性失败。

## 6. 结论与限制

在上述本地测试范围内，未发现 Baseline / Fusion 的数值正确性、目标位索引或输入处理问题。但自动测试不能证明对所有可能输入绝对正确；大规模独立参考为抽样检查。本模块不支持对任意多门序列、其他量子门类型作出正确性结论，也不提供性能或实际硬件 Cache 指标。

## 7. 提交说明

本 PR 仅提交 `d_oracle.py`、`test_d_independent.py`、`verify_d.py` 和 `README_D.md`，可按需附上生成的 `results/d_full.json`。不应把 `.venv/`、`__pycache__/`、Numba 缓存或未经团队确认的 A/B 源码更改包含进来。由于 A/B 可能仍在独立 PR 中，请在团队指定的集成分支上创建 D PR，避免意外把组员尚未合并的提交带入 PR。
