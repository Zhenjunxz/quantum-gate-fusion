# A 模块交接与验收清单

本文件只记录 **Baseline（不融合的 H 门参考实现）** 的交付要求；不要求 B、C、D、E 修改自己的代码。

## 建议对接接口（等待 C 在 Issue #3 确认）

- **当前实验 A/B 必要能力**：`run_baseline_inplace(state, q1=0, q2=1)`。默认 H₀→H₁ 兼容此前探索实验；支持不同位置，如 H₀→H₅。
- **可选扩展**：`run_baseline_sequence_inplace(state, targets=[...])` 和 `prepare_baseline_sequence(n_qubits,targets)`，用于更长的 H-only 不融合对照；不要因此强制 B 立即实现多门融合。
- **内部分析工具**：`baseline_access_profile`、`baseline_logical_access_trace`、`baseline_cache_line_pair_profile`、`baseline_cache_model.py`。理论地址几何和理想 LRU **不等于真实 CPU Cache Miss/带宽测量**。

## 正确性验收（A 自己负责）

在 `feature/baseline-targets` 分支，安装项目依赖后：

```bash
python -m pytest -q
python verify_baseline_a.py
python verify_baseline_a.py --output results/baseline_a_acceptance.json
```

- 默认不写文件；指定 `--output` 会生成带环境版本、7 个验收案例、最大数值误差、范数误差和理论逻辑字节数的 **JSON 正确性记录**。
- 记录只在全部检查通过后写入；它**不包含 CPU 运行时间、Cache Miss 或 DRAM 实测**，也不能代替 E 的正式 Benchmark。
- 新增 `test_baseline_invariants.py`：逐一检验小规模下每个 H 门目标位置的**基态变换公式**，核对逻辑访问轨迹是否使每个振幅恰好读取/写入一次；另测不同 H 门的交换性、同一门两次作用的恒等性质，以及验收 JSON 的格式。
- `results/` 已被 .gitignore 忽略；如需共享验收 JSON，可在 Issue/PR 评论贴出执行环境或另行上传，不要把它混进正式计时数据。

## 原理与边界

对第 `q` 个 qubit 的一个 H 门，状态索引按 bit `q` 配成 `(i, i ^ (1 << q))`。枚举下标 bit `q=0` 的所有 `i`，得到互不重叠的 `2**(n-1)` 对，总计恰好覆盖 `2**n` 个振幅。这保证**源码逻辑访问模型**每门读取、写回各 `2**n` 次；不保证实际 CPU 指令或 DRAM 流量就是这个数量。

## 合并前由小组确认

- [ ] C 确认 `q1/q2`（公共接口）与 `targets`（A 可选接口）的命名和用途。
- [ ] A 的两门默认行为未改变；`python -m pytest -q` 通过。
- [ ] 能独立运行 `python verify_baseline_a.py` 并解释至少一个基态的 H 门变换。
- [ ] 明确内部 Cache 模型只是教学/机理分析，不进入 E 的实测数值。
- [ ] 至少一名其他成员 Review PR #4 后再决定合并 `main`。

相关链接：
- [PR #4](https://github.com/Zhenjunxz/quantum-gate-fusion/pull/4)
- [Issue #3（接口待确认）](https://github.com/Zhenjunxz/quantum-gate-fusion/issues/3)
