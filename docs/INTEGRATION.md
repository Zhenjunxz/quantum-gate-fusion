# ABCDE 集成候选版（2026-11-26）

## 来源与边界
A #4 bca34bf、B #5 7a8216f、C #8 65b1796、D #6 334dd87、E #7 6c6918e。
以 A 的完整提交树为基础，复制 B/C/D/E 对应 PR 的文件快照（含各自测试和文档），保留原 PR 供审核。不改变 A/B 数值内核。不是 Windows 报告里的本地提交 130bbfd2，也没有冒充该版本的性能数据。

## 当前统一接口
```python
from benchmark import prepare_backend, run_benchmark, run_paired_benchmark
gates = [("H", [0]), ("H", [5])]
groups = [[0, 1]]
backend = prepare_backend("baseline", 20, gates, groups, "kernel")
result = run_benchmark(backend, 20, gates, groups, repetitions=100)
# result 是 interfaces.BenchmarkResult；interfaces.run_benchmark 也可调用。
# 两种方法交替测量并输出 CSV：
directory = run_paired_benchmark(20, gates, groups, repeats=20)
```
C 的 Gate/Result/API 签名和默认 10 次预热、100 次正式调用已落实。E 原单配置双方法入口更名 run_paired_benchmark；批量 CLI 仍支持 --repeats（也支持 --repetitions），明确指定重复次数即可复测历史协议。兼容 E 旧标量门格式 ("H", 0)，输出统一规范 ("H", [0])。仅支持两个不同目标 H 门、[[0,1]] 分组。A/B 的 q1/q2 低层接口保持不变。

两种入口共用 measure_once（perf_counter_ns 整数差换算秒），复制及参数准备在计时外。C 单 backend API 使用 seed 42 与独立 NumPy oracle；成对驱动保留 E 的交替顺序、全部原始 CSV、IQR、环境记录，补充均值及总体标准差。单 backend 通过 prepare_backend 绑定参数，仍有 partial / Python 调度开销；成对驱动直接调用 A/B，不应混合两条路径的性能结果。自定义 backend 需自行保证计时范围。kernel 不计公开参数校验，public 包含其正常校验。复制会改变缓存状态，不保证冷缓存。

## 验收 / Jetson 试跑
```bash
python -m pip install -r requirements.txt
python -m pytest -q
python verify_baseline_a.py
python verify_d.py --mode full
python benchmark.py --experiment both --repeats 3 --warmup-iterations 2
```
正式复测由全组明确种子、重复次数、供电、功耗与采集方式。例：
```bash
python benchmark.py --experiment both --repeats 30 --warmup-iterations 5 --seed 42 --timing-scope kernel
```
分别为 43、44 重复并保留独立目录。上面只是复现墙钟计时的候选指令，不包含 Windows WPR 或 Jetson PMU；不可用整进程 perf stat 混合计数当作单内核计数。新版本数据不能与旧提交未经核对混合。
