# E 模块使用说明

此版已接入 A 的 PR #4（bca34bf）与 B 的 PR #5（7a8216f）。完整集成包可以解压到新目录使用；不要用桌面的旧 baseline.py/fusion.py 搭配新版 benchmark.py。没有合并或修改 GitHub 上的 PR，C 的接口确认仍待进行。

## 已完成

- 实验 A 默认 n=12、14、16、18、20、22，固定 H0/H1。
- 实验 B 固定 n=20，自动测 H0/H1、H0/H5、H0/H10，已完成对接和试跑。
- 预热次数、重复次数、随机种子均可配置。默认预热 3 次、正式重复 20 次；正式次数仍需小组确定。
- 每次计时前复制相同初始状态；复制、正确性检查、首次编译不计时。交替两个版本的运行顺序。计时前也核对所测内核与各自公开函数的输出。
- 默认 --timing-scope kernel，测计算内核调用，排除公开函数的输入检查，仍包含 Python 适配调用与 Numba 调度开销。Baseline 对所有位置统一使用通用内核。
- 可选 --timing-scope public，测完整公开函数调用，包含输入检查。该模式中 Baseline 的 (0,1) 使用其原有专用路径，已记录在环境文件。两种模式的数据必须分开分析。
- 保存原始时间、执行顺序、中位数、Q1、Q3、IQR、最小值、最大值、加速比和正确性误差。四分位区间是运行波动，不是置信区间。
- 保存运行环境、源文件 SHA256、可获得时的 Git 提交号及测量规则。
- 每次运行新建一个 UTC 时间命名的目录，不覆盖之前数据。完成每组参数后保存一次。环境文件状态标明是否完整完成。
- 可用时输出实验 A/B 的运行时间和加速比图。

## 命令

在项目目录、激活项目原有 Python 环境后执行：

```bash
# 小规模试跑，只检查流程
python benchmark.py --experiment A --ns 4 6 --repeats 3 --warmups 1

# 实验 A 的候选正式配置（统一测量协议后重新运行）
python benchmark.py --experiment A --ns 12 14 16 18 20 22 --repeats 20 --warmups 3

# 实验 B（可运行，正式测量参数仍待小组确定）
python benchmark.py --experiment B --fixed-n 20 --pairs 0,1 0,5 0,10

# 两组实验连续执行
python benchmark.py --experiment both

# 完整公开函数调用模式，对比时不要与内核模式混合
python benchmark.py --experiment both --timing-scope public
```

## 输出

- raw_timings.csv：一行是一种方法的一次运行，含实验类型、n、门位置、轮次、轮内顺序和纳秒时间。
- summary.csv：一行是一组参数；speedup 为 Baseline 中位时间 / Fusion 中位时间。
- environment.json：实验参数、环境、测量规则、代码指纹和运行状态。
- experiment_A.png / experiment_B.png：运行时间（中位数与四分位区间）和加速比图，需 Matplotlib。

## 接口与仍需确认的事项

1. 两个公开函数均使用 run_xxx_inplace(state, q1=0, q2=1)。已通过 resolve_methods 接入。C 后续更改接口时，集中修改这个函数即可。
2. 当前默认直接调用内部 _baseline_targets_kernel 与 _fusion_kernel。这是基于同学当前提交的临时接法，请 C 确认是否正式提供这些计时入口，以及是否认可所有位置统一用通用内核。不能称完全没有额外开销。
3. 正式预热次数、重复次数、机器和输入准备方式。
4. 已根据这两份内核核对：Baseline 两次完整读写、Fusion 一次完整读写，对应 4N×16 与 2N×16 字节；这不是硬件内存流量。内核后续改变时需重新核对。

每次计时前复制数组会影响 Cache，当前记录为 Cache 状态未受控。该方案既不保证冷 Cache，也不保证整个数组都在热 Cache 中。不要把观察到的加速直接当成 Cache Miss 减少的证据。

本程序中的输出一致性检查不能替代 D 的独立参照正确性测试。正式采集应在集成和正确性验收后进行，新旧计时范围的数据不要混用。

## 本次验证

在独立副本整合 A、B 的上述提交，用现有项目 Python 环境检查。两组实验共九个计划配置都完成了试跑（每个版本每组重复 3 次，额外预热 1 次），核对了输出数量、加速比计算、数值误差、图表生成；另验证公开函数模式和反序门位置。试跑数据只用于验证功能，不是正式性能结论，且试跑期间另有测试进程运行。
