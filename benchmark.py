"""Fair, repeatable *software* timing (not DRAM/CPU cache measurements).

Each in-place kernel receives its own copy of one normalized input state.
Copies, correctness checks, and Numba compilation are excluded from timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numba
import numpy as np

from baseline import run_baseline_inplace
from common import random_state
from fusion import run_fusion_inplace


METHODS = {"baseline": run_baseline_inplace, "fusion": run_fusion_inplace}


def measure_once(method, source: np.ndarray) -> int:
    state = source.copy()  # Not part of kernel timing.
    start = perf_counter_ns()
    method(state)
    return perf_counter_ns() - start


def run_benchmark(ns: list[int], repeats: int, seed: int, output: Path) -> None:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if any(n < 2 or n > 24 for n in ns):
        raise ValueError("n_qubits must be between 2 and 24; large arrays need substantial RAM")

    output.mkdir(parents=True, exist_ok=True)
    records = []
    summary = []
    for n in ns:
        source = random_state(n, seed)
        left, right = source.copy(), source.copy()
        run_baseline_inplace(left)  # First call triggers JIT compilation if needed.
        run_fusion_inplace(right)
        max_error = float(np.max(np.abs(left - right)))
        if not np.allclose(left, right, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"Correctness test failed at n={n}: max_error={max_error}")

        # A few untimed warm-up calls minimize first-call and cold-start effects.
        for method in METHODS.values():
            for _ in range(2):
                measure_once(method, source)

        timings = {name: [] for name in METHODS}
        for i in range(repeats):
            # Alternate order to reduce systematic drift favoring one implementation.
            order = list(METHODS) if i % 2 == 0 else list(reversed(METHODS))
            for name in order:
                elapsed_ns = measure_once(METHODS[name], source)
                timings[name].append(elapsed_ns)
                records.append({"n_qubits": n, "method": name, "repeat": i + 1,
                                "elapsed_ns": elapsed_ns, "elapsed_ms": elapsed_ns / 1e6})

        baseline_ms = median(timings["baseline"]) / 1e6
        fusion_ms = median(timings["fusion"]) / 1e6
        n_amplitudes = 1 << n
        bytes_per_amplitude = np.dtype(np.complex128).itemsize
        # Baseline: 2 full reads + 2 full writes; fusion: 1 read + 1 write.
        # These are algorithm-level byte counts, not measured memory-bus traffic.
        summary.append({
            "n_qubits": n,
            "array_mib": n_amplitudes * bytes_per_amplitude / (1024 ** 2),
            "baseline_median_ms": baseline_ms,
            "fusion_median_ms": fusion_ms,
            "speedup": baseline_ms / fusion_ms if fusion_ms > 0 else float("nan"),
            "baseline_logical_bytes": 4 * n_amplitudes * bytes_per_amplitude,
            "fusion_logical_bytes": 2 * n_amplitudes * bytes_per_amplitude,
            "max_abs_error": max_error,
        })
        print(f"n={n:>2} baseline={baseline_ms:.3f}ms fusion={fusion_ms:.3f}ms "
              f"speedup={baseline_ms/fusion_ms:.2f}x error={max_error:.2e}")

    write_csv(output / "raw_timings.csv", records)
    write_csv(output / "summary.csv", summary)
    environment = {
        "python": sys.version,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "seed": seed,
        "repeats": repeats,
        "n_qubits": ns,
        "method": "Numba compiled in-place kernels; state copies excluded from timing; alternating order",
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(ns, [r["baseline_median_ms"] for r in summary], marker="o", label="Baseline")
        ax.plot(ns, [r["fusion_median_ms"] for r in summary], marker="o", label="Fusion")
        ax.set(xlabel="Qubits (n)", ylabel="Median runtime (ms)", title="H0 + H1: Unfused vs Fused")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output / "runtime.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("Matplotlib not installed; skipping plot.")
    print(f"Saved benchmark outputs in {output.resolve()}")


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare H-gate baseline and fusion with Numba")
    p.add_argument("--ns", type=int, nargs="+", default=[12, 14, 16, 18, 20])
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("results"))
    args = p.parse_args()
    run_benchmark(args.ns, args.repeats, args.seed, args.output)


if __name__ == "__main__":
    main()
