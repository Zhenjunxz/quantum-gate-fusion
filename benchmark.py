"""Repeatable timings for configurable two-H-gate implementations.

Copies, correctness checks and compilation are outside the timed interval.
Kernel timing excludes public validation, but includes Python adapter and
Numba dispatch overhead. Public-call timing is available as a separate scope.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from functools import partial
import json
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

import numba
import numpy as np

from baseline import run_baseline_inplace
from common import random_state
from fusion import run_fusion_inplace
import baseline
import fusion


METHODS = {"baseline": run_baseline_inplace, "fusion": run_fusion_inplace}
INPUT_POLICY = "fresh_source_copy_immediately_before_each_call; copy_not_timed; cache_state_uncontrolled"
DEFAULT_NS = [12, 14, 16, 18, 20, 22]
DEFAULT_PAIRS = [(0, 1), (0, 5), (0, 10)]


def resolve_methods(targets: tuple[int, int], timing_scope="kernel") -> dict:
    """Adapter for PR4 bca34bf / PR5 7a8216f; pending C confirmation.

    Use the generic baseline kernel for ALL targets in kernel mode, avoiding
    a specialized (0,1) path as a confound in the target-position experiment.
    Both adapters incur partial-call and Numba dispatch overhead.
    """
    if timing_scope == "kernel":
        functions = {"baseline": baseline._baseline_targets_kernel,
                     "fusion": fusion._fusion_kernel}
    elif timing_scope == "public":
        functions = METHODS
    else:
        raise ValueError("timing_scope must be kernel or public")
    return {name: partial(fn, q1=targets[0], q2=targets[1])
            for name, fn in functions.items()}


def measure_once(method, source: np.ndarray) -> int:
    state = source.copy()
    start = perf_counter_ns()
    method(state)
    return perf_counter_ns() - start


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty measurement table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_cases(experiment, ns, fixed_n, pairs):
    cases = []
    if experiment in ("A", "both"):
        cases.extend({"experiment": "A", "n_qubits": n, "targets": (0, 1)} for n in ns)
    if experiment in ("B", "both"):
        cases.extend({"experiment": "B", "n_qubits": fixed_n, "targets": pair} for pair in pairs)
    if not cases:
        raise ValueError("At least one case is required")
    for case in cases:
        n, targets = case["n_qubits"], case["targets"]
        if not 2 <= n <= 24:
            raise ValueError("n_qubits must be between 2 and 24")
        if len(targets) != 2 or len(set(targets)) != 2 or any(t < 0 or t >= n for t in targets):
            raise ValueError(f"Need two distinct targets in [0, {n}): {targets}")
    if len({(c["experiment"], c["n_qubits"], c["targets"]) for c in cases}) != len(cases):
        raise ValueError("Duplicate experiment cases are not allowed")
    return cases


def source_identity():
    root = Path(__file__).resolve().parent
    hashes = {}
    for name in ("benchmark.py", "baseline.py", "fusion.py", "common.py"):
        path = root / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "source_sha256": hashes}


def save_plots(output, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; CSV data saved without plots.")
        return
    for experiment in ("A", "B"):
        rows = [r for r in summary if r["experiment"] == experiment]
        if not rows:
            continue
        if experiment == "A":
            rows.sort(key=lambda r: r["n_qubits"])
        x = list(range(len(rows)))
        labels = [str(r["n_qubits"]) if experiment == "A" else
                  f"H{r['target0']}/H{r['target1']}" for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for method in METHODS:
            values = np.array([r[f"{method}_median_ms"] for r in rows])
            lower = values - np.array([r[f"{method}_q1_ms"] for r in rows])
            upper = np.array([r[f"{method}_q3_ms"] for r in rows]) - values
            axes[0].errorbar(x, values, yerr=[lower, upper], marker="o", capsize=3, label=method)
        axes[0].set_ylabel("Runtime (ms), median and Q1-Q3")
        axes[0].legend()
        axes[1].plot(x, [r["speedup"] for r in rows], marker="o")
        axes[1].axhline(1, color="gray", linestyle="--")
        axes[1].set_ylabel("Speedup (baseline median / fusion median)")
        for axis in axes:
            axis.set_xticks(x, labels)
            axis.set_xlabel("Qubits" if experiment == "A" else "H gate targets")
            axis.grid(alpha=0.3)
        fig.suptitle(f"Experiment {experiment}")
        fig.tight_layout()
        fig.savefig(output / f"experiment_{experiment}.png", dpi=150)
        plt.close(fig)


def run_benchmark(ns, repeats, seed, output, *, experiment="A", fixed_n=20,
                  pairs=None, warmups=3, timing_scope="kernel"):
    if repeats < 1 or warmups < 1:
        raise ValueError("repeats and warmups must both be >= 1")
    cases = build_cases(experiment, ns, fixed_n, DEFAULT_PAIRS if pairs is None else pairs)
    methods_by_case = [resolve_methods(case["targets"], timing_scope) for case in cases]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = output / f"run_{stamp}"
    run_dir.mkdir(exist_ok=False)
    environment = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "numpy": np.__version__, "numba": numba.__version__,
        "platform": platform.platform(), "processor": platform.processor() or "unknown",
        "machine": platform.machine(), "numba_threads": numba.get_num_threads(),
        "dtype": "complex128", "seed": seed, "repeats": repeats, "warmups": warmups,
        "cases": cases, "timing_scope": timing_scope, "input_policy": INPUT_POLICY,
        "adapter": "PR4 bca34bf2c9284bd9473a9dec5706be66b595a2cb + PR5 7a8216fc481a8b5a2d6c04e6e0b5242dc813ab6e; pending C confirmation",
        "timing_detail": ("generic kernels via Python partial and Numba dispatch; public validation excluded"
                          if timing_scope == "kernel" else
                          "public calls via Python partial; validation and dispatch included; baseline (0,1) uses specialized path"),
        "order_policy": "baseline-first on odd rounds; fusion-first on even rounds",
        "speedup_definition": "ratio of medians, not median of per-round ratios",
        "quartile_method": "numpy.percentile linear", "argv": sys.argv,
        **source_identity(),
    }
    env_path = run_dir / "environment.json"

    def save_environment():
        env_path.write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")

    save_environment()
    raw, summary = [], []
    try:
        for case, methods in zip(cases, methods_by_case):
            n, targets = case["n_qubits"], case["targets"]
            source = random_state(n, seed)
            left, right = source.copy(), source.copy()
            methods["baseline"](left)
            methods["fusion"](right)
            error = float(np.max(np.abs(left - right)))
            if not np.allclose(left, right, rtol=1e-12, atol=1e-12):
                raise RuntimeError(f"Correctness failure: {case}; max error={error}")
            # Public entry points validate the state and targets OUTSIDE timing.
            # Also verify that timed entry points match each public operation.
            for name, actual in (("baseline", left), ("fusion", right)):
                expected = source.copy()
                METHODS[name](expected, q1=targets[0], q2=targets[1])
                if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
                    raise RuntimeError(f"Timed/public mismatch: {name}, {case}")
                del expected
            del left, right
            for method in methods.values():
                for _ in range(warmups):
                    measure_once(method, source)
            timings = {name: [] for name in methods}
            fields = {"experiment": case["experiment"], "n_qubits": n,
                      "target0": targets[0], "target1": targets[1],
                      "max_pair_index_stride": 1 << max(targets),
                      "seed": seed, "timing_scope": timing_scope}
            for i in range(repeats):
                order = list(methods) if i % 2 == 0 else list(reversed(methods))
                for position, name in enumerate(order, 1):
                    elapsed = measure_once(methods[name], source)
                    timings[name].append(elapsed)
                    raw.append({**fields, "method": name, "repeat": i + 1,
                                "position_in_round": position, "elapsed_ns": elapsed,
                                "elapsed_ms": elapsed / 1e6})
            size_bytes = source.nbytes
            row = {**fields, "n_amplitudes": source.size, "array_mib": size_bytes / 2**20,
                   "repeats": repeats, "warmups": warmups, "max_abs_error": error,
                   # Valid for CURRENT two-pass baseline and single-pass fusion.
                   # Recheck if A/B changes their algorithms. Not hardware traffic.
                   "baseline_logical_bytes": 4 * size_bytes,
                   "fusion_logical_bytes": 2 * size_bytes}
            for name, values in timings.items():
                q1, med, q3 = np.percentile(values, [25, 50, 75], method="linear") / 1e6
                row.update({f"{name}_median_ms": float(med), f"{name}_q1_ms": float(q1),
                            f"{name}_q3_ms": float(q3), f"{name}_iqr_ms": float(q3 - q1),
                            f"{name}_min_ms": min(values) / 1e6,
                            f"{name}_max_ms": max(values) / 1e6})
            fusion_ms = row["fusion_median_ms"]
            row["speedup"] = row["baseline_median_ms"] / fusion_ms if fusion_ms > 0 else float("nan")
            summary.append(row)
            # Checkpoint completed cases, keeping any partial run explicitly marked.
            write_csv(run_dir / "raw_timings.csv", raw)
            write_csv(run_dir / "summary.csv", summary)
            environment["completed_cases"] = len(summary)
            save_environment()
            print(f"{case['experiment']} n={n} targets={targets}: "
                  f"baseline={row['baseline_median_ms']:.4f}ms "
                  f"fusion={fusion_ms:.4f}ms speedup={row['speedup']:.3f}x")
        environment["status"] = "complete"
        environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_environment()
    except BaseException as exc:
        environment["status"] = "failed_or_interrupted"
        environment["error"] = f"{type(exc).__name__}: {exc}"
        save_environment()
        raise
    save_plots(run_dir, summary)
    print(f"Saved outputs: {run_dir.resolve()}")
    return run_dir


def parse_pair(value):
    try:
        pair = tuple(int(v) for v in value.split(","))
        if len(pair) != 2:
            raise ValueError
        return pair
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use two comma-separated targets, e.g. 0,5") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["A", "B", "both"], default="A")
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS)
    parser.add_argument("--fixed-n", type=int, default=20)
    parser.add_argument("--pairs", type=parse_pair, nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timing-scope", choices=["kernel", "public"], default="kernel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    try:
        run_benchmark(args.ns, args.repeats, args.seed, args.output,
                      experiment=args.experiment, fixed_n=args.fixed_n,
                      pairs=args.pairs, warmups=args.warmups, timing_scope=args.timing_scope)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
