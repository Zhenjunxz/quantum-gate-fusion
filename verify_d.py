"""Generate D-module correctness acceptance JSON (NOT a performance benchmark).

Usage:
    python verify_d.py --mode quick --output results/d_quick.json
    python verify_d.py --mode full  --output results/d_full.json

Full mode includes all nine configurations from the team's experiment plan.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

import baseline
import fusion
from baseline import run_baseline_inplace
from fusion import run_fusion_inplace
from d_oracle import dense_expected, expected_amplitude, sample_output_indices

ATOL = 1e-12
RTOL = 1e-12

QUICK_CASES = [(2,0,1), (3,0,2), (4,3,1), (6,0,5), (8,7,0),
               (10,0,9), (12,0,1), (12,0,5)]
FULL_CASES = [(n,0,1) for n in (12,14,16,18,20,22)] + [
    (20,0,1), (20,0,5), (20,0,10),
]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_info():
    def command(*args):
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                           text=True, timeout=2).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"head": command("rev-parse", "HEAD"),
            "baseline_pr4": command("rev-parse", "review/a-pr4"),
            "fusion_pr5": command("rev-parse", "review/b-pr5")}


def make_state(n, seed):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(
        rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n),
        dtype=np.complex128)
    x /= np.linalg.norm(x)
    return x


def measure_case(n, q1, q2, seed):
    """Only compute numerical correctness metrics; do not time anything."""
    source = make_state(n, seed)
    saved = source.copy()
    a, b = source.copy(), source.copy()
    assert run_baseline_inplace(a, q1=q1, q2=q2) is None
    assert run_fusion_inplace(b, q1=q1, q2=q2) is None
    np.testing.assert_array_equal(source, saved)
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
    direct_err = float(np.max(np.abs(a - b)))
    samples = sample_output_indices(n, q1, q2, count=128, seed=seed + 99)
    ref_err_a = 0.0
    ref_err_b = 0.0
    for j in samples:
        truth = expected_amplitude(source, int(j), q1, q2)
        ref_err_a = max(ref_err_a, abs(a[j] - truth))
        ref_err_b = max(ref_err_b, abs(b[j] - truth))
        np.testing.assert_allclose(a[j], truth, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(b[j], truth, rtol=RTOL, atol=ATOL)
    dense_err = None
    if n <= 6:
        truth = dense_expected(source, q1, q2)
        dense_err = float(max(np.max(np.abs(a - truth)), np.max(np.abs(b - truth))))
        np.testing.assert_allclose(a, truth, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(b, truth, rtol=RTOL, atol=ATOL)
    norm_err = float(max(abs(np.linalg.norm(a) - 1.0), abs(np.linalg.norm(b) - 1.0)))
    assert norm_err < 1e-12
    return {
        "n_qubits": n, "q1": q1, "q2": q2, "seed": seed,
        "amplitude_count": int(source.size),
        "state_bytes": int(source.nbytes),
        "sampled_amplitudes": int(len(samples)),
        "max_baseline_fusion_abs_error": direct_err,
        "max_baseline_oracle_sample_abs_error": float(ref_err_a),
        "max_fusion_oracle_sample_abs_error": float(ref_err_b),
        "max_dense_oracle_abs_error": dense_err,
        "max_norm_error": norm_err,
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = QUICK_CASES if args.mode == "quick" else FULL_CASES
    report = {
        "schema_version": 1,
        "module": "D - cross-PR numerical correctness acceptance",
        "mode": args.mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(), "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "source": {"git": git_info(),
                   "baseline_sha256": file_sha256(baseline.__file__),
                   "fusion_sha256": file_sha256(fusion.__file__)},
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "warning": "Correctness only. Sparse oracle samples selected indices, not all. "
                   "No runtime timing, actual CPU cache miss or DRAM traffic is measured.",
        "cases": [], "status": "passed",
    }
    try:
        for n, q1, q2 in cases:
            print(f"Checking n={n:2d}, targets=({q1},{q2}) ...", flush=True)
            case = measure_case(n, q1, q2, seed=42)
            report["cases"].append(case)
            print(f"  PASS, A-B maximum abs error {case['max_baseline_fusion_abs_error']:.3e}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    report["case_count"] = len(report["cases"])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.output}")
    print(f"D acceptance: {report['status'].upper()} ({report['case_count']} cases completed)")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
