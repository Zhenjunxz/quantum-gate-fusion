"""Cross-module acceptance of C's API and E's paired driver."""
import inspect
import json
import csv
from types import SimpleNamespace
import numpy as np
import pytest
import benchmark as b
import interfaces as c
from common import random_state
from baseline_reference import h_sequence_numpy_reference

GATES = [("H", [0]), ("H", [5])]
GROUPS = [[0, 1]]

def test_signatures_and_c_defaults():
    assert inspect.signature(c.run_benchmark) == inspect.signature(b.run_benchmark)
    assert inspect.signature(b.run_benchmark).parameters["repetitions"].default == 100
    assert inspect.signature(b.run_benchmark).parameters["warmup_iterations"].default == 10

@pytest.mark.parametrize("name", ["baseline", "fusion"])
@pytest.mark.parametrize("scope", ["kernel", "public"])
def test_c_backend_result(name, scope):
    fn = b.prepare_backend(name, 6, GATES, GROUPS, scope)
    result = c.run_benchmark(fn, 6, GATES, GROUPS, scope, 1, 2)
    assert isinstance(result, c.BenchmarkResult)
    assert result.backend_name == name
    assert result.n_qubits == 6 and result.timing_scope == scope
    assert result.max_abs_error < 1e-12
    assert result.mean_ms >= 0 and result.std_ms >= 0

def test_c_timing_excludes_warmup_and_validation(monkeypatch):
    fn = b.prepare_backend("baseline", 6, GATES, GROUPS)
    ticks = iter([0, 1000000, 10000000, 13000000])
    monkeypatch.setattr(b, "time", SimpleNamespace(perf_counter_ns=lambda: next(ticks)))
    result = b.run_benchmark(fn, 6, GATES, GROUPS, repetitions=2, warmup_iterations=2)
    assert result.mean_ms == pytest.approx(2)
    assert result.std_ms == pytest.approx(1)
    assert result.median_ms == pytest.approx(2)

def test_wrong_backend_or_wrong_scope_rejected():
    fn = b.prepare_backend("baseline", 6, GATES, GROUPS, "kernel")
    with pytest.raises(ValueError):
        b.run_benchmark(fn, 6, GATES, GROUPS, "public", 1, 1)
    with pytest.raises(AssertionError):
        b.run_benchmark(lambda state: None, 6, GATES, GROUPS, repetitions=1)

def test_pair_output_canonical_format_and_timer(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "save_plots", lambda *args: None)
    out = b.run_paired_benchmark(6, GATES, GROUPS, repeats=2, warmup_iterations=1, output=tmp_path)
    with (out / "summary.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert json.loads(row["gate_sequence"]) == [["H", [0]], ["H", [5]]]
    assert float(row["max_abs_error"]) < 1e-12
    assert "baseline_mean_ms" in row and "fusion_std_ms" in row
    assert json.loads((out / "environment.json").read_text())["timer"] == "time.perf_counter_ns"

@pytest.mark.parametrize("bad", [[("H", [0, 1]), ("H", [5])], [("H", [True]), ("H", [5])], [("H", [0]), ("H", [])]])
def test_reject_unsupported_c_gates(bad):
    with pytest.raises(ValueError):
        c.validate_case(6, bad, GROUPS)
