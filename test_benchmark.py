"""E-module timing-contract regression tests; run alongside the A/B files."""
import csv
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

import benchmark as b


@pytest.mark.parametrize("kwargs", [
    {"n_qubits": True}, {"n_qubits": 1}, {"n_qubits": 25},
    {"gate_sequence": [("H", 0)]},
    {"gate_sequence": [("X", 0), ("H", 1)]},
    {"gate_sequence": [("H", True), ("H", 1)]},
    {"gate_sequence": [("H", -1), ("H", 1)]},
    {"gate_sequence": [("H", 0), ("H", 4)]},
    {"gate_sequence": [("H", 0), ("H", 0)]},
    {"fusion_groups": []}, {"fusion_groups": [[0, 2]]},
    {"fusion_groups": [[False, 1]]}, {"fusion_groups": [[0, 1], [0, 1]]},
    {"repeats": 0}, {"repeats": True}, {"warmup_iterations": 0},
    {"warmup_iterations": 1.5}, {"seed": -1}, {"seed": True},
    {"timing_scope": "invalid"},
])
def test_invalid_parameters_rejected_before_timer_or_output(kwargs, monkeypatch, tmp_path):
    def forbidden():
        pytest.fail("Invalid arguments reached the timer")
    monkeypatch.setattr(b, "time", SimpleNamespace(perf_counter=forbidden, get_clock_info=time.get_clock_info))
    options = dict(n_qubits=4, gate_sequence=[("H", 0), ("H", 1)],
                   fusion_groups=[[0, 1]], output=tmp_path / "not_created")
    options.update(kwargs)
    with pytest.raises(ValueError):
        b.run_benchmark(**options)
    assert not options["output"].exists()


def test_copy_and_unpack_precede_timer(monkeypatch):
    events = []
    class Source:
        def copy(self):
            events.append("copy")
            return "copied"
    ticks = iter([1.0, 1.25])
    def clock():
        events.append("timer")
        return next(ticks)
    def method(state, q1, q2):
        events.append("compute")
        assert (state, q1, q2) == ("copied", 0, 1)
    monkeypatch.setattr(b, "time", SimpleNamespace(perf_counter=clock, get_clock_info=time.get_clock_info))
    assert b.measure_once(method, Source(), (0, 1)) == .25
    assert events == ["copy", "timer", "compute", "timer"]


def test_all_measurements_precede_csv_and_warmups_not_counted(monkeypatch, tmp_path):
    events = []
    original_write = b.write_csv
    original_resolve = b.resolve_methods
    measured = []
    clock_calls = 0
    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return clock_calls * .001
    calls = {"baseline": 0, "fusion": 0}
    def resolve(scope):
        methods = original_resolve(scope)
        def wrap(name, method):
            def run(state, q1, q2):
                calls[name] += 1
                events.append("compute")
                method(state, q1, q2)
            return run
        return {name: wrap(name, method) for name, method in methods.items()}
    original_measure = b.measure_once
    def measure(method, source, targets):
        measured.append(targets)
        return original_measure(method, source, targets)
    def write(path, rows):
        assert len(measured) == 12  # 2 cases x 3 repeats x 2 methods
        events.append("csv")
        original_write(path, rows)
    monkeypatch.setattr(b, "resolve_methods", resolve)
    monkeypatch.setattr(b, "measure_once", measure)
    monkeypatch.setattr(b, "write_csv", write)
    monkeypatch.setattr(b, "save_plots", lambda *args: None)
    monkeypatch.setattr(b, "time", SimpleNamespace(perf_counter=clock, get_clock_info=time.get_clock_info))
    # Default warmup_iterations=10 must not add timer calls or raw records.
    result = b.run_experiments([4, 5], repeats=3, output=tmp_path)
    assert clock_calls == 24
    assert calls == {"baseline": 28, "fusion": 28}  # 2*(1 check+10 warmups+3 measured)
    assert "compute" not in events[events.index("csv"):]
    raw = list(csv.DictReader((result / "raw_timings.csv").open()))
    summary = list(csv.DictReader((result / "summary.csv").open()))
    assert len(raw) == 12 and len(summary) == 2
    for row in raw:
        assert float(row["elapsed_seconds"]) == pytest.approx(.001)
        assert float(row["elapsed_ms"]) == pytest.approx(1)
    for row in summary:
        assert float(row["baseline_median_ms"]) == pytest.approx(1)
        assert float(row["speedup"]) == pytest.approx(1)
    env = json.loads((result / "environment.json").read_text())
    assert env["warmup_iterations"] == 10 and env["status"] == "complete"
    for n in (4, 5):
        first = [r["method"] for r in raw if int(r["n_qubits"]) == n
                 and int(r["position_in_round"]) == 1]
        assert first == ["baseline", "fusion", "baseline"]


@pytest.mark.parametrize("scope", ["kernel", "public"])
def test_unified_entry_and_nonadjacent_targets(scope, monkeypatch, tmp_path):
    monkeypatch.setattr(b, "save_plots", lambda *args: None)
    path = b.run_benchmark(6, [("H", 0), ("H", 5)], [[0, 1]], repeats=2,
                           warmup_iterations=1, timing_scope=scope, output=tmp_path)
    rows = list(csv.DictReader((path / "summary.csv").open()))
    assert len(rows) == 1
    assert rows[0]["timing_scope"] == scope
    assert json.loads(rows[0]["gate_sequence"]) == [["H", 0], ["H", 5]]
    assert json.loads(rows[0]["fusion_groups"]) == [[0, 1]]
    assert float(rows[0]["max_abs_error"]) < 1e-12
    assert int(rows[0]["baseline_logical_bytes"]) == 2 * int(rows[0]["fusion_logical_bytes"])


def test_batch_rejects_invalid_later_case_before_measurement(monkeypatch, tmp_path):
    monkeypatch.setattr(b, "time", SimpleNamespace(perf_counter=lambda: pytest.fail("timer started")))
    with pytest.raises(ValueError):
        b.run_experiments(experiment="both", pairs=[(0, 20)], output=tmp_path / "no")
    assert not (tmp_path / "no").exists()
