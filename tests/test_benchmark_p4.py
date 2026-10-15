import sys
import os
import pytest
import subprocess
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from benchmark import run_subprocess_solver
from generate_report import load_combined

def test_run_subprocess_solver_timeout(monkeypatch):
    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="test", timeout=30.0)
    monkeypatch.setattr(subprocess, "run", mock_run)
    res = run_subprocess_solver("SAT", 8, "sequential", "cadical", 30.0)
    assert res["status"] == "TIMEOUT"
    assert pd.isna(res["build_time"])
    assert pd.isna(res["solve_time"])
    assert res["time_seconds"] == 30.0

def test_run_subprocess_solver_sat_real():
    res = run_subprocess_solver("SAT", 8, "sequential", "cadical", 30.0)
    assert res["status"] in ("SAT", "SUCCESS", "OPTIMAL")
    assert res["verified"] is True
    assert res["import_time"] >= 0
    assert res["build_time"] >= 0
    assert res["solve_time"] >= 0

def test_cpsat_import_vs_build():
    pytest.importorskip("ortools")
    res = run_subprocess_solver("CP-SAT", 8, "integer", "ortools-cpsat", 30.0)
    assert res["status"] in ("SUCCESS", "OPTIMAL", "FEASIBLE")
    assert res["import_time"] >= 0
    assert res["build_time"] >= 0

def test_load_combined_old_csv(tmp_path):
    d = {
        'n': [8],
        'paradigm': ['SAT'],
        'model_variant': ['sequential'],
        'solver': ['cadical'],
        'status': ['SUCCESS'],
        'build_time': [0.1],
        'solve_time': [0.05],
        'time_seconds': [0.15]
    }
    df = pd.DataFrame(d)
    csv_path = tmp_path / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_combined(str(tmp_path))
    assert len(loaded) == 1
    assert loaded.iloc[0]['paradigm'] == 'SAT'
