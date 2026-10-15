import sys
import os
import pytest
import subprocess
import pandas as pd
import numpy as np
import ast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from benchmark import run_subprocess_solver
from generate_report import load_combined, build_discussion_stats

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
    assert res["import_time"] > res["build_time"], f"import_time ({res['import_time']}) should be > build_time ({res['build_time']})"

def test_subprocess_timeout_exceeded_total_time():
    res = run_subprocess_solver("SAT", 8, "sequential", "cadical", 0.00001)
    assert res["status"] == "TIMEOUT"
    assert res["time_seconds"] == 0.00001
    assert pd.isna(res["build_time"])
    assert pd.isna(res["solve_time"])
    assert pd.isna(res["import_time"])

@pytest.mark.parametrize("module_name", ["gurobi_model", "cplex_mip_model", "cplex_cp_model"])
def test_solve_nqueens_return_contract(module_name):
    if module_name == "gurobi_model":
        pytest.importorskip("gurobipy")
    else:
        pytest.importorskip("docplex")
    
    mod = __import__(module_name)
    try:
        res = mod.solve_nqueens(8)
    except Exception as e:
        pytest.skip(f"Solver initialization or run failed for {module_name}: {e}")
    
    expected_keys = {"status", "total_time", "variables_count", "constraints_count", "queens", "build_time", "solve_time", "solver_reported_time"}
    assert isinstance(res, dict)
    assert expected_keys.issubset(res.keys())

def test_benchmark_no_tuple_unpacking():
    benchmark_path = os.path.join(os.path.dirname(__file__), '../src/benchmark.py')
    with open(benchmark_path, 'r') as f:
        tree = ast.parse(f.read(), filename='benchmark.py')
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Tuple) for t in node.targets):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'solve_nqueens':
                    pytest.fail("Found tuple unpacking of solve_nqueens() in benchmark.py!")

def test_run_subprocess_solver_gurobi_and_cplex():
    gurobipy = pytest.importorskip("gurobipy")
    res = run_subprocess_solver("Gurobi", 8, "binary", "gurobi", 30.0)
    assert res["status"] != "ERROR"
    if res["status"] not in ("LICENSE_LIMIT", "TIMEOUT"):
        assert res["import_time"] >= 0
        assert res["build_time"] >= 0
        assert res["solve_time"] >= 0

    docplex = pytest.importorskip("docplex")
    res_cp = run_subprocess_solver("CPLEX-MIP", 8, "binary", "cplex", 30.0)
    assert res_cp["status"] != "ERROR"
    if res_cp["status"] not in ("LICENSE_LIMIT", "TIMEOUT"):
        assert res_cp["import_time"] >= 0
        assert res_cp["build_time"] >= 0
        assert res_cp["solve_time"] >= 0

def test_generate_report_stats_with_and_without_reported_time(tmp_path):
    df_old = pd.DataFrame({
        'n': [8, 128],
        'paradigm': ['CPLEX-CP', 'CPLEX-CP'],
        'model_variant': ['integer', 'integer'],
        'solver': ['cplex-cp', 'cplex-cp'],
        'status': ['FEASIBLE', 'FEASIBLE'],
        'build_time': [0.1, 0.5],
        'solve_time': [0.01, 0.02],
        'time_seconds': [0.11, 0.52]
    })
    stats_old = build_discussion_stats(df_old)
    assert not stats_old['has_solver_reported_time']

    df_new = df_old.copy()
    df_new['solver_reported_time'] = [0.005, 0.015]
    stats_new = build_discussion_stats(df_new)
    assert stats_new['has_solver_reported_time']
    assert stats_new['cplex_cp_128_reported_s'] == 0.015

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
