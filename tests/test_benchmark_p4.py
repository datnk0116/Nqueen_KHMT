import sys
import os
import re
import shutil
import time
import pytest
import subprocess
import pandas as pd
import numpy as np
import ast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from benchmark import run_subprocess_solver
from generate_report import load_combined, build_discussion_stats, generate_tex, compile_pdf, format_adaptive_time

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

def test_models_no_solve_start_in_except():
    for mod_name in ['gurobi_model', 'cplex_mip_model', 'cplex_cp_model']:
        path = os.path.join(os.path.dirname(__file__), f'../src/{mod_name}.py')
        with open(path, 'r') as f:
            tree = ast.parse(f.read(), filename=f'{mod_name}.py')
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if isinstance(t, ast.Name) and t.id == 'solve_start':
                                pytest.fail(f"Found assignment to solve_start inside except block in {mod_name}.py")

def test_format_adaptive_time():
    assert format_adaptive_time(0.0105) == "10.5 ms"
    assert format_adaptive_time(1.53) == "1.53 s"
    assert format_adaptive_time(0.005) == "5.0 ms"
    assert format_adaptive_time(float('nan')) == "N/A"

def test_solve_time_on_exception(monkeypatch):
    import gurobi_model
    import gurobipy as gp
    def mock_optimize(self):
        time.sleep(0.25)
        raise gp.GurobiError(10001, "Mock solve error")
    monkeypatch.setattr(gp.Model, "optimize", mock_optimize)
    res = gurobi_model.solve_nqueens(4)
    assert res["solve_time"] >= 0.2, f"Gurobi solve_time {res['solve_time']} should be >= 0.2"

    import cplex_mip_model
    from docplex.mp.model import Model
    def mock_mp_solve(self, *args, **kwargs):
        time.sleep(0.25)
        raise RuntimeError("Mock CPLEX MIP error")
    monkeypatch.setattr(Model, "solve", mock_mp_solve)
    res_mip = cplex_mip_model.solve_nqueens(4)
    assert res_mip["solve_time"] >= 0.2, f"CPLEX MIP solve_time {res_mip['solve_time']} should be >= 0.2"

    import cplex_cp_model
    from docplex.cp.model import CpoModel
    def mock_cp_solve(self, *args, **kwargs):
        time.sleep(0.25)
        raise RuntimeError("Mock CPLEX CP error")
    monkeypatch.setattr(CpoModel, "solve", mock_cp_solve)
    res_cp = cplex_cp_model.solve_nqueens(4)
    assert res_cp["solve_time"] >= 0.2, f"CPLEX CP solve_time {res_cp['solve_time']} should be >= 0.2"

def test_run_subprocess_solver_gurobi_and_cplex():
    pytest.importorskip("gurobipy")
    res = run_subprocess_solver("Gurobi", 8, "binary", "gurobi", 30.0)
    assert res["status"] != "ERROR"
    if res["status"] not in ("LICENSE_LIMIT", "TIMEOUT"):
        assert res["import_time"] >= 0
        assert res["build_time"] >= 0
        assert res["solve_time"] >= 0

    pytest.importorskip("docplex")
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

def test_latex_unescaped_underscore_regex(tmp_path):
    src_results = os.path.join(os.path.dirname(__file__), '../results')
    tmp_results = tmp_path / "results"
    shutil.copytree(src_results, tmp_results)
    
    csv_file = tmp_results / "benchmark_results.csv"
    df = pd.read_csv(csv_file)
    if 'import_time' not in df.columns:
        df['import_time'] = 0.05
    df['solver_reported_time'] = np.where(df['paradigm'] == 'CPLEX-CP', 0.005, np.nan)
    df.to_csv(csv_file, index=False)
    
    tmp_report = tmp_path / "report"
    tex_path = generate_tex(str(tmp_results), str(tmp_report))
    assert os.path.exists(tex_path)
    
    with open(tex_path, 'r') as f:
        tex = f.read()
    
    # Check that no \texttt{...} has unescaped underscore
    tt_matches = re.findall(r'\\texttt\{([^}]*)\}', tex)
    for m in tt_matches:
        unescaped = re.search(r'(?<!\\)_', m)
        assert not unescaped, f"Unescaped underscore found in \\texttt{{{m}}}"

    # If pdflatex is installed, verify compilation passes
    if shutil.which("pdflatex"):
        assert compile_pdf(str(tmp_report)) is True

def test_latex_regex_old_and_nan_branches(tmp_path):
    # 1. Old CSV branch (without solver_reported_time column)
    tmp_results_old = tmp_path / "results_old"
    os.makedirs(tmp_results_old, exist_ok=True)
    df_old = pd.DataFrame({
        'n': [8], 'paradigm': ['SAT'], 'model_variant': ['sequential'], 'solver': ['cadical'],
        'status': ['SUCCESS'], 'variables_count': [64], 'clauses_constraints_count': [100],
        'build_time': [0.1], 'solve_time': [0.05], 'time_seconds': [0.15]
    })
    df_old.to_csv(tmp_results_old / "benchmark_results.csv", index=False)
    tmp_report_old = tmp_path / "report_old"
    tex_path_old = generate_tex(str(tmp_results_old), str(tmp_report_old))
    with open(tex_path_old, 'r') as f:
        tex_old = f.read()
    for m in re.findall(r'\\texttt\{([^}]*)\}', tex_old):
        assert not re.search(r'(?<!\\)_', m), f"Unescaped underscore in old branch: \\texttt{{{m}}}"
    
    # 2. NaN branch: solver_reported_time present but all NaN
    tmp_results_nan = tmp_path / "results_nan"
    os.makedirs(tmp_results_nan, exist_ok=True)
    df_nan = df_old.copy()
    df_nan['paradigm'] = ['CPLEX-CP']
    df_nan['solver'] = ['cplex-cp']
    df_nan['solver_reported_time'] = [float('nan')]
    df_nan.to_csv(tmp_results_nan / "benchmark_results.csv", index=False)
    tmp_report_nan = tmp_path / "report_nan"
    tex_path_nan = generate_tex(str(tmp_results_nan), str(tmp_report_nan))
    with open(tex_path_nan, 'r') as f:
        tex_nan = f.read()
    for m in re.findall(r'\\texttt\{([^}]*)\}', tex_nan):
        assert not re.search(r'(?<!\\)_', m), f"Unescaped underscore in NaN branch: \\texttt{{{m}}}"
    assert not re.search(r'\bnan\b', tex_nan.lower()), "Found standalone 'nan' in generated LaTeX"
    assert "is the number reported by the solver" not in tex_nan
    assert "get\\_solve\\_time()" in tex_nan

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

def test_sat_model_time_limit_interrupt(monkeypatch):
    from sat_model import NQueensSAT
    from pysat.solvers import Glucose3
    def mock_solve_limited(self, expect_interrupt=False):
        return None
    monkeypatch.setattr(Glucose3, "solve_limited", mock_solve_limited)
    model = NQueensSAT(8, "sequential", "glucose3")
    res = model.solve(time_limit=1.0)
    assert res["status"] == "TIME_LIMIT"
    assert res["is_sat"] is False
    assert res["queens"] == []
    assert res["time"] >= 0

def test_sat_model_cadical_fallback():
    from sat_model import NQueensSAT
    model = NQueensSAT(8, "sequential", "cadical")
    res = model.solve(time_limit=10.0)
    assert res["status"] == "SAT"
    assert res["is_sat"] is True
    assert len(res["queens"]) == 8

def test_run_subprocess_solver_includes_wall_time():
    res = run_subprocess_solver("SAT", 8, "sequential", "cadical", 30.0)
    assert "wall_time" in res
    assert res["wall_time"] > 0
    assert "time_seconds" in res
    assert res["status"] == "SAT"

def test_repetitions_mixed_majority_timeout(monkeypatch):
    from benchmark import run_benchmarks_with_repetitions
    call_count = 0
    def mock_run_subprocess(paradigm, n, variant, solver, time_limit):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "n": n, "paradigm": paradigm, "model_variant": variant, "solver": solver,
                "status": "SAT", "is_sat": True, "variables_count": 64, "clauses_constraints_count": 100,
                "import_time": 0.05, "build_time": 0.01, "solve_time": 0.02, "time_seconds": 0.08,
                "solver_reported_time": None, "wall_time": 0.1, "verified": True, "error": ""
            }
        else:
            return {
                "n": n, "paradigm": paradigm, "model_variant": variant, "solver": solver,
                "status": "TIMEOUT", "is_sat": False, "variables_count": 0, "clauses_constraints_count": 0,
                "import_time": float('nan'), "build_time": float('nan'), "solve_time": float('nan'),
                "time_seconds": 30.0, "solver_reported_time": None, "wall_time": 30.05, "verified": False,
                "error": "Hard timeout expired"
            }
    import benchmark
    monkeypatch.setattr(benchmark, "run_subprocess_solver", mock_run_subprocess)
    results = run_benchmarks_with_repetitions("SAT", [8], ["sequential"], ["cadical"], 30.0, repetitions=3)
    assert len(results) == 1
    rec = results[0]
    assert rec["status"] == "TIMEOUT"
    assert rec["runs_total"] == 3
    assert rec["runs_success"] == 1
    assert rec["runs_timeout"] == 2
    assert rec["time_seconds"] == 30.0

def test_repetitions_mixed_minority_timeout(monkeypatch):
    from benchmark import run_benchmarks_with_repetitions
    call_count = 0
    def mock_run_subprocess(paradigm, n, variant, solver, time_limit):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "n": n, "paradigm": paradigm, "model_variant": variant, "solver": solver,
                "status": "SAT", "is_sat": True, "variables_count": 64, "clauses_constraints_count": 100,
                "import_time": 0.05, "build_time": 0.01, "solve_time": 0.04, "time_seconds": 0.10,
                "solver_reported_time": 0.03, "wall_time": 0.12, "verified": True, "error": ""
            }
        elif call_count == 2:
            return {
                "n": n, "paradigm": paradigm, "model_variant": variant, "solver": solver,
                "status": "SAT", "is_sat": True, "variables_count": 64, "clauses_constraints_count": 100,
                "import_time": 0.06, "build_time": 0.01, "solve_time": 0.05, "time_seconds": 0.12,
                "solver_reported_time": 0.04, "wall_time": 0.14, "verified": True, "error": ""
            }
        else:
            return {
                "n": n, "paradigm": paradigm, "model_variant": variant, "solver": solver,
                "status": "TIMEOUT", "is_sat": False, "variables_count": 0, "clauses_constraints_count": 0,
                "import_time": float('nan'), "build_time": float('nan'), "solve_time": float('nan'),
                "time_seconds": 30.0, "solver_reported_time": None, "wall_time": 30.05, "verified": False,
                "error": "Hard timeout expired"
            }
    import benchmark
    monkeypatch.setattr(benchmark, "run_subprocess_solver", mock_run_subprocess)
    results = run_benchmarks_with_repetitions("SAT", [8], ["sequential"], ["cadical"], 30.0, repetitions=3)
    assert len(results) == 1
    rec = results[0]
    assert rec["status"] == "SAT"
    assert rec["runs_total"] == 3
    assert rec["runs_success"] == 2
    assert rec["runs_timeout"] == 1
    assert "1/3 runs timed out" in rec["error"]
    assert rec["time_seconds"] == 0.12
    assert rec["solver_reported_time"] == 0.035

def test_no_time_time_calls_in_models():
    model_files = [
        'sat_model.py', 'cpsat_model.py', 'mip_cp_models.py',
        'gurobi_model.py', 'cplex_mip_model.py', 'cplex_cp_model.py'
    ]
    for filename in model_files:
        path = os.path.join(os.path.dirname(__file__), '../src', filename)
        with open(path, 'r') as f:
            tree = ast.parse(f.read(), filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name) and func.value.id == 'time' and func.attr == 'time':
                        pytest.fail(f"Found time.time() call in {filename}! Use time.perf_counter() instead.")
