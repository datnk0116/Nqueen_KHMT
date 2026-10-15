"""
Robust large-scale benchmark runner for N-Queens across SAT solvers, encodings, CP-SAT models, MIP models, Gurobi, and CPLEX solvers.
Runs each benchmark in a subprocess with a hard timeout, records TIMEOUT/MEMOUT/ERROR/LICENSE_LIMIT status,
performs 3-5 repetitions with median reporting, and saves results to CSV files in results/.
"""

import os
import sys
import time
import csv
import json
import subprocess
import statistics
import argparse
import pandas as pd

current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from board import verify
from sat_model import ENCODINGS as SAT_ENCODINGS, SOLVERS as SAT_SOLVERS
from cpsat_model import CONFIGURATIONS as CPSAT_CONFIGS
from mip_cp_models import CONFIGURATIONS as MIP_CONFIGS, SOLVERS as MIP_SOLVERS

def run_subprocess_solver(paradigm, n, model_variant, solver, time_limit):
    """
    Runs a single solver configuration in an isolated subprocess with a hard timeout (time_limit)
    covering the entire process (import + build + solve).
    Returns a dict with benchmark metrics including parent-measured wall_time.
    """
    code = f"""
import sys, os, time, json
import traceback
current_dir = {repr(current_dir)}
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from board import verify

paradigm = {repr(paradigm)}
n = {n}
model_variant = {repr(model_variant)}
solver = {repr(solver)}
time_limit = {time_limit}

solver_reported_time = None
try:
    if paradigm == "SAT":
        import_start = time.perf_counter()
        from sat_model import NQueensSAT
        import_time = time.perf_counter() - import_start

        build_start = time.perf_counter()
        model = NQueensSAT(n, encoding=model_variant, solver_name=solver)
        model.build()
        build_time = time.perf_counter() - build_start
        
        elapsed_so_far = import_time + build_time
        if time_limit is not None and elapsed_so_far >= time_limit:
            solve_time = 0.0
            status = "TIME_LIMIT"
            queens = []
            is_sat = False
            variables_count = model.variables_count
            constraints_count = model.clauses_count
            verified = False
        else:
            rem_limit = max(0.001, time_limit - elapsed_so_far) if time_limit is not None else None
            solve_start = time.perf_counter()
            res = model.solve(time_limit=rem_limit)
            solve_time = time.perf_counter() - solve_start

            queens = res.get("queens", [])
            is_sat = res.get("is_sat", False) and res.get("status") != "TIME_LIMIT"
            status = res.get("status", "UNKNOWN")
            variables_count = res.get("variables_count", 0)
            constraints_count = res.get("clauses_count", 0)
            verified = verify(queens, n) if is_sat and queens else (status == "UNSAT")
        
    elif paradigm == "CP-SAT":
        import_start = time.perf_counter()
        from cpsat_model import NQueensCPSAT
        import_time = time.perf_counter() - import_start

        build_start = time.perf_counter()
        model = NQueensCPSAT(n, config=model_variant)
        model.build()
        build_time = time.perf_counter() - build_start
        
        elapsed_so_far = import_time + build_time
        if time_limit is not None and elapsed_so_far >= time_limit:
            solve_time = 0.0
            status = "TIME_LIMIT"
            queens = []
            is_sat = False
            variables_count = model.variables_count
            constraints_count = model.constraints_count
            verified = False
        else:
            rem_limit = max(0.001, time_limit - elapsed_so_far) if time_limit is not None else None
            solve_start = time.perf_counter()
            res = model.solve(time_limit=rem_limit)
            solve_time = time.perf_counter() - solve_start

            queens = res.get("queens", [])
            is_sat = res.get("is_sat", False) and res.get("status") != "TIME_LIMIT"
            status = res.get("status", "UNKNOWN")
            variables_count = res.get("variables_count", 0)
            constraints_count = res.get("constraints_count", 0)
            verified = verify(queens, n) if is_sat and queens else (status in ("INFEASIBLE", "MODEL_INVALID"))
        
    elif paradigm == "MIP":
        import_start = time.perf_counter()
        from mip_cp_models import NQueensMIP
        import_time = time.perf_counter() - import_start

        build_start = time.perf_counter()
        model = NQueensMIP(n, config=model_variant, solver_name=solver)
        model.build()
        build_time = time.perf_counter() - build_start
        
        elapsed_so_far = import_time + build_time
        if time_limit is not None and elapsed_so_far >= time_limit:
            solve_time = 0.0
            status = "TIME_LIMIT"
            queens = []
            is_sat = False
            variables_count = model.variables_count
            constraints_count = model.constraints_count
            verified = False
        else:
            rem_limit = max(0.001, time_limit - elapsed_so_far) if time_limit is not None else None
            solve_start = time.perf_counter()
            res = model.solve(time_limit=rem_limit)
            solve_time = time.perf_counter() - solve_start

            queens = res.get("queens", [])
            is_sat = res.get("is_sat", False) and res.get("status") != "TIME_LIMIT"
            status = res.get("status", "UNKNOWN")
            variables_count = res.get("variables_count", 0)
            constraints_count = res.get("constraints_count", 0)
            verified = verify(queens, n) if is_sat and queens else False
        
    elif paradigm == "Gurobi":
        import_start = time.perf_counter()
        from gurobi_model import solve_nqueens
        import_time = time.perf_counter() - import_start
        rem_limit = max(0.001, time_limit - import_time) if time_limit is not None else None
        res = solve_nqueens(n, time_limit=rem_limit)
        status = res["status"]
        total_time = res["total_time"]
        variables_count = res["variables_count"]
        constraints_count = res["constraints_count"]
        queens = res["queens"]
        build_time = res["build_time"]
        solve_time = res["solve_time"]
        solver_reported_time = res["solver_reported_time"]
        is_sat = (status in ("OPTIMAL", "SUBOPTIMAL") or len(queens) > 0) and status != "TIME_LIMIT"
        verified = verify(queens, n) if queens and is_sat else False
        
    elif paradigm == "CPLEX-MIP":
        import_start = time.perf_counter()
        from cplex_mip_model import solve_nqueens
        import_time = time.perf_counter() - import_start
        rem_limit = max(0.001, time_limit - import_time) if time_limit is not None else None
        res = solve_nqueens(n, time_limit=rem_limit)
        status = res["status"]
        total_time = res["total_time"]
        variables_count = res["variables_count"]
        constraints_count = res["constraints_count"]
        queens = res["queens"]
        build_time = res["build_time"]
        solve_time = res["solve_time"]
        solver_reported_time = res["solver_reported_time"]
        is_sat = (status in ("OPTIMAL", "FEASIBLE") or len(queens) > 0) and status != "TIME_LIMIT"
        verified = verify(queens, n) if queens and is_sat else False
        
    elif paradigm == "CPLEX-CP":
        import_start = time.perf_counter()
        from cplex_cp_model import solve_nqueens
        import_time = time.perf_counter() - import_start
        rem_limit = max(0.001, time_limit - import_time) if time_limit is not None else None
        res = solve_nqueens(n, time_limit=rem_limit)
        status = res["status"]
        total_time = res["total_time"]
        variables_count = res["variables_count"]
        constraints_count = res["constraints_count"]
        queens = res["queens"]
        build_time = res["build_time"]
        solve_time = res["solve_time"]
        solver_reported_time = res["solver_reported_time"]
        is_sat = (status in ("OPTIMAL", "FEASIBLE", "Feasible") or len(queens) > 0) and status != "TIME_LIMIT"
        verified = verify(queens, n) if queens and is_sat else False
        
    else:
        raise ValueError(f"Unknown paradigm: {{paradigm}}")

    total_time = import_time + build_time + solve_time
    if time_limit is not None and total_time > time_limit and status not in ("TIMEOUT", "ERROR", "MEMOUT", "LICENSE_LIMIT"):
        status = "TIMEOUT"
        build_time = float('nan')
        solve_time = float('nan')
        import_time = float('nan')
        total_time = time_limit

    out = {{
        "status": status,
        "is_sat": is_sat,
        "variables_count": variables_count,
        "constraints_count": constraints_count,
        "import_time": import_time,
        "build_time": build_time,
        "solve_time": solve_time,
        "time_seconds": total_time,
        "solver_reported_time": solver_reported_time,
        "verified": verified,
        "error": "" if status != "LICENSE_LIMIT" else "License/size limit reached"
    }}
    print("RESULT_JSON:" + json.dumps(out))
except Exception as e:
    err_str = str(e).lower()
    status = "LICENSE_LIMIT" if ("license" in err_str or "size limit" in err_str or "trial" in err_str or "community limit" in err_str) else "ERROR"
    out = {{
        "status": status,
        "is_sat": False,
        "variables_count": 0,
        "constraints_count": 0,
        "import_time": float('nan'),
        "build_time": float('nan'),
        "solve_time": float('nan'),
        "time_seconds": 0.0,
        "solver_reported_time": None,
        "verified": False,
        "error": str(e)
    }}
    print("RESULT_JSON:" + json.dumps(out))
"""
    cmd = [sys.executable, "-c", code]
    p_start = time.perf_counter()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit)
        p_wall = time.perf_counter() - p_start
        stdout = res.stdout
        stderr = res.stderr
        
        if res.returncode == -9 or "OutOfMemoryError" in stderr or "Killed" in stderr:
            return {
                "n": n, "paradigm": paradigm, "model_variant": model_variant, "solver": solver,
                "status": "MEMOUT", "is_sat": False, "variables_count": 0, "clauses_constraints_count": 0,
                "import_time": float('nan'), "build_time": float('nan'), "solve_time": float('nan'),
                "time_seconds": time_limit, "solver_reported_time": None, "wall_time": p_wall,
                "verified": False, "error": "Memory limit exceeded (MEMOUT)"
            }

        for line in stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                data = json.loads(line[len("RESULT_JSON:"):])
                return {
                    "n": n,
                    "paradigm": paradigm,
                    "model_variant": model_variant,
                    "solver": solver,
                    "status": data["status"],
                    "is_sat": data["is_sat"],
                    "variables_count": data["variables_count"],
                    "clauses_constraints_count": data["constraints_count"],
                    "import_time": data["import_time"],
                    "build_time": data["build_time"],
                    "solve_time": data["solve_time"],
                    "time_seconds": data["time_seconds"],
                    "solver_reported_time": data["solver_reported_time"],
                    "wall_time": p_wall,
                    "verified": data["verified"],
                    "error": data["error"]
                }
                
        return {
            "n": n, "paradigm": paradigm, "model_variant": model_variant, "solver": solver,
            "status": "ERROR", "is_sat": False, "variables_count": 0, "clauses_constraints_count": 0,
            "import_time": float('nan'), "build_time": float('nan'), "solve_time": float('nan'),
            "time_seconds": 0.0, "solver_reported_time": None, "wall_time": p_wall,
            "verified": False, "error": stderr or "No JSON output"
        }
    except subprocess.TimeoutExpired:
        p_wall = time.perf_counter() - p_start
        return {
            "n": n, "paradigm": paradigm, "model_variant": model_variant, "solver": solver,
            "status": "TIMEOUT", "is_sat": False, "variables_count": 0, "clauses_constraints_count": 0,
            "import_time": float('nan'), "build_time": float('nan'), "solve_time": float('nan'),
            "time_seconds": time_limit, "solver_reported_time": None, "wall_time": p_wall,
            "verified": False, "error": "Hard timeout expired"
        }
    except Exception as e:
        p_wall = time.perf_counter() - p_start
        return {
            "n": n, "paradigm": paradigm, "model_variant": model_variant, "solver": solver,
            "status": "ERROR", "is_sat": False, "variables_count": 0, "clauses_constraints_count": 0,
            "import_time": float('nan'), "build_time": float('nan'), "solve_time": float('nan'),
            "time_seconds": 0.0, "solver_reported_time": None, "wall_time": p_wall,
            "verified": False, "error": str(e)
        }

def run_benchmarks_with_repetitions(paradigm, n_values, encodings_or_configs, solvers, time_limit, repetitions=3):
    results = []
    for n in n_values:
        for variant in encodings_or_configs:
            for solver in solvers:
                print(f"[{paradigm}] N={n}, variant={variant}, solver={solver} (running {repetitions} repetitions)...")
                run_records = []
                for rep in range(repetitions):
                    rec = run_subprocess_solver(paradigm, n, variant, solver, time_limit)
                    run_records.append(rec)
                    # For LICENSE_LIMIT, repeating is redundant because license size limit is deterministic.
                    if rec["status"] == "LICENSE_LIMIT":
                        while len(run_records) < repetitions:
                            run_records.append(rec.copy())
                        break

                runs_total = len(run_records)
                runs_success = sum(1 for r in run_records if r["status"] in ("OPTIMAL", "FEASIBLE", "SAT"))
                runs_timeout = sum(1 for r in run_records if r["status"] in ("TIMEOUT", "TIME_LIMIT"))
                runs_memout = sum(1 for r in run_records if r["status"] == "MEMOUT")
                runs_license = sum(1 for r in run_records if r["status"] == "LICENSE_LIMIT")
                runs_error = sum(1 for r in run_records if r["status"] == "ERROR")

                successful = [r for r in run_records if r["status"] in ("OPTIMAL", "FEASIBLE", "SAT")]

                # Determine aggregate status
                if runs_license > 0:
                    status = "LICENSE_LIMIT"
                    is_sat = False
                    verified = False
                    error = "License/size limit reached"
                elif runs_memout > 0:
                    status = "MEMOUT"
                    is_sat = False
                    verified = False
                    error = "Memory limit exceeded (MEMOUT)"
                elif runs_error == runs_total:
                    status = "ERROR"
                    is_sat = False
                    verified = False
                    error = run_records[-1].get("error", "Error")
                elif runs_timeout >= (runs_total / 2.0) or (runs_success == 0 and runs_timeout > 0):
                    soft_timeouts = sum(1 for r in run_records if r["status"] == "TIME_LIMIT")
                    status = "TIME_LIMIT" if soft_timeouts == runs_timeout else "TIMEOUT"
                    is_sat = False
                    verified = False
                    error = f"{runs_timeout}/{runs_total} runs timed out"
                elif runs_success > 0:
                    status = successful[0]["status"]
                    is_sat = successful[0]["is_sat"]
                    verified = successful[0]["verified"]
                    error = f"{runs_timeout}/{runs_total} runs timed out" if runs_timeout > 0 else ""
                else:
                    status = run_records[-1]["status"]
                    is_sat = False
                    verified = False
                    error = run_records[-1].get("error", "")

                all_times = [r["time_seconds"] for r in run_records if not pd.isna(r["time_seconds"])]
                all_walls = [r["wall_time"] for r in run_records if ("wall_time" in r and not pd.isna(r["wall_time"]))]
                
                median_time = statistics.median(all_times) if all_times else time_limit
                median_wall = statistics.median(all_walls) if all_walls else median_time

                if status in ("TIMEOUT", "TIME_LIMIT", "MEMOUT", "ERROR", "LICENSE_LIMIT") and runs_success == 0:
                    median_import = float('nan')
                    median_build = float('nan')
                    median_solve = float('nan')
                    reported_times = [r["solver_reported_time"] for r in run_records if r.get("solver_reported_time") is not None and not pd.isna(r.get("solver_reported_time"))]
                    median_reported = statistics.median(reported_times) if reported_times else None
                    base_rec = run_records[0]
                    variables_count = base_rec["variables_count"] if status == "LICENSE_LIMIT" else 0
                    constraints_count = base_rec["clauses_constraints_count"] if status == "LICENSE_LIMIT" else 0
                else:
                    import_times = [r["import_time"] for r in successful if not pd.isna(r["import_time"])]
                    build_times = [r["build_time"] for r in successful if not pd.isna(r["build_time"])]
                    solve_times = [r["solve_time"] for r in successful if not pd.isna(r["solve_time"])]
                    reported_times = [r["solver_reported_time"] for r in run_records if r.get("solver_reported_time") is not None and not pd.isna(r.get("solver_reported_time"))]

                    median_import = statistics.median(import_times) if import_times else float('nan')
                    median_build = statistics.median(build_times) if build_times else float('nan')
                    median_solve = statistics.median(solve_times) if solve_times else float('nan')
                    median_reported = statistics.median(reported_times) if reported_times else None
                    base_rec = successful[0]
                    variables_count = base_rec["variables_count"]
                    constraints_count = base_rec["clauses_constraints_count"]

                rep_res = {
                    "n": n,
                    "paradigm": paradigm,
                    "model_variant": variant,
                    "solver": solver,
                    "status": status,
                    "is_sat": is_sat,
                    "variables_count": variables_count,
                    "clauses_constraints_count": constraints_count,
                    "import_time": median_import,
                    "build_time": median_build,
                    "solve_time": median_solve,
                    "time_seconds": median_time,
                    "solver_reported_time": median_reported,
                    "wall_time": median_wall,
                    "runs_total": runs_total,
                    "runs_success": runs_success,
                    "runs_timeout": runs_timeout,
                    "runs_error": runs_error,
                    "verified": verified,
                    "error": error
                }
                results.append(rep_res)
    return results

def main():
    parser = argparse.ArgumentParser(description="Run large-scale N-Queens benchmarks with subprocess isolation and repetitions.")
    parser.add_argument("--time-limit", type=float, default=30.0, help="Time limit in seconds per run")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions for median reporting")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save CSV results")
    args = parser.parse_args()

    output_dir = os.path.join(current_dir, "..", args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sat_encodings = list(SAT_ENCODINGS.keys())
    sat_solvers = [s for s in ["glucose3", "minisat22", "cadical", "kissat", "maplecm"] if s in SAT_SOLVERS]
    cpsat_configs = ["integer", "boolean", "allowed"]
    mip_configs = ["binary", "integer"]
    mip_solvers = ["cbc", "scip"]

    sat_n_values = [8, 16, 32, 64, 128]
    cpsat_n_values = [8, 16, 32, 64, 128]
    mip_n_values = [8, 16, 32, 64]
    gurobi_n_values = [8, 16, 32, 44]
    cplex_n_values = [8, 16, 32, 64, 128]

    print("=== Running SAT Benchmarks ===")
    sat_results = run_benchmarks_with_repetitions("SAT", sat_n_values, sat_encodings, sat_solvers, args.time_limit, args.repetitions)

    print("=== Running CP-SAT Benchmarks ===")
    cpsat_results = run_benchmarks_with_repetitions("CP-SAT", cpsat_n_values, cpsat_configs, ["ortools-cpsat"], args.time_limit, args.repetitions)

    print("=== Running MIP Benchmarks ===")
    mip_results = run_benchmarks_with_repetitions("MIP", mip_n_values, mip_configs, mip_solvers, args.time_limit, args.repetitions)

    print("=== Running Gurobi Benchmarks ===")
    gurobi_results = run_benchmarks_with_repetitions("Gurobi", gurobi_n_values, ["binary"], ["gurobi"], args.time_limit, args.repetitions)

    print("=== Running CPLEX MIP Benchmarks ===")
    cplex_mip_results = run_benchmarks_with_repetitions("CPLEX-MIP", cplex_n_values, ["binary"], ["cplex"], args.time_limit, args.repetitions)

    print("=== Running CPLEX CP Benchmarks ===")
    cplex_cp_results = run_benchmarks_with_repetitions("CPLEX-CP", cplex_n_values, ["integer"], ["cplex-cp"], args.time_limit, args.repetitions)

    all_results = sat_results + cpsat_results + mip_results + gurobi_results + cplex_mip_results + cplex_cp_results
    df = pd.DataFrame(all_results)

    combined_csv = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(combined_csv, index=False)
    print(f"\nSaved all benchmark results to {combined_csv}")

    pd.DataFrame(sat_results).to_csv(os.path.join(output_dir, "sat_results.csv"), index=False)
    pd.DataFrame(cpsat_results).to_csv(os.path.join(output_dir, "cpsat_results.csv"), index=False)
    pd.DataFrame(mip_results).to_csv(os.path.join(output_dir, "mip_results.csv"), index=False)
    pd.DataFrame(gurobi_results).to_csv(os.path.join(output_dir, "gurobi_results.csv"), index=False)
    pd.DataFrame(cplex_mip_results + cplex_cp_results).to_csv(os.path.join(output_dir, "cplex_results.csv"), index=False)

    print(f"Summary generated: {len(all_results)} total benchmark configurations evaluated.")

if __name__ == "__main__":
    main()
