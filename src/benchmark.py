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
    Runs a single solver configuration in an isolated subprocess with a hard timeout.
    Returns a dict with benchmark metrics.
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

build_start = time.time()
try:
    if paradigm == "SAT":
        from sat_model import NQueensSAT
        model = NQueensSAT(n, encoding=model_variant, solver_name=solver)
        model.build()
        build_time = time.time() - build_start
        
        solve_start = time.time()
        res = model.solve(time_limit=time_limit)
        solve_time = time.time() - solve_start
        
        queens = res.get("queens", [])
        is_sat = res.get("is_sat", False)
        status = res.get("status", "UNKNOWN")
        variables_count = res.get("variables_count", 0)
        constraints_count = res.get("clauses_count", 0)
        verified = verify(queens, n) if is_sat and queens else (status == "UNSAT")
        
    elif paradigm == "CP-SAT":
        from cpsat_model import NQueensCPSAT
        model = NQueensCPSAT(n, config=model_variant)
        model.build()
        build_time = time.time() - build_start
        
        solve_start = time.time()
        res = model.solve(time_limit=time_limit)
        solve_time = time.time() - solve_start
        
        queens = res.get("queens", [])
        is_sat = res.get("is_sat", False)
        status = res.get("status", "UNKNOWN")
        variables_count = res.get("variables_count", 0)
        constraints_count = res.get("constraints_count", 0)
        verified = verify(queens, n) if is_sat and queens else (status in ("INFEASIBLE", "MODEL_INVALID"))
        
    elif paradigm == "MIP":
        from mip_cp_models import NQueensMIP
        model = NQueensMIP(n, config=model_variant, solver_name=solver)
        model.build()
        build_time = time.time() - build_start
        
        solve_start = time.time()
        res = model.solve(time_limit=time_limit)
        solve_time = time.time() - solve_start
        
        queens = res.get("queens", [])
        is_sat = res.get("is_sat", False)
        status = res.get("status", "UNKNOWN")
        variables_count = res.get("variables_count", 0)
        constraints_count = res.get("constraints_count", 0)
        verified = verify(queens, n) if is_sat and queens else False
        
    elif paradigm == "Gurobi":
        from gurobi_model import solve_nqueens
        status, total_time, variables_count, constraints_count, queens, build_time, solve_time = solve_nqueens(n, time_limit=time_limit)
        is_sat = status in ("OPTIMAL", "SUBOPTIMAL") or len(queens) > 0
        verified = verify(queens, n) if queens else False
        
    elif paradigm == "CPLEX-MIP":
        from cplex_mip_model import solve_nqueens
        status, total_time, variables_count, constraints_count, queens, build_time, solve_time = solve_nqueens(n, time_limit=time_limit)
        is_sat = status in ("OPTIMAL", "FEASIBLE") or len(queens) > 0
        verified = verify(queens, n) if queens else False
        
    elif paradigm == "CPLEX-CP":
        from cplex_cp_model import solve_nqueens
        status, total_time, variables_count, constraints_count, queens, build_time, solve_time = solve_nqueens(n, time_limit=time_limit)
        is_sat = status in ("OPTIMAL", "FEASIBLE", "Feasible") or len(queens) > 0
        verified = verify(queens, n) if queens else False
        
    else:
        raise ValueError(f"Unknown paradigm: {{paradigm}}")

    total_time = build_time + solve_time
    out = {{
        "status": status,
        "is_sat": is_sat,
        "variables_count": variables_count,
        "constraints_count": constraints_count,
        "build_time": build_time,
        "solve_time": solve_time,
        "time_seconds": total_time,
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
        "build_time": 0.0,
        "solve_time": 0.0,
        "time_seconds": 0.0,
        "verified": False,
        "error": str(e)
    }}
    print("RESULT_JSON:" + json.dumps(out))
"""
    cmd = [sys.executable, "-c", code]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit + 10.0)
        stdout = res.stdout
        stderr = res.stderr
        
        # Check for memory error or signal
        if res.returncode == -9 or "OutOfMemoryError" in stderr or "Killed" in stderr:
            return {
                "n": n, "paradigm": paradigm, "model_variant": model_variant, "solver": solver,
                "status": "MEMOUT", "is_sat": False, "variables_count": 0, "clauses_constraints_count": 0,
                "build_time": 0.0, "solve_time": 0.0, "time_seconds": time_limit, "verified": False, "error": "Memory limit exceeded (MEMOUT)"
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
                    "build_time": data["build_time"],
                    "solve_time": data["solve_time"],
                    "time_seconds": data["time_seconds"],
                    "verified": data["verified"],
                    "error": data["error"]
                }
                
        return {
            "n": n, "paradigm": paradigm, "model_variant": model_variant, "solver": solver,
            "status": "ERROR", "is_sat": False, "variables_count": 0, "clauses_constraints_count": 0,
            "build_time": 0.0, "solve_time": 0.0, "time_seconds": 0.0, "verified": False, "error": stderr or "No JSON output"
        }
    except subprocess.TimeoutExpired:
        return {
            "n": n, "paradigm": paradigm, "model_variant": model_variant, "solver": solver,
            "status": "TIMEOUT", "is_sat": False, "variables_count": 0, "clauses_constraints_count": 0,
            "build_time": time_limit, "solve_time": time_limit, "time_seconds": time_limit, "verified": False, "error": "Hard timeout expired"
        }
    except Exception as e:
        return {
            "n": n, "paradigm": paradigm, "model_variant": model_variant, "solver": solver,
            "status": "ERROR", "is_sat": False, "variables_count": 0, "clauses_constraints_count": 0,
            "build_time": 0.0, "solve_time": 0.0, "time_seconds": 0.0, "verified": False, "error": str(e)
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
                    # If TIMEOUT or MEMOUT or LICENSE_LIMIT, no need to repeat
                    if rec["status"] in ("TIMEOUT", "MEMOUT", "LICENSE_LIMIT", "ERROR"):
                        break
                
                # If all failed or non-success, take the first or summary record
                successful = [r for r in run_records if r["status"] not in ("TIMEOUT", "MEMOUT", "ERROR", "LICENSE_LIMIT")]
                if successful:
                    # Median reporting on time
                    median_time = statistics.median([r["time_seconds"] for r in successful])
                    median_build = statistics.median([r["build_time"] for r in successful])
                    median_solve = statistics.median([r["solve_time"] for r in successful])
                    
                    rep_res = successful[0].copy()
                    rep_res["build_time"] = median_build
                    rep_res["solve_time"] = median_solve
                    rep_res["time_seconds"] = median_time
                    results.append(rep_res)
                else:
                    results.append(run_records[-1])
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

    # Expanded N values
    sat_n_values = [8, 16, 32, 64, 128]
    cpsat_n_values = [8, 16, 32, 64, 128]
    mip_n_values = [8, 16, 32, 64]
    gurobi_n_values = [8, 16, 32, 44]  # Up to free license limit N=44 (2000 vars)
    cplex_n_values = [8, 16, 31]       # Up to community edition limit N=31 (1000 vars)

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
