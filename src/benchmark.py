"""
Benchmark runner for N-Queens across SAT solvers, encodings, CP-SAT models, and MIP models.
Measures execution time, status, variable counts, clause/constraint counts, and saves results to CSV.
"""

import os
import sys
import time
import csv
import argparse
import pandas as pd

# Add current directory and src to sys.path
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from sat_model import solve_nqueens as solve_sat, ENCODINGS, SOLVERS as SAT_SOLVERS
from cpsat_model import solve_nqueens as solve_cpsat, CONFIGURATIONS as CPSAT_CONFIGS
from mip_cp_models import solve_nqueens as solve_mip, CONFIGURATIONS as MIP_CONFIGS, SOLVERS as MIP_SOLVERS

def run_sat_benchmarks(min_n=4, max_n=12, encodings=None, solvers=None, time_limit=None):
    if encodings is None:
        encodings = list(ENCODINGS.keys())
    if solvers is None:
        solvers = ["glucose3", "minisat22", "cadical"]
        
    results = []
    for n in range(min_n, max_n + 1):
        for encoding in encodings:
            for solver in solvers:
                print(f"[SAT] N={n}, encoding={encoding}, solver={solver}...")
                start = time.time()
                try:
                    res = solve_sat(n, encoding=encoding, solver_name=solver)
                    results.append({
                        "n": n,
                        "paradigm": "SAT",
                        "model_variant": encoding,
                        "solver": solver,
                        "status": res.get("status", "UNKNOWN"),
                        "is_sat": res.get("is_sat", False),
                        "variables_count": res.get("variables_count", 0),
                        "clauses_constraints_count": res.get("clauses_count", 0),
                        "time_seconds": res.get("time", time.time() - start),
                        "error": res.get("error", "")
                    })
                except Exception as e:
                    elapsed = time.time() - start
                    results.append({
                        "n": n,
                        "paradigm": "SAT",
                        "model_variant": encoding,
                        "solver": solver,
                        "status": "ERROR",
                        "is_sat": False,
                        "variables_count": 0,
                        "clauses_constraints_count": 0,
                        "time_seconds": elapsed,
                        "error": str(e)
                    })
    return results

def run_cpsat_benchmarks(min_n=4, max_n=16, configs=None, time_limit=30.0):
    if configs is None:
        configs = sorted(list(CPSAT_CONFIGS))
        
    results = []
    for n in range(min_n, max_n + 1):
        for config in configs:
            print(f"[CP-SAT] N={n}, config={config}...")
            start = time.time()
            try:
                res = solve_cpsat(n, config=config, time_limit=time_limit)
                results.append({
                    "n": n,
                    "paradigm": "CP-SAT",
                    "model_variant": config,
                    "solver": "ortools-cpsat",
                    "status": res.get("status", "UNKNOWN"),
                    "is_sat": res.get("is_sat", False),
                    "variables_count": res.get("variables_count", 0),
                    "clauses_constraints_count": res.get("constraints_count", 0),
                    "time_seconds": res.get("time", time.time() - start),
                    "error": ""
                })
            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "n": n,
                    "paradigm": "CP-SAT",
                    "model_variant": config,
                    "solver": "ortools-cpsat",
                    "status": "ERROR",
                    "is_sat": False,
                    "variables_count": 0,
                    "clauses_constraints_count": 0,
                    "time_seconds": elapsed,
                    "error": str(e)
                })
    return results

def run_mip_benchmarks(min_n=4, max_n=12, configs=None, solvers=None, time_limit=30.0):
    if configs is None:
        configs = sorted(list(MIP_CONFIGS))
    if solvers is None:
        solvers = ["cbc", "scip"]
        
    results = []
    for n in range(min_n, max_n + 1):
        for config in configs:
            for solver in solvers:
                print(f"[MIP] N={n}, config={config}, solver={solver}...")
                start = time.time()
                try:
                    res = solve_mip(n, config=config, solver_name=solver, time_limit=time_limit)
                    results.append({
                        "n": n,
                        "paradigm": "MIP",
                        "model_variant": config,
                        "solver": res.get("solver", solver),
                        "status": res.get("status", "UNKNOWN"),
                        "is_sat": res.get("is_sat", False),
                        "variables_count": res.get("variables_count", 0),
                        "clauses_constraints_count": res.get("constraints_count", 0),
                        "time_seconds": res.get("time", time.time() - start),
                        "error": ""
                    })
                except Exception as e:
                    elapsed = time.time() - start
                    results.append({
                        "n": n,
                        "paradigm": "MIP",
                        "model_variant": config,
                        "solver": solver,
                        "status": "ERROR",
                        "is_sat": False,
                        "variables_count": 0,
                        "clauses_constraints_count": 0,
                        "time_seconds": elapsed,
                        "error": str(e)
                    })
    return results

def main():
    parser = argparse.ArgumentParser(description="Run N-Queens benchmarks across SAT, CP-SAT, and MIP models.")
    parser.add_argument("--min-n", type=int, default=4, help="Minimum N value")
    parser.add_argument("--max-n-sat", type=int, default=10, help="Maximum N value for SAT")
    parser.add_argument("--max-n-cpsat", type=int, default=14, help="Maximum N value for CP-SAT")
    parser.add_argument("--max-n-mip", type=int, default=10, help="Maximum N value for MIP")
    parser.add_argument("--time-limit", type=float, default=10.0, help="Time limit in seconds for CP-SAT / MIP")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save CSV results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Running SAT Benchmarks ===")
    sat_results = run_sat_benchmarks(min_n=args.min_n, max_n=args.max_n_sat)

    print("=== Running CP-SAT Benchmarks ===")
    cpsat_results = run_cpsat_benchmarks(min_n=args.min_n, max_n=args.max_n_cpsat, time_limit=args.time_limit)

    print("=== Running MIP Benchmarks ===")
    mip_results = run_mip_benchmarks(min_n=args.min_n, max_n=args.max_n_mip, time_limit=args.time_limit)

    all_results = sat_results + cpsat_results + mip_results
    df = pd.DataFrame(all_results)

    combined_csv = os.path.join(args.output_dir, "benchmark_results.csv")
    df.to_csv(combined_csv, index=False)
    print(f"\nSaved all benchmark results to {combined_csv}")

    sat_csv = os.path.join(args.output_dir, "sat_results.csv")
    pd.DataFrame(sat_results).to_csv(sat_csv, index=False)

    cpsat_csv = os.path.join(args.output_dir, "cpsat_results.csv")
    pd.DataFrame(cpsat_results).to_csv(cpsat_csv, index=False)

    mip_csv = os.path.join(args.output_dir, "mip_results.csv")
    pd.DataFrame(mip_results).to_csv(mip_csv, index=False)

    print(f"Summary generated: {len(all_results)} total benchmark runs completed.")

if __name__ == "__main__":
    main()
