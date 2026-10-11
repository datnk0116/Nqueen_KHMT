"""
Benchmark runner for N-Queens across SAT solvers, encodings, CP-SAT models, MIP models, Gurobi, and CPLEX solvers.
Measures build time, solve time, total time, status, variable counts, clause/constraint counts,
solution verification using Board, and saves results to CSV.
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

from sat_model import NQueensSAT, ENCODINGS as SAT_ENCODINGS, SOLVERS as SAT_SOLVERS
from cpsat_model import NQueensCPSAT, CONFIGURATIONS as CPSAT_CONFIGS
from mip_cp_models import NQueensMIP, CONFIGURATIONS as MIP_CONFIGS, SOLVERS as MIP_SOLVERS
from gurobi_model import solve_nqueens as solve_gurobi
from cplex_mip_model import solve_nqueens as solve_cplex_mip
from cplex_cp_model import solve_nqueens as solve_cplex_cp
from board import verify

def run_sat_benchmarks(min_n=4, max_n=12, encodings=None, solvers=None, time_limit=None):
    if encodings is None:
        encodings = list(SAT_ENCODINGS.keys())
    if solvers is None:
        # Exclude redundant aliases (e.g. cadical153 if cadical is present)
        solvers = [s for s in ["glucose3", "minisat22", "cadical", "kissat", "maplecm"] if s in SAT_SOLVERS]
        
    results = []
    for n in range(min_n, max_n + 1):
        for encoding in encodings:
            for solver in solvers:
                print(f"[SAT] N={n}, encoding={encoding}, solver={solver}...")
                build_start = time.time()
                try:
                    model = NQueensSAT(n, encoding=encoding, solver_name=solver)
                    model.build()
                    build_time = time.time() - build_start
                    
                    solve_start = time.time()
                    res = model.solve()
                    solve_time = time.time() - solve_start
                    total_time = build_time + solve_time
                    
                    queens = res.get("queens", [])
                    is_sat = res.get("is_sat", False)
                    status = res.get("status", "UNKNOWN")
                    verified = verify(queens, n) if is_sat and queens else (status == "UNSAT")
                    
                    results.append({
                        "n": n,
                        "paradigm": "SAT",
                        "model_variant": encoding,
                        "solver": solver,
                        "status": status,
                        "is_sat": is_sat,
                        "variables_count": res.get("variables_count", 0),
                        "clauses_constraints_count": res.get("clauses_count", 0),
                        "build_time": build_time,
                        "solve_time": solve_time,
                        "time_seconds": total_time,
                        "verified": verified,
                        "error": res.get("error", "")
                    })
                except Exception as e:
                    build_time = time.time() - build_start
                    results.append({
                        "n": n,
                        "paradigm": "SAT",
                        "model_variant": encoding,
                        "solver": solver,
                        "status": "ERROR",
                        "is_sat": False,
                        "variables_count": 0,
                        "clauses_constraints_count": 0,
                        "build_time": build_time,
                        "solve_time": 0.0,
                        "time_seconds": build_time,
                        "verified": False,
                        "error": str(e)
                    })
    return results

def run_cpsat_benchmarks(min_n=4, max_n=16, configs=None, time_limit=30.0):
    if configs is None:
        # Remove duplicate configurations / redundant aliases (integer, boolean, allowed)
        configs = ["integer", "boolean", "allowed"]
        
    results = []
    for n in range(min_n, max_n + 1):
        for config in configs:
            print(f"[CP-SAT] N={n}, config={config}...")
            build_start = time.time()
            try:
                model = NQueensCPSAT(n, config=config)
                model.build()
                build_time = time.time() - build_start
                
                solve_start = time.time()
                res = model.solve(time_limit=time_limit)
                solve_time = time.time() - solve_start
                total_time = build_time + solve_time
                
                queens = res.get("queens", [])
                is_sat = res.get("is_sat", False)
                status = res.get("status", "UNKNOWN")
                verified = verify(queens, n) if is_sat and queens else (status in ("INFEASIBLE", "MODEL_INVALID"))
                
                results.append({
                    "n": n,
                    "paradigm": "CP-SAT",
                    "model_variant": config,
                    "solver": "ortools-cpsat",
                    "status": status,
                    "is_sat": is_sat,
                    "variables_count": res.get("variables_count", 0),
                    "clauses_constraints_count": res.get("constraints_count", 0),
                    "build_time": build_time,
                    "solve_time": solve_time,
                    "time_seconds": total_time,
                    "verified": verified,
                    "error": ""
                })
            except Exception as e:
                build_time = time.time() - build_start
                results.append({
                    "n": n,
                    "paradigm": "CP-SAT",
                    "model_variant": config,
                    "solver": "ortools-cpsat",
                    "status": "ERROR",
                    "is_sat": False,
                    "variables_count": 0,
                    "clauses_constraints_count": 0,
                    "build_time": build_time,
                    "solve_time": 0.0,
                    "time_seconds": build_time,
                    "verified": False,
                    "error": str(e)
                })
    return results

def run_mip_benchmarks(min_n=4, max_n=12, configs=None, solvers=None, time_limit=30.0):
    if configs is None:
        # Remove duplicate configurations / redundant aliases (binary, integer)
        configs = ["binary", "integer"]
    if solvers is None:
        solvers = ["cbc", "scip"]
        
    results = []
    for n in range(min_n, max_n + 1):
        for config in configs:
            for solver in solvers:
                print(f"[MIP] N={n}, config={config}, solver={solver}...")
                build_start = time.time()
                try:
                    model = NQueensMIP(n, config=config, solver_name=solver)
                    model.build()
                    build_time = time.time() - build_start
                    
                    solve_start = time.time()
                    res = model.solve(time_limit=time_limit)
                    solve_time = time.time() - solve_start
                    total_time = build_time + solve_time
                    
                    queens = res.get("queens", [])
                    is_sat = res.get("is_sat", False)
                    status = res.get("status", "UNKNOWN")
                    verified = verify(queens, n) if is_sat and queens else False
                    
                    results.append({
                        "n": n,
                        "paradigm": "MIP",
                        "model_variant": config,
                        "solver": res.get("solver", solver),
                        "status": status,
                        "is_sat": is_sat,
                        "variables_count": res.get("variables_count", 0),
                        "clauses_constraints_count": res.get("constraints_count", 0),
                        "build_time": build_time,
                        "solve_time": solve_time,
                        "time_seconds": total_time,
                        "verified": verified,
                        "error": ""
                    })
                except Exception as e:
                    build_time = time.time() - build_start
                    results.append({
                        "n": n,
                        "paradigm": "MIP",
                        "model_variant": config,
                        "solver": solver,
                        "status": "ERROR",
                        "is_sat": False,
                        "variables_count": 0,
                        "clauses_constraints_count": 0,
                        "build_time": build_time,
                        "solve_time": 0.0,
                        "time_seconds": build_time,
                        "verified": False,
                        "error": str(e)
                    })
    return results

def run_gurobi_benchmarks(min_n=4, max_n=16, time_limit=30.0):
    results = []
    for n in range(min_n, max_n + 1):
        print(f"[Gurobi] N={n}...")
        try:
            status, total_time, variables_count, queens, build_time, solve_time = solve_gurobi(n, time_limit=time_limit)
            is_sat = status in ("OPTIMAL", "SUBOPTIMAL") or len(queens) > 0
            verified = verify(queens, n) if queens else False
            results.append({
                "n": n,
                "paradigm": "Gurobi",
                "model_variant": "binary",
                "solver": "gurobi",
                "status": status,
                "is_sat": is_sat,
                "variables_count": variables_count,
                "clauses_constraints_count": 0,
                "build_time": build_time,
                "solve_time": solve_time,
                "time_seconds": total_time,
                "verified": verified,
                "error": "" if status != "LICENSE_LIMIT" else "Gurobi license/size limit reached"
            })
        except Exception as e:
            results.append({
                "n": n,
                "paradigm": "Gurobi",
                "model_variant": "binary",
                "solver": "gurobi",
                "status": "ERROR",
                "is_sat": False,
                "variables_count": 0,
                "clauses_constraints_count": 0,
                "build_time": 0.0,
                "solve_time": 0.0,
                "time_seconds": 0.0,
                "verified": False,
                "error": str(e)
            })
    return results

def run_cplex_mip_benchmarks(min_n=4, max_n=16, time_limit=30.0):
    results = []
    for n in range(min_n, max_n + 1):
        print(f"[CPLEX-MIP] N={n}...")
        try:
            status, total_time, variables_count, queens, build_time, solve_time = solve_cplex_mip(n, time_limit=time_limit)
            is_sat = status in ("OPTIMAL", "FEASIBLE") or len(queens) > 0
            verified = verify(queens, n) if queens else False
            results.append({
                "n": n,
                "paradigm": "CPLEX-MIP",
                "model_variant": "binary",
                "solver": "cplex",
                "status": status,
                "is_sat": is_sat,
                "variables_count": variables_count,
                "clauses_constraints_count": 0,
                "build_time": build_time,
                "solve_time": solve_time,
                "time_seconds": total_time,
                "verified": verified,
                "error": "" if status != "LICENSE_LIMIT" else "CPLEX license/size limit reached"
            })
        except Exception as e:
            results.append({
                "n": n,
                "paradigm": "CPLEX-MIP",
                "model_variant": "binary",
                "solver": "cplex",
                "status": "ERROR",
                "is_sat": False,
                "variables_count": 0,
                "clauses_constraints_count": 0,
                "build_time": 0.0,
                "solve_time": 0.0,
                "time_seconds": 0.0,
                "verified": False,
                "error": str(e)
            })
    return results

def run_cplex_cp_benchmarks(min_n=4, max_n=16, time_limit=30.0):
    results = []
    for n in range(min_n, max_n + 1):
        print(f"[CPLEX-CP] N={n}...")
        try:
            status, total_time, variables_count, queens, build_time, solve_time = solve_cplex_cp(n, time_limit=time_limit)
            is_sat = status in ("OPTIMAL", "Feasible") or len(queens) > 0
            verified = verify(queens, n) if queens else False
            results.append({
                "n": n,
                "paradigm": "CPLEX-CP",
                "model_variant": "integer",
                "solver": "cplex-cp",
                "status": status,
                "is_sat": is_sat,
                "variables_count": variables_count,
                "clauses_constraints_count": 0,
                "build_time": build_time,
                "solve_time": solve_time,
                "time_seconds": total_time,
                "verified": verified,
                "error": "" if status != "LICENSE_LIMIT" else "CPLEX CP license/size limit reached"
            })
        except Exception as e:
            results.append({
                "n": n,
                "paradigm": "CPLEX-CP",
                "model_variant": "integer",
                "solver": "cplex-cp",
                "status": "ERROR",
                "is_sat": False,
                "variables_count": 0,
                "clauses_constraints_count": 0,
                "build_time": 0.0,
                "solve_time": 0.0,
                "time_seconds": 0.0,
                "verified": False,
                "error": str(e)
            })
    return results

def main():
    parser = argparse.ArgumentParser(description="Run N-Queens benchmarks across SAT, CP-SAT, MIP, Gurobi, and CPLEX models.")
    parser.add_argument("--min-n", type=int, default=4, help="Minimum N value")
    parser.add_argument("--max-n-sat", type=int, default=12, help="Maximum N value for SAT")
    parser.add_argument("--max-n-cpsat", type=int, default=16, help="Maximum N value for CP-SAT")
    parser.add_argument("--max-n-mip", type=int, default=12, help="Maximum N value for MIP")
    parser.add_argument("--max-n-gurobi", type=int, default=16, help="Maximum N value for Gurobi")
    parser.add_argument("--max-n-cplex", type=int, default=16, help="Maximum N value for CPLEX MIP/CP")
    parser.add_argument("--time-limit", type=float, default=30.0, help="Time limit in seconds for solvers")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save CSV results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Running SAT Benchmarks ===")
    sat_results = run_sat_benchmarks(min_n=args.min_n, max_n=args.max_n_sat, time_limit=args.time_limit)

    print("=== Running CP-SAT Benchmarks ===")
    cpsat_results = run_cpsat_benchmarks(min_n=args.min_n, max_n=args.max_n_cpsat, time_limit=args.time_limit)

    print("=== Running MIP Benchmarks ===")
    mip_results = run_mip_benchmarks(min_n=args.min_n, max_n=args.max_n_mip, time_limit=args.time_limit)

    print("=== Running Gurobi Benchmarks ===")
    gurobi_results = run_gurobi_benchmarks(min_n=args.min_n, max_n=args.max_n_gurobi, time_limit=args.time_limit)

    print("=== Running CPLEX MIP Benchmarks ===")
    cplex_mip_results = run_cplex_mip_benchmarks(min_n=args.min_n, max_n=args.max_n_cplex, time_limit=args.time_limit)

    print("=== Running CPLEX CP Benchmarks ===")
    cplex_cp_results = run_cplex_cp_benchmarks(min_n=args.min_n, max_n=args.max_n_cplex, time_limit=args.time_limit)

    all_results = sat_results + cpsat_results + mip_results + gurobi_results + cplex_mip_results + cplex_cp_results
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

    gurobi_csv = os.path.join(args.output_dir, "gurobi_results.csv")
    pd.DataFrame(gurobi_results).to_csv(gurobi_csv, index=False)

    cplex_csv = os.path.join(args.output_dir, "cplex_results.csv")
    pd.DataFrame(cplex_mip_results + cplex_cp_results).to_csv(cplex_csv, index=False)

    print(f"Summary generated: {len(all_results)} total benchmark runs completed.")

if __name__ == "__main__":
    main()
