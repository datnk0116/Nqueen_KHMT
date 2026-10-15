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
import sys, os, time, json, math
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
    model_time = build_time + solve_time if (not math.isnan(build_time) and not math.isnan(solve_time)) else float('nan')
    if time_limit is not None and total_time > time_limit and status not in ("TIMEOUT", "ERROR", "MEMOUT", "LICENSE_LIMIT"):
        status = "TIMEOUT"
        build_time = float('nan')
        solve_time = float('nan')
        import_time = float('nan')
        model_time = float('nan')
        total_time = time_limit

    out = {{
        "status": status,
        "is_sat": is_sat,
        "variables_count": variables_count,
        "constraints_count": constraints_count,
        "import_time": import_time,
        "build_time": build_time,
        "solve_time": solve_time,
        "model_time": model_time,
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
        "model_time": float('nan'),
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
                "model_time": float('nan'),
                "time_seconds": time_limit, "solver_reported_time": None, "wall_time": p_wall,
                "verified": False, "error": "Memory limit exceeded (MEMOUT)"
            }

        for line in stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                data = json.loads(line[len("RESULT_JSON:"):])
                b_t = data.get("build_time", float('nan'))
                s_t = data.get("solve_time", float('nan'))
                m_t = data.get("model_time")
                if m_t is None or pd.isna(m_t):
                    m_t = b_t + s_t if (not pd.isna(b_t) and not pd.isna(s_t)) else float('nan')
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
                    "build_time": b_t,
                    "solve_time": s_t,
                    "model_time": m_t,
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
            "model_time": float('nan'),
            "time_seconds": 0.0, "solver_reported_time": None, "wall_time": p_wall,
            "verified": False, "error": stderr or "No JSON output"
        }
    except subprocess.TimeoutExpired:
        p_wall = time.perf_counter() - p_start
        return {
            "n": n, "paradigm": paradigm, "model_variant": model_variant, "solver": solver,
            "status": "TIMEOUT", "is_sat": False, "variables_count": 0, "clauses_constraints_count": 0,
            "import_time": float('nan'), "build_time": float('nan'), "solve_time": float('nan'),
            "model_time": float('nan'),
            "time_seconds": time_limit, "solver_reported_time": None, "wall_time": p_wall,
            "verified": False, "error": "Hard timeout expired"
        }
    except Exception as e:
        p_wall = time.perf_counter() - p_start
        return {
            "n": n, "paradigm": paradigm, "model_variant": model_variant, "solver": solver,
            "status": "ERROR", "is_sat": False, "variables_count": 0, "clauses_constraints_count": 0,
            "import_time": float('nan'), "build_time": float('nan'), "solve_time": float('nan'),
            "model_time": float('nan'),
            "time_seconds": 0.0, "solver_reported_time": None, "wall_time": p_wall,
            "verified": False, "error": str(e)
        }

from collections import Counter

def aggregate_repetition_records(run_records, time_limit):
    """
    Aggregates execution records across multiple repetitions of a solver benchmark configuration.

    Aggregation Policy:
    1. A configuration is classified as successful ('OPTIMAL', 'FEASIBLE', 'SAT') only if successful
       runs form a strict majority (runs_success > runs_total / 2.0).
       - Aggregate status is the most common status among successful repetitions (mode).
       - is_sat is True, verified is True if all successful runs are verified.
       - Diagnostic notes in 'error' record non-success occurrences (e.g. "1/3 runs timed out",
         "1/3 runs ran out of memory", "1/3 runs errored", "1/3 runs hit license limits").
       - Median phase times (import, build, solve, model) are computed across successful repetitions.
    2. If successful runs do not form a strict majority (runs_success <= runs_total / 2.0):
       - is_sat is False, verified is False.
       - The configuration is never classified as success (indeterminate/failure).
       - If a failure category forms a strict majority (count > runs_total / 2.0), that status is selected:
         * LICENSE_LIMIT if runs_license > runs_total / 2.0
         * MEMOUT if runs_memout > runs_total / 2.0
         * TIMEOUT (or TIME_LIMIT if all timeouts are TIME_LIMIT) if runs_timeout > runs_total / 2.0
         * ERROR if runs_error > runs_total / 2.0
       - If no category achieves a strict majority (tied counts in even repetitions or split failures),
         the failure category with the highest count is selected, breaking ties deterministically
         by severity/determinism: LICENSE_LIMIT > MEMOUT > TIMEOUT > ERROR.
       - Diagnostic notes in 'error' describe the primary failure message and all mixed outcomes.
       - Median phase times (import, build, solve, model) are set to NaN, ensuring timed out / failed
         configurations do not report misleading solve times.
    3. time_seconds and wall_time are medians across all valid repetition records.
    4. Counters runs_total, runs_success, runs_timeout, runs_error accurately represent run records.
    """
    if not run_records:
        raise ValueError("run_records cannot be empty")

    base = run_records[0]
    n = base.get("n")
    paradigm = base.get("paradigm")
    variant = base.get("model_variant")
    solver = base.get("solver")

    runs_total = len(run_records)
    successful = [r for r in run_records if r.get("status") in ("OPTIMAL", "FEASIBLE", "SAT")]
    runs_success = len(successful)
    runs_timeout = sum(1 for r in run_records if r.get("status") in ("TIMEOUT", "TIME_LIMIT"))
    runs_license = sum(1 for r in run_records if r.get("status") == "LICENSE_LIMIT")
    runs_memout = sum(1 for r in run_records if r.get("status") == "MEMOUT")
    runs_error = sum(1 for r in run_records if r.get("status") == "ERROR")

    majority_threshold = runs_total / 2.0

    if runs_success > majority_threshold:
        status_counts = Counter(r["status"] for r in successful)
        status = status_counts.most_common(1)[0][0]
        is_sat = True
        verified = all(r.get("verified", False) for r in successful)

        notes = []
        if runs_timeout > 0:
            notes.append(f"{runs_timeout}/{runs_total} runs timed out")
        if runs_memout > 0:
            notes.append(f"{runs_memout}/{runs_total} runs ran out of memory")
        if runs_error > 0:
            notes.append(f"{runs_error}/{runs_total} runs errored")
        if runs_license > 0:
            notes.append(f"{runs_license}/{runs_total} runs hit license limits")
        error = "; ".join(notes)
    else:
        is_sat = False
        verified = False

        if runs_license > majority_threshold:
            status = "LICENSE_LIMIT"
        elif runs_memout > majority_threshold:
            status = "MEMOUT"
        elif runs_timeout > majority_threshold:
            soft_timeouts = sum(1 for r in run_records if r.get("status") == "TIME_LIMIT")
            status = "TIME_LIMIT" if soft_timeouts == runs_timeout else "TIMEOUT"
        elif runs_error > majority_threshold:
            status = "ERROR"
        else:
            failure_counts = [
                ("LICENSE_LIMIT", runs_license),
                ("MEMOUT", runs_memout),
                ("TIMEOUT", runs_timeout),
                ("ERROR", runs_error),
            ]
            max_status, _ = max(failure_counts, key=lambda x: x[1])
            if max_status == "TIMEOUT":
                soft_timeouts = sum(1 for r in run_records if r.get("status") == "TIME_LIMIT")
                status = "TIME_LIMIT" if (runs_timeout > 0 and soft_timeouts == runs_timeout) else "TIMEOUT"
            else:
                status = max_status

        notes = []
        if status == "LICENSE_LIMIT":
            lic_errs = [r.get("error", "") for r in run_records if r.get("status") == "LICENSE_LIMIT" and r.get("error")]
            first_lic = lic_errs[0] if lic_errs else "License/size limit reached"
            if runs_license == runs_total:
                notes.append(first_lic)
            else:
                notes.append(f"{runs_license}/{runs_total} runs hit license limits: {first_lic}")
        elif status == "MEMOUT":
            mem_errs = [r.get("error", "") for r in run_records if r.get("status") == "MEMOUT" and r.get("error")]
            first_mem = mem_errs[0] if mem_errs else "Memory limit exceeded (MEMOUT)"
            if runs_memout == runs_total:
                notes.append(first_mem)
            else:
                notes.append(f"{runs_memout}/{runs_total} runs ran out of memory")
        elif status in ("TIMEOUT", "TIME_LIMIT"):
            notes.append(f"{runs_timeout}/{runs_total} runs timed out")
        elif status == "ERROR":
            err_msgs = [r.get("error", "") for r in run_records if r.get("status") == "ERROR" and r.get("error")]
            first_err = err_msgs[0] if err_msgs else "Error"
            notes.append(f"{runs_error}/{runs_total} runs errored: {first_err}")

        if status not in ("TIMEOUT", "TIME_LIMIT") and runs_timeout > 0:
            notes.append(f"{runs_timeout}/{runs_total} runs timed out")
        if status != "MEMOUT" and runs_memout > 0:
            notes.append(f"{runs_memout}/{runs_total} runs ran out of memory")
        if status != "ERROR" and runs_error > 0:
            notes.append(f"{runs_error}/{runs_total} runs errored")
        if status != "LICENSE_LIMIT" and runs_license > 0:
            notes.append(f"{runs_license}/{runs_total} runs hit license limits")
        if runs_success > 0:
            notes.append(f"{runs_success}/{runs_total} runs succeeded")

        error = "; ".join(notes)

    all_times = [r["time_seconds"] for r in run_records if not pd.isna(r.get("time_seconds"))]
    all_walls = [r["wall_time"] for r in run_records if ("wall_time" in r and not pd.isna(r.get("wall_time")))]

    median_time = statistics.median(all_times) if all_times else time_limit
    median_wall = statistics.median(all_walls) if all_walls else median_time

    if status in ("TIMEOUT", "TIME_LIMIT", "MEMOUT", "ERROR", "LICENSE_LIMIT"):
        median_import = float('nan')
        median_build = float('nan')
        median_solve = float('nan')
        median_model = float('nan')
        reported_times = [r["solver_reported_time"] for r in run_records if r.get("solver_reported_time") is not None and not pd.isna(r.get("solver_reported_time"))]
        median_reported = statistics.median(reported_times) if reported_times else None
        variables_count = base.get("variables_count", 0) if status == "LICENSE_LIMIT" else 0
        constraints_count = base.get("clauses_constraints_count", 0) if status == "LICENSE_LIMIT" else 0
    else:
        import_times = [r["import_time"] for r in successful if not pd.isna(r.get("import_time"))]
        build_times = [r["build_time"] for r in successful if not pd.isna(r.get("build_time"))]
        solve_times = [r["solve_time"] for r in successful if not pd.isna(r.get("solve_time"))]
        model_times = [r["model_time"] for r in successful if ("model_time" in r and not pd.isna(r.get("model_time")))]
        reported_times = [r["solver_reported_time"] for r in successful if r.get("solver_reported_time") is not None and not pd.isna(r.get("solver_reported_time"))]

        median_import = statistics.median(import_times) if import_times else float('nan')
        median_build = statistics.median(build_times) if build_times else float('nan')
        median_solve = statistics.median(solve_times) if solve_times else float('nan')
        median_model = statistics.median(model_times) if model_times else (median_build + median_solve if (not pd.isna(median_build) and not pd.isna(median_solve)) else float('nan'))
        median_reported = statistics.median(reported_times) if reported_times else None
        variables_count = successful[0].get("variables_count", 0)
        constraints_count = successful[0].get("clauses_constraints_count", 0)

    return {
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
        "model_time": median_model,
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

                rep_res = aggregate_repetition_records(run_records, time_limit)
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
