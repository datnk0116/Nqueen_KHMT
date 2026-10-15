"""
CPLEX CP Optimizer model builder and solver for N-Queens.
"""
import time
import os
from docplex.cp.model import CpoModel
from docplex.cp.config import context

cp_path = os.environ.get("CPLEX_CP_EXECUTABLE")
if not cp_path and os.environ.get("CPLEX_STUDIO_DIR"):
    cp_path = os.path.join(os.environ.get("CPLEX_STUDIO_DIR"), "cpoptimizer/bin/arm64_osx/cpoptimizer")
if not cp_path or not os.path.exists(cp_path):
    for fallback in [
        'SAT/bin/cpoptimizer',
        '/Applications/CPLEX_Studio222/cpoptimizer/bin/arm64_osx/cpoptimizer',
        '/Applications/CPLEX_Studio221/cpoptimizer/bin/arm64_osx/cpoptimizer',
        'cpoptimizer'
    ]:
        if (fallback == 'cpoptimizer') or os.path.exists(fallback):
            cp_path = fallback
            break

if cp_path and (cp_path == 'cpoptimizer' or os.path.exists(cp_path)):
    context.solver.local.execfile = cp_path

def solve_nqueens(n: int, time_limit: float = None):
    """
    Solves N-Queens using CPLEX CP Optimizer model.
    Returns dict: {"status", "total_time", "variables_count", "constraints_count", "queens", "build_time", "solve_time", "solver_reported_time"}
    """
    build_start = time.perf_counter()
    variables_count = 0
    constraints_count = 0
    q = []
    m = None
    try:
        m = CpoModel("nqueens_cp")
        q = [m.integer_var(0, n - 1, f"q{i}") for i in range(n)]
        m.add(m.all_diff(q))
        m.add(m.all_diff([q[i] + i for i in range(n)]))
        m.add(m.all_diff([q[i] - i for i in range(n)]))
        
        variables_count = len(q)
        try:
            constraints_count = len(m.get_constraint_list())
        except Exception:
            constraints_count = 3
        build_time = time.perf_counter() - build_start
    except Exception as e:
        build_time = time.perf_counter() - build_start
        err_str = str(e).lower()
        status = "LICENSE_LIMIT" if ("1016" in err_str or "license" in err_str or "size limit" in err_str) else "ERROR"
        return {
            "status": status,
            "total_time": build_time,
            "variables_count": variables_count,
            "constraints_count": constraints_count,
            "queens": [],
            "build_time": build_time,
            "solve_time": 0.0,
            "solver_reported_time": None
        }

    solve_start = time.perf_counter()
    try:
        params = {"LogVerbosity": "Quiet"}
        if time_limit is not None:
            params["TimeLimit"] = max(0.001, time_limit - build_time)
        else:
            params["TimeLimit"] = 60
            
        sol = m.solve(**params)
        solve_time = time.perf_counter() - solve_start
        total_time = build_time + solve_time
        
        solver_reported_time = None
        if sol is not None and hasattr(sol, 'get_solve_time'):
            try:
                solver_reported_time = sol.get_solve_time()
            except Exception:
                pass

        if sol is None or not sol.is_solution():
            status = str(sol.get_solve_status()) if sol is not None else "NO_SOLUTION"
            stop_cause = str(sol.get_stop_cause()).upper() if (sol is not None and hasattr(sol, 'get_stop_cause')) else ""
            if "LIMIT" in stop_cause or "TIME" in stop_cause or (time_limit is not None and solve_time >= time_limit * 0.95):
                status = "TIME_LIMIT"
            return {
                "status": status,
                "total_time": total_time,
                "variables_count": variables_count,
                "constraints_count": constraints_count,
                "queens": [],
                "build_time": build_time,
                "solve_time": solve_time,
                "solver_reported_time": solver_reported_time
            }
            
        raw_status = str(sol.get_solve_status()) if sol is not None else "NO_SOLUTION"
        status_upper = raw_status.upper()
        if "OPTIMAL" in status_upper:
            status = "OPTIMAL"
        elif "FEASIBLE" in status_upper:
            status = "FEASIBLE"
        else:
            status = raw_status
            
        queens = [(i, sol[q[i]]) for i in range(n)]
        return {
            "status": status,
            "total_time": total_time,
            "variables_count": variables_count,
            "constraints_count": constraints_count,
            "queens": queens,
            "build_time": build_time,
            "solve_time": solve_time,
            "solver_reported_time": solver_reported_time
        }
    except Exception as e:
        solve_time = time.perf_counter() - solve_start
        total_time = build_time + solve_time
        err_str = str(e).lower()
        status = "LICENSE_LIMIT" if ("1016" in err_str or "license" in err_str or "size limit" in err_str) else "ERROR"
        solver_reported_time = None
        if 'sol' in locals() and sol is not None and hasattr(sol, 'get_solve_time'):
            try:
                solver_reported_time = sol.get_solve_time()
            except Exception:
                pass
        return {
            "status": status,
            "total_time": total_time,
            "variables_count": variables_count,
            "constraints_count": constraints_count,
            "queens": [],
            "build_time": build_time,
            "solve_time": solve_time,
            "solver_reported_time": solver_reported_time
        }
