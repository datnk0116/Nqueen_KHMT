"""
CPLEX CP Optimizer model builder and solver for N-Queens.
"""
import time
from docplex.cp.model import CpoModel

def solve_nqueens(n: int, time_limit: float = None):
    """
    Solves N-Queens using CPLEX CP Optimizer model.
    Returns: (status, time, variables_count, queens, build_time, solve_time)
    """
    build_start = time.time()
    try:
        m = CpoModel("nqueens_cp")
        q = [m.integer_var(0, n - 1, f"q{i}") for i in range(n)]
        m.add(m.all_diff(q))
        m.add(m.all_diff([q[i] + i for i in range(n)]))
        m.add(m.all_diff([q[i] - i for i in range(n)]))
        
        variables_count = len(q)
        build_time = time.time() - build_start
    except Exception as e:
        build_time = time.time() - build_start
        print(f"CPLEX CP build exception: {e}")
        err_str = str(e).lower()
        if "1016" in err_str or "license" in err_str or "size limit" in err_str:
            return ("LICENSE_LIMIT", build_time, 0, [], build_time, 0.0)
        return ("ERROR", build_time, 0, [], build_time, 0.0)

    solve_start = time.time()
    try:
        params = {"LogVerbosity": "Quiet"}
        if time_limit is not None:
            params["TimeLimit"] = time_limit
        else:
            params["TimeLimit"] = 60
            
        sol = m.solve(**params)
        solve_time = time.time() - solve_start
        total_time = build_time + solve_time
        
        if sol is None or not sol.is_solution():
            status = sol.get_solve_status() if sol is not None else "NO_SOLUTION"
            return (str(status), total_time, variables_count, [], build_time, solve_time)
            
        raw_status = str(sol.get_solve_status()) if sol is not None else "NO_SOLUTION"
        status_upper = raw_status.upper()
        if "OPTIMAL" in status_upper:
            status = "OPTIMAL"
        elif "FEASIBLE" in status_upper:
            status = "FEASIBLE"
        else:
            status = raw_status
            
        queens = [(i, sol[q[i]]) for i in range(n)]
        return (status, total_time, variables_count, queens, build_time, solve_time)
    except Exception as e:
        solve_time = time.time() - solve_start
        total_time = build_time + solve_time
        print(f"CPLEX CP solve exception: {e}")
        err_str = str(e).lower()
        if "1016" in err_str or "license" in err_str or "size limit" in err_str:
            return ("LICENSE_LIMIT", total_time, variables_count, [], build_time, solve_time)
        return ("ERROR", total_time, variables_count, [], build_time, solve_time)
