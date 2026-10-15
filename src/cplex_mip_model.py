"""
CPLEX MIP model builder and solver for N-Queens.
"""
import time
from docplex.mp.model import Model

def solve_nqueens(n: int, time_limit: float = None):
    """
    Solves N-Queens using CPLEX MIP model.
    Returns: (status, time, variables_count, constraints_count, queens, build_time, solve_time)
    """
    build_start = time.perf_counter()
    try:
        m = Model("nqueens_mip")
        m.context.solver.log_output = None
        if time_limit is not None:
            m.set_time_limit(time_limit)
            
        x = m.binary_var_matrix(n, n, "x")
        for i in range(n):
            m.add_constraint(m.sum(x[i, j] for j in range(n)) == 1)
            m.add_constraint(m.sum(x[j, i] for j in range(n)) == 1)
        for d in range(-(n - 1), n):
            cells = [x[i, i - d] for i in range(n) if 0 <= i - d < n]
            if len(cells) > 1:
                m.add_constraint(m.sum(cells) <= 1)
        for s in range(2 * n - 1):
            cells = [x[i, s - i] for i in range(n) if 0 <= s - i < n]
            if len(cells) > 1:
                m.add_constraint(m.sum(cells) <= 1)
                
        variables_count = m.number_of_variables
        constraints_count = m.number_of_constraints
        build_time = time.perf_counter() - build_start
    except Exception as e:
        build_time = time.perf_counter() - build_start
        err_str = str(e).lower()
        if "1016" in err_str or "license" in err_str or "limit" in err_str:
            return ("LICENSE_LIMIT", build_time, 0, 0, [], build_time, 0.0)
        return ("ERROR", build_time, 0, 0, [], build_time, 0.0)

    solve_start = time.perf_counter()
    try:
        sol = m.solve(log_output=False)
        solve_time = time.perf_counter() - solve_start
        total_time = build_time + solve_time
        
        if sol is None:
            return ("NO_SOLUTION", total_time, variables_count, constraints_count, [], build_time, solve_time)
            
        raw_status = str(sol.solve_details.status) if hasattr(sol, 'solve_details') and hasattr(sol.solve_details, 'status') else "OPTIMAL"
        status_upper = raw_status.upper()
        if "OPTIMAL" in status_upper:
            status = "OPTIMAL"
        elif "FEASIBLE" in status_upper:
            status = "FEASIBLE"
        else:
            status = raw_status
            
        queens = [(i, j) for i in range(n) for j in range(n) if sol.get_value(x[i, j]) > 0.5]
        return (status, total_time, variables_count, constraints_count, queens, build_time, solve_time)
    except Exception as e:
        solve_time = time.perf_counter() - solve_start
        total_time = build_time + solve_time
        err_str = str(e).lower()
        if "1016" in err_str or "license" in err_str or "limit" in err_str:
            return ("LICENSE_LIMIT", total_time, variables_count, constraints_count, [], build_time, solve_time)
        return ("ERROR", total_time, variables_count, constraints_count, [], build_time, solve_time)
