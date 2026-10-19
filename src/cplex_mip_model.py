"""
CPLEX MIP model builder and solver for N-Queens.
"""
import time
from docplex.mp.model import Model

def solve_nqueens(n: int, time_limit: float = None):
    """
    Solves N-Queens using CPLEX MIP model.
    Returns dict: {"status", "total_time", "variables_count", "constraints_count", "queens", "build_time", "solve_time", "solver_reported_time"}
    """
    build_start = time.perf_counter()
    variables_count = 0
    constraints_count = 0
    m = None
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
        status = "LICENSE_LIMIT" if ("1016" in err_str or "license" in err_str or "limit" in err_str) else "ERROR"
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
        if time_limit is not None:
            m.set_time_limit(max(0.001, time_limit - build_time))
        sol = m.solve(log_output=False)
        solve_time = time.perf_counter() - solve_start
        total_time = build_time + solve_time
        
        solver_reported_time = None
        if sol is not None and hasattr(sol, 'solve_details') and hasattr(sol.solve_details, 'time'):
            solver_reported_time = sol.solve_details.time
        elif hasattr(m, 'solve_details') and hasattr(m.solve_details, 'time'):
            solver_reported_time = m.solve_details.time

        if sol is None:
            status = "NO_SOLUTION"
            if m is not None and hasattr(m, 'solve_details') and hasattr(m.solve_details, 'status'):
                detail_status = str(m.solve_details.status).upper()
                if "TIME" in detail_status or "LIMIT" in detail_status:
                    status = "TIME_LIMIT"
            elif time_limit is not None and solve_time >= time_limit * 0.95:
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
            
        raw_status = str(sol.solve_details.status) if hasattr(sol, 'solve_details') and hasattr(sol.solve_details, 'status') else "OPTIMAL"
        status_upper = raw_status.upper()
        if "OPTIMAL" in status_upper:
            status = "OPTIMAL"
        elif "FEASIBLE" in status_upper:
            status = "FEASIBLE"
        else:
            status = raw_status
            
        queens = [(i, j) for i in range(n) for j in range(n) if sol.get_value(x[i, j]) > 0.5]
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
        status = "LICENSE_LIMIT" if ("1016" in err_str or "license" in err_str or "limit" in err_str) else "ERROR"
        solver_reported_time = None
        if m is not None and hasattr(m, 'solve_details') and hasattr(m.solve_details, 'time'):
            solver_reported_time = m.solve_details.time
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
