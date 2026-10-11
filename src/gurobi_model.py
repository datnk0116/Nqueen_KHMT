"""
Gurobi model builder and solver for N-Queens.
"""
import time
import gurobipy as gp
from gurobipy import GRB

def solve_nqueens(n: int, time_limit: float = None):
    """
    Solves N-Queens using Gurobi MIP model.
    Returns: (status, time, variables_count, queens, build_time, solve_time)
    """
    build_start = time.time()
    try:
        m = gp.Model("nqueens_gurobi")
        m.Params.OutputFlag = 0
        if time_limit is not None:
            m.Params.TimeLimit = time_limit
            
        x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
        for i in range(n):
            m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
            m.addConstr(gp.quicksum(x[j, i] for j in range(n)) == 1)
        for d in range(-(n - 1), n):
            cells = [x[i, i - d] for i in range(n) if 0 <= i - d < n]
            if len(cells) > 1:
                m.addConstr(gp.quicksum(cells) <= 1)
        for s in range(2 * n - 1):
            cells = [x[i, s - i] for i in range(n) if 0 <= s - i < n]
            if len(cells) > 1:
                m.addConstr(gp.quicksum(cells) <= 1)
                
        m.update()
        variables_count = m.NumVars
        constraints_count = m.NumConstrs
        build_time = time.time() - build_start
    except Exception as e:
        build_time = time.time() - build_start
        err_str = str(e).lower()
        if "license" in err_str or "limit" in err_str or "size" in err_str:
            return ("LICENSE_LIMIT", build_time, 0, 0, [], build_time, 0.0)
        return ("ERROR", build_time, 0, 0, [], build_time, 0.0)

    solve_start = time.time()
    try:
        m.optimize()
        solve_time = time.time() - solve_start
        total_time = build_time + solve_time
        
        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.TIME_LIMIT: "TIME_LIMIT",
        }
        status = status_map.get(m.Status, str(m.Status))
        
        queens = []
        if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or (hasattr(m, 'SolCount') and m.SolCount > 0):
            queens = [(i, j) for i in range(n) for j in range(n) if x[i, j].X > 0.5]
            
        return (status, total_time, variables_count, constraints_count, queens, build_time, solve_time)
    except gp.GurobiError as e:
        solve_time = time.time() - solve_start
        total_time = build_time + solve_time
        err_str = str(e).lower()
        if "license" in err_str or "limit" in err_str or "size" in err_str:
            return ("LICENSE_LIMIT", total_time, variables_count, constraints_count, [], build_time, solve_time)
        return ("ERROR", total_time, variables_count, constraints_count, [], build_time, solve_time)
    except Exception as e:
        solve_time = time.time() - solve_start
        total_time = build_time + solve_time
        return ("ERROR", total_time, variables_count, constraints_count, [], build_time, solve_time)
