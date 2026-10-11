"""
MIP model builder and solver for N-Queens using OR-Tools linear solver (CBC / SCIP / Gurobi / CPLEX).
"""

import time
import os
import importlib.util
import sys
from ortools.linear_solver import pywraplp

current_dir = os.path.dirname(__file__)

def _load_module(name, filename):
    filepath = os.path.join(current_dir, filename)
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

board_mod = _load_module("mip_model_board", "board.py")
verify = board_mod.verify
cell_from_var = board_mod.cell_from_var

CONFIGURATIONS = {
    "binary",
    "matrix",
    "integer",
}

SOLVERS = {
    "cbc": "CBC",
    "scip": "SCIP",
    "gurobi": "GUROBI",
    "cplex": "CPLEX",
}

class NQueensMIP:
    """
    Builder and solver for N-Queens problem using OR-Tools Linear Solver (`pywraplp.Solver`).
    """
    def __init__(self, n: int, config: str = "binary", solver_name: str = "cbc"):
        self.n = n
        self.config_name = config.lower()
        if self.config_name not in CONFIGURATIONS:
            raise ValueError(f"Unknown configuration: {config}. Choose from {sorted(list(CONFIGURATIONS))}")
        
        self.requested_solver = solver_name.lower()
        backend_name = SOLVERS.get(self.requested_solver, "CBC")
        
        # Try creating requested solver
        solver = pywraplp.Solver.CreateSolver(backend_name)
        if solver is None:
            # Graceful fallback to CBC or SCIP
            fallback_backend = "SCIP" if backend_name != "SCIP" else "CBC"
            solver = pywraplp.Solver.CreateSolver(fallback_backend)
            if solver is None:
                solver = pywraplp.Solver.CreateSolver("CBC")
            self.solver_name = fallback_backend.lower()
        else:
            self.solver_name = self.requested_solver

        self.solver = solver
        self.variables = None
        self.stats = {}
        self.variables_count = 0
        self.constraints_count = 0

    def build(self):
        """Builds the MIP model for N-Queens with the selected configuration."""
        n = self.n
        solver = self.solver
        
        if self.config_name in ("binary", "matrix"):
            x = [[solver.BoolVar(f"x_{i}_{j}") for j in range(n)] for i in range(n)]
            self.variables = x
            
            # Exactly one queen per row
            for i in range(n):
                solver.Add(solver.Sum(x[i][j] for j in range(n)) == 1)
                
            # Exactly one queen per column
            for j in range(n):
                solver.Add(solver.Sum(x[i][j] for i in range(n)) == 1)
                
            # At most one queen per diagonal
            diags1 = {}
            diags2 = {}
            for i in range(n):
                for j in range(n):
                    diags1.setdefault(i - j, []).append(x[i][j])
                    diags2.setdefault(i + j, []).append(x[i][j])
                    
            for d_vars in diags1.values():
                solver.Add(solver.Sum(d_vars) <= 1)
            for d_vars in diags2.values():
                solver.Add(solver.Sum(d_vars) <= 1)
                
        elif self.config_name == "integer":
            # Integer variables q[i] in [0, n-1] plus binary indicator variables x[i][j]
            q = [solver.IntVar(0, n - 1, f"q_{i}") for i in range(n)]
            x = [[solver.BoolVar(f"x_{i}_{j}") for j in range(n)] for i in range(n)]
            self.variables = (q, x)
            
            for i in range(n):
                # Exactly one column per row
                solver.Add(solver.Sum(x[i][j] for j in range(n)) == 1)
                # Link q[i] to x[i][j]
                solver.Add(q[i] == solver.Sum(j * x[i][j] for j in range(n)))
                
            for j in range(n):
                # Exactly one queen per column
                solver.Add(solver.Sum(x[i][j] for i in range(n)) == 1)
                
            # At most one queen per diagonal
            diags1 = {}
            diags2 = {}
            for i in range(n):
                for j in range(n):
                    diags1.setdefault(i - j, []).append(x[i][j])
                    diags2.setdefault(i + j, []).append(x[i][j])
                    
            for d_vars in diags1.values():
                solver.Add(solver.Sum(d_vars) <= 1)
            for d_vars in diags2.values():
                solver.Add(solver.Sum(d_vars) <= 1)

        self.variables_count = solver.NumVariables()
        self.constraints_count = solver.NumConstraints()
        return solver

    def solve(self, time_limit: float = None) -> dict:
        """Solves the N-Queens problem and returns solving statistics and solution."""
        if self.variables is None:
            self.build()
            
        if time_limit is not None:
            self.solver.set_time_limit(int(time_limit * 1000))
            
        start_time = time.time()
        status = self.solver.Solve()
        elapsed = time.time() - start_time
        
        status_map = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        }
        status_name = status_map.get(status, str(status))
        is_sat = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
        
        queens = []
        if is_sat:
            if self.config_name in ("binary", "matrix"):
                board_dict = {}
                x = self.variables
                for i in range(self.n):
                    for j in range(self.n):
                        if x[i][j].solution_value() > 0.5:
                            board_dict[i] = j
                queens = [board_dict.get(i, -1) for i in range(self.n)]
            elif self.config_name == "integer":
                q, x = self.variables
                queens = [int(round(var.solution_value())) for var in q]

        self.stats = {
            "status": status_name,
            "is_sat": is_sat,
            "model": self.variables,
            "queens": queens,
            "time": elapsed,
            "variables_count": self.variables_count,
            "constraints_count": self.constraints_count,
            "n": self.n,
            "config": self.config_name,
            "solver": self.solver_name,
        }
        return self.stats


def solve_nqueens(n: int, config: str = "binary", solver_name: str = "cbc", time_limit: float = None) -> dict:
    """
    Convenience function to build and solve N-Queens using MIP.
    """
    model = NQueensMIP(n, config=config, solver_name=solver_name)
    model.build()
    return model.solve(time_limit=time_limit)
