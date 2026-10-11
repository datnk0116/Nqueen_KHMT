"""
CP-SAT model builder and solver for N-Queens using OR-Tools CP-SAT.
"""

import time
import os
import importlib.util
import sys
from ortools.sat.python import cp_model

current_dir = os.path.dirname(__file__)

def _load_module(name, filename):
    filepath = os.path.join(current_dir, filename)
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

board_mod = _load_module("cpsat_model_board", "board.py")
verify = board_mod.verify
cell_from_var = board_mod.cell_from_var

CONFIGURATIONS = {
    "integer",
    "alldiff",
    "boolean",
    "matrix",
    "allowed",
    "assignments",
}

class NQueensCPSAT:
    """
    Builder and solver for N-Queens problem using OR-Tools CP-SAT (`ortools.sat.python.cp_model.CpModel`).
    """
    def __init__(self, n: int, config: str = "integer"):
        self.n = n
        self.config_name = config.lower()
        if self.config_name not in CONFIGURATIONS:
            raise ValueError(f"Unknown configuration: {config}. Choose from {sorted(list(CONFIGURATIONS))}")
        
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.variables = None
        self.stats = {}
        self.variables_count = 0
        self.constraints_count = 0

    def build(self):
        """Builds the CP-SAT model for N-Queens with the selected configuration."""
        n = self.n
        model = self.model
        
        if self.config_name in ("integer", "alldiff"):
            # Integer variables for queen column per row
            q = [model.NewIntVar(0, n - 1, f"q_{i}") for i in range(n)]
            self.variables = q
            
            # All different columns
            model.AddAllDifferent(q)
            # All different positive diagonals (q[i] + i)
            model.AddAllDifferent([model.NewConstant(i) + q[i] for i in range(n)])
            # All different negative diagonals (q[i] - i)
            model.AddAllDifferent([model.NewConstant(-i) + q[i] for i in range(n)])
            
        elif self.config_name in ("boolean", "matrix"):
            # Boolean variables x[i][j]
            x = [[model.NewBoolVar(f"x_{i}_{j}") for j in range(n)] for i in range(n)]
            self.variables = x
            
            # Exactly one queen per row
            for i in range(n):
                model.Add(sum(x[i][j] for j in range(n)) == 1)
                
            # Exactly one queen per column
            for j in range(n):
                model.Add(sum(x[i][j] for i in range(n)) == 1)
                
            # At most one queen per diagonal
            diags1 = {}
            diags2 = {}
            for i in range(n):
                for j in range(n):
                    diags1.setdefault(i - j, []).append(x[i][j])
                    diags2.setdefault(i + j, []).append(x[i][j])
                    
            for d_vars in diags1.values():
                model.Add(sum(d_vars) <= 1)
            for d_vars in diags2.values():
                model.Add(sum(d_vars) <= 1)
                
        elif self.config_name in ("allowed", "assignments"):
            # Integer variables q[i] with AddAllDifferent and AddAllowedAssignments
            q = [model.NewIntVar(0, n - 1, f"q_{i}") for i in range(n)]
            self.variables = q
            
            model.AddAllDifferent(q)
            
            for a in range(n):
                for b in range(a + 1, n):
                    row_diff = b - a
                    allowed_tuples = []
                    for ca in range(n):
                        for cb in range(n):
                            if ca != cb and abs(ca - cb) != row_diff:
                                allowed_tuples.append((ca, cb))
                    model.AddAllowedAssignments([q[a], q[b]], allowed_tuples)

        proto = model.Proto()
        self.variables_count = len(proto.variables)
        self.constraints_count = len(proto.constraints)
        return model

    def solve(self, time_limit: float = None) -> dict:
        """Solves the N-Queens problem and returns solving statistics and solution."""
        if self.variables is None:
            self.build()
            
        if time_limit is not None:
            self.solver.parameters.max_time_in_seconds = time_limit
            
        start_time = time.time()
        status = self.solver.Solve(self.model)
        elapsed = time.time() - start_time
        
        status_name = self.solver.StatusName(status)
        is_sat = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        
        queens = []
        if is_sat:
            if self.config_name in ("integer", "alldiff", "allowed", "assignments"):
                queens = [self.solver.Value(var) for var in self.variables]
            elif self.config_name in ("boolean", "matrix"):
                board_dict = {}
                for i in range(self.n):
                    for j in range(self.n):
                        if self.solver.Value(self.variables[i][j]) == 1:
                            board_dict[i] = j
                queens = [board_dict.get(i, -1) for i in range(self.n)]

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
            "solver": "cpsat",
        }
        return self.stats


def solve_nqueens(n: int, config: str = "integer", time_limit: float = None) -> dict:
    """
    Convenience function to build and solve N-Queens using CP-SAT.
    """
    model = NQueensCPSAT(n, config=config)
    model.build()
    return model.solve(time_limit=time_limit)
