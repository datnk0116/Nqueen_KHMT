"""
SAT model builder and solver for N-Queens using PySAT and various AMO encodings.
"""

import time
import os
import importlib.util
from pysat.formula import IDPool
from pysat.solvers import Solver, Minisat22, Glucose3, Glucose4, Cadical153, Kissat404, MapleCM

# Load local board and encodings modules explicitly to avoid stdlib encodings conflict
current_dir = os.path.dirname(__file__)

def _load_module(name, filename):
    filepath = os.path.join(current_dir, filename)
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

import sys
board_mod = _load_module("sat_model_board", "board.py")
encodings_mod = _load_module("sat_model_encodings", "encodings.py")

verify = board_mod.verify
cell_from_var = board_mod.cell_from_var

amo_pairwise = encodings_mod.amo_pairwise
amo_binary = encodings_mod.amo_binary
amo_sequential = encodings_mod.amo_sequential
amo_commander = encodings_mod.amo_commander
amo_product = encodings_mod.amo_product
nqueens_encoding = encodings_mod.nqueens_encoding

ENCODINGS = {
    "pairwise": amo_pairwise,
    "binary": amo_binary,
    "sequential": amo_sequential,
    "commander": amo_commander,
    "product": amo_product,
}

SOLVERS = {
    "minisat22": Minisat22,
    "glucose3": Glucose3,
    "glucose4": Glucose4,
    "cadical": Cadical153,
    "cadical153": Cadical153,
    "kissat": Kissat404,
    "maplecm": MapleCM,
}

class NQueensSAT:
    """
    Builder and solver for N-Queens problem using SAT encodings.
    """
    def __init__(self, n: int, encoding: str = "sequential", solver_name: str = "glucose3"):
        self.n = n
        self.encoding_name = encoding.lower()
        self.solver_name = solver_name.lower()
        
        if self.encoding_name not in ENCODINGS:
            raise ValueError(f"Unknown encoding: {encoding}. Choose from {list(ENCODINGS.keys())}")
            
        self.amo_func = ENCODINGS[self.encoding_name]
        self.pool = IDPool()
        self.clauses = []
        self.variables_count = 0
        self.clauses_count = 0
        self.stats = {}

    def build(self) -> list[list[int]]:
        """Builds the CNF clauses for N-Queens with the selected encoding."""
        self.clauses = nqueens_encoding(self.n, self.amo_func, self.pool)
        self.variables_count = self.pool.top
        self.clauses_count = len(self.clauses)
        return self.clauses

    def solve(self, time_limit: float = None) -> dict:
        """Solves the N-Queens problem and returns solving statistics and solution."""
        if not self.clauses:
            self.build()
            
        solver_cls = SOLVERS.get(self.solver_name)
        start_time = time.time()
        
        is_sat = False
        model = None
        
        try:
            if solver_cls:
                with solver_cls(bootstrap_with=self.clauses) as solver:
                    is_sat = solver.solve()
                    if is_sat:
                        model = solver.get_model()
            else:
                with Solver(name=self.solver_name, bootstrap_with=self.clauses) as solver:
                    is_sat = solver.solve()
                    if is_sat:
                        model = solver.get_model()
        except Exception as e:
            elapsed = time.time() - start_time
            self.stats = {
                "status": "ERROR",
                "is_sat": False,
                "model": None,
                "queens": [],
                "time": elapsed,
                "variables_count": self.variables_count,
                "clauses_count": self.clauses_count,
                "n": self.n,
                "encoding": self.encoding_name,
                "solver": self.solver_name,
                "error": str(e)
            }
            return self.stats

        elapsed = time.time() - start_time
        
        queens = []
        if is_sat and model:
            active = [v for v in model if v > 0 and v <= self.n * self.n]
            board_dict = {}
            for v in active:
                try:
                    r, c = cell_from_var(v, self.n)
                    board_dict[r] = c
                except ValueError:
                    pass
            queens = [board_dict.get(r, -1) for r in range(self.n)]

        self.stats = {
            "status": "SAT" if is_sat else "UNSAT",
            "is_sat": is_sat,
            "model": model,
            "queens": queens,
            "time": elapsed,
            "variables_count": self.variables_count,
            "clauses_count": self.clauses_count,
            "n": self.n,
            "encoding": self.encoding_name,
            "solver": self.solver_name,
        }
        return self.stats


def solve_nqueens(n: int, encoding: str = "sequential", solver_name: str = "glucose3") -> dict:
    """
    Convenience function to build and solve N-Queens for given n, encoding, and solver.
    """
    model = NQueensSAT(n, encoding=encoding, solver_name=solver_name)
    model.build()
    return model.solve()
