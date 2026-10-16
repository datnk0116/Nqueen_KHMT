"""
SAT model builder and solver for N-Queens using PySAT and various AMO encodings.
"""

import time
import threading
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
        """
        Solves the N-Queens problem and returns solving statistics and solution.
        If time_limit is provided (in seconds), cooperative interrupt is scheduled via
        threading.Timer and solver.interrupt() (using solve_limited).
        For backends lacking interrupt support in PySAT (such as CaDiCaL and Kissat),
        cooperative interrupt raises NotImplementedError and normal solve() is invoked,
        relying on the parent process's hard timeout.
        """
        if not self.clauses:
            self.build()
            
        solver_cls = SOLVERS.get(self.solver_name)
        start_time = time.perf_counter()
        
        is_sat = False
        model = None
        
        SOLVERS_WITH_INTERRUPT = {"minisat22", "glucose3", "glucose4", "maplecm"}

        def _run_solve(solver_instance):
            nonlocal is_sat, model
            timer = None
            lock = threading.Lock()
            interrupted = False

            def _interrupt_callback():
                nonlocal interrupted
                with lock:
                    if not interrupted:
                        interrupted = True
                        try:
                            solver_instance.interrupt()
                        except Exception:
                            pass

            if time_limit is not None and time_limit > 0:
                if self.solver_name in SOLVERS_WITH_INTERRUPT:
                    try:
                        timer = threading.Timer(time_limit, _interrupt_callback)
                        timer.daemon = True
                        timer.start()
                        is_sat = solver_instance.solve_limited(expect_interrupt=True)
                    except NotImplementedError:
                        is_sat = solver_instance.solve()
                    finally:
                        if timer is not None:
                            timer.cancel()
                        with lock:
                            interrupted = True
                else:
                    # PySAT backends like CaDiCaL and Kissat do not implement interrupt().
                    # For these backends, cooperative interrupt is unavailable,
                    # and timeout enforcement relies on the parent process hard timeout.
                    is_sat = solver_instance.solve()
            else:
                is_sat = solver_instance.solve()

            if is_sat:
                model = solver_instance.get_model()

        try:
            if solver_cls:
                with solver_cls(bootstrap_with=self.clauses) as solver:
                    _run_solve(solver)
            else:
                with Solver(name=self.solver_name, bootstrap_with=self.clauses) as solver:
                    _run_solve(solver)
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            err_str = str(e).lower()
            if "out of memory" in err_str or "bad_alloc" in err_str:
                status = "MEMOUT"
            elif "license" in err_str or "size limit" in err_str:
                status = "LICENSE_LIMIT"
            else:
                status = "ERROR"
            self.stats = {
                "status": status,
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

        elapsed = time.perf_counter() - start_time

        if is_sat is None:
            # Solver was interrupted by time_limit
            self.stats = {
                "status": "TIME_LIMIT",
                "is_sat": False,
                "model": None,
                "queens": [],
                "time": elapsed,
                "variables_count": self.variables_count,
                "clauses_count": self.clauses_count,
                "n": self.n,
                "encoding": self.encoding_name,
                "solver": self.solver_name,
            }
            return self.stats
        
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
            "is_sat": bool(is_sat),
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
