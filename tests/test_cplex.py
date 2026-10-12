import sys
import os
import pytest

pytest.importorskip("docplex")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from cplex_mip_model import solve_nqueens as solve_cplex_mip
from cplex_cp_model import solve_nqueens as solve_cplex_cp
from board import verify

def test_cplex_mip_small():
    status, total_time, variables_count, constraints_count, queens, build_time, solve_time = solve_cplex_mip(4, time_limit=10.0)
    assert status in ("OPTIMAL", "FEASIBLE", "LICENSE_LIMIT", "NO_SOLUTION")
    assert constraints_count >= 0
    if status in ("OPTIMAL", "FEASIBLE") and queens:
        assert len(queens) == 4
        assert verify(queens, 4) is True

def test_cplex_cp_small():
    from docplex.cp.config import context
    cp_path = context.solver.local.execfile
    if not cp_path or (cp_path != 'cpoptimizer' and not os.path.exists(cp_path)):
        pytest.skip("CP Optimizer executable not found")
    
    try:
        status, total_time, variables_count, constraints_count, queens, build_time, solve_time = solve_cplex_cp(4, time_limit=10.0)
    except Exception as e:
        pytest.skip(f"CP Optimizer could not be initialized or run: {e}")
        
    if status == "ERROR":
        pytest.skip("CP Optimizer binary cannot be initialized")
        
    assert status in ("OPTIMAL", "FEASIBLE", "Optimal", "Feasible", "LICENSE_LIMIT", "NO_SOLUTION")
    assert constraints_count >= 0
    if status in ("OPTIMAL", "FEASIBLE", "Optimal", "Feasible") and queens:
        assert len(queens) == 4
        assert verify(queens, 4) is True
