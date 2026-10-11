import sys
import os
import pytest

pytest.importorskip("gurobipy")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from gurobi_model import solve_nqueens
from board import verify

def test_gurobi_small():
    status, total_time, variables_count, queens, build_time, solve_time = solve_nqueens(4, time_limit=10.0)
    assert status in ("OPTIMAL", "SUBOPTIMAL", "LICENSE_LIMIT")
    if status in ("OPTIMAL", "SUBOPTIMAL"):
        assert len(queens) == 4
        assert verify(queens, 4) is True
