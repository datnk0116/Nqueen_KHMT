import sys
import os
import pytest

pytest.importorskip("gurobipy")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from gurobi_model import solve_nqueens
from board import verify

def test_gurobi_small():
    res = solve_nqueens(4, time_limit=10.0)
    status = res["status"]
    constraints_count = res["constraints_count"]
    queens = res["queens"]
    assert status in ("OPTIMAL", "SUBOPTIMAL", "LICENSE_LIMIT")
    assert constraints_count >= 0
    if status in ("OPTIMAL", "SUBOPTIMAL"):
        assert len(queens) == 4
        assert verify(queens, 4) is True
