import sys
import os
import pytest

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from board import verify
from sat_model import NQueensSAT, solve_nqueens, ENCODINGS, SOLVERS

ENCODING_NAMES = list(ENCODINGS.keys())
SOLVER_NAMES = ["glucose3", "minisat22", "cadical"]

@pytest.mark.parametrize("encoding", ENCODING_NAMES)
@pytest.mark.parametrize("solver_name", SOLVER_NAMES)
@pytest.mark.parametrize("n", [4, 8])
def test_solve_nqueens(encoding, solver_name, n):
    """
    Test solving N-Queens for n=4 and n=8 across all encodings and major solvers.
    """
    result = solve_nqueens(n, encoding=encoding, solver_name=solver_name)
    
    assert result["status"] == "SAT", f"Failed for n={n}, encoding={encoding}, solver={solver_name}"
    assert result["is_sat"] is True
    assert result["n"] == n
    assert result["encoding"] == encoding
    assert result["solver"] == solver_name
    assert result["variables_count"] > 0
    assert result["clauses_count"] > 0
    assert isinstance(result["time"], float)
    
    # Verify the solution queens array
    queens = result["queens"]
    assert len(queens) == n
    assert verify(queens, n=n) is True, f"Solution verification failed for n={n}, encoding={encoding}, solver={solver_name}: {queens}"

def test_nqueens_sat_class():
    """
    Test NQueensSAT class builder and solver methods directly.
    """
    model = NQueensSAT(n=4, encoding="sequential", solver_name="glucose3")
    clauses = model.build()
    assert len(clauses) > 0
    assert model.variables_count >= 16
    
    stats = model.solve()
    assert stats["is_sat"] is True
    assert verify(stats["queens"], n=4) is True

def test_invalid_encoding():
    """
    Test that invalid encoding raises ValueError.
    """
    with pytest.raises(ValueError):
        solve_nqueens(4, encoding="nonexistent")
