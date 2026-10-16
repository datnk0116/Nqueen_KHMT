import sys
import os
import pytest

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from board import verify
from mip_cp_models import NQueensMIP, solve_nqueens, CONFIGURATIONS, SOLVERS

CONFIG_NAMES = list(CONFIGURATIONS)
SOLVER_NAMES = list(SOLVERS.keys())

@pytest.mark.parametrize("config", CONFIG_NAMES)
@pytest.mark.parametrize("solver_name", SOLVER_NAMES)
@pytest.mark.parametrize("n", [4, 8])
def test_solve_nqueens_mip(config, solver_name, n):
    """
    Test solving N-Queens for n=4 and n=8 across all MIP configurations and solvers (with fallback).
    """
    result = solve_nqueens(n, config=config, solver_name=solver_name)
    
    assert result["status"] in ("OPTIMAL", "FEASIBLE"), f"Failed for n={n}, config={config}, solver={solver_name}, status={result['status']}"
    assert result["is_sat"] is True
    assert result["n"] == n
    assert result["config"] == config
    assert result["variables_count"] > 0
    assert result["constraints_count"] > 0
    assert isinstance(result["time"], float)
    
    # Verify the solution queens array
    queens = result["queens"]
    assert len(queens) == n
    assert verify(queens, n=n) is True, f"Solution verification failed for n={n}, config={config}, solver={solver_name}: {queens}"

def test_nqueens_mip_class():
    """
    Test NQueensMIP class builder and solver methods directly.
    """
    model = NQueensMIP(n=4, config="binary", solver_name="cbc")
    solver_obj = model.build()
    assert solver_obj is not None
    assert model.variables_count >= 4
    
    stats = model.solve()
    assert stats["is_sat"] is True
    assert verify(stats["queens"], n=4) is True

def test_invalid_config():
    """
    Test that invalid configuration raises ValueError.
    """
    with pytest.raises(ValueError):
        solve_nqueens(4, config="nonexistent")
