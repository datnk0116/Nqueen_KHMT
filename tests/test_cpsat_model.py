import sys
import os
import pytest

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from board import verify
from cpsat_model import NQueensCPSAT, solve_nqueens, CONFIGURATIONS

CONFIG_NAMES = list(CONFIGURATIONS)

@pytest.mark.parametrize("config", CONFIG_NAMES)
@pytest.mark.parametrize("n", [4, 8])
def test_solve_nqueens_cpsat(config, n):
    """
    Test solving N-Queens for n=4 and n=8 across all CP-SAT configurations.
    """
    result = solve_nqueens(n, config=config)
    
    assert result["status"] == "OPTIMAL", f"Failed for n={n}, config={config}, status={result['status']}"
    assert result["is_sat"] is True
    assert result["n"] == n
    assert result["config"] == config
    assert result["solver"] == "cpsat"
    assert result["variables_count"] > 0
    assert result["constraints_count"] > 0
    assert isinstance(result["time"], float)
    
    # Verify the solution queens array
    queens = result["queens"]
    assert len(queens) == n
    assert verify(queens, n=n) is True, f"Solution verification failed for n={n}, config={config}: {queens}"

def test_nqueens_cpsat_class():
    """
    Test NQueensCPSAT class builder and solver methods directly.
    """
    model = NQueensCPSAT(n=4, config="integer")
    cp_model_obj = model.build()
    assert cp_model_obj is not None
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
