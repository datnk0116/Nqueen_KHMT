import sys
import os

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from board import var, cell_from_var, get_row_groups, get_col_groups, get_diag1_groups, get_diag2_groups, verify

def test_var_and_cell_mapping():
    n = 4
    # Cell (0, 0) -> var 1
    assert var(0, 0, n) == 1
    assert cell_from_var(1, n) == (0, 0)
    
    # Cell (1, 2) -> var 1 * 4 + 2 + 1 = 7
    assert var(1, 2, n) == 7
    assert cell_from_var(7, n) == (1, 2)
    
    # Cell (3, 3) -> var 16
    assert var(3, 3, n) == 16
    assert cell_from_var(16, n) == (3, 3)

def test_groups():
    n = 4
    rows = get_row_groups(n)
    assert len(rows) == 4
    assert len(rows[0]) == 4
    
    cols = get_col_groups(n)
    assert len(cols) == 4
    assert len(cols[0]) == 4
    
    diag1 = get_diag1_groups(n)
    diag2 = get_diag2_groups(n)
    assert len(diag1) > 0
    assert len(diag2) > 0

def test_verify_valid_8x8():
    # A known valid 8x8 N-Queens solution (column index per row)
    valid_solution = [4, 0, 3, 5, 7, 1, 6, 2]
    assert verify(valid_solution, n=8) is True
    assert verify(valid_solution) is True  # auto-infer n

def test_verify_invalid_board():
    # Invalid: multiple queens in same column / diagonal conflicts
    invalid_solution_1 = [0, 0, 0, 0, 0, 0, 0, 0]
    assert verify(invalid_solution_1, n=8) is False
    
    # Invalid: diagonal conflict
    invalid_solution_2 = [0, 1, 2, 3, 4, 5, 6, 7]
    assert verify(invalid_solution_2, n=8) is False

def test_verify_sat_model():
    # Test verifying with 1-based variable indices or list of positive vars
    # For n=4, solution [1, 3, 0, 2] -> cols: 1, 3, 0, 2
    # row 0 col 1 -> var 2
    # row 1 col 3 -> var 8
    # row 2 col 0 -> var 9
    # row 3 col 2 -> var 15
    valid_4x4_vars = [2, 8, 9, 15]
    assert verify(valid_4x4_vars, n=4) is True
