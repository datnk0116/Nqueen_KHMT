import sys
import os
import itertools
import importlib.util
import pytest

# Load src/encodings.py explicitly to avoid standard library encodings package conflict
encodings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/encodings.py'))
spec = importlib.util.spec_from_file_location("custom_encodings", encodings_path)
encodings = importlib.util.module_from_spec(spec)
sys.modules["custom_encodings"] = encodings
spec.loader.exec_module(encodings)

# Add src dir and parent dir to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from board import verify

from pysat.formula import IDPool
from pysat.solvers import Minisat22
from custom_encodings import (
    amo_pairwise,
    amo_binary,
    amo_sequential,
    amo_commander,
    amo_product,
    nqueens_encoding
)

AMO_FUNCTIONS = [
    ("pairwise", amo_pairwise),
    ("binary", amo_binary),
    ("sequential", amo_sequential),
    ("commander", amo_commander),
    ("product", amo_product),
]

@pytest.mark.parametrize("name, amo_func", AMO_FUNCTIONS)
@pytest.mark.parametrize("k", range(2, 11))
def test_amo_exhaustively(name, amo_func, k):
    """
    Test all possible truth value assignments to k base literals (k from 2 to 10)
    for each AMO encoding function.
    """
    pool = IDPool()
    lits = [pool.id() for _ in range(k)]
    clauses = amo_func(lits, pool)

    with Minisat22(bootstrap_with=clauses) as solver:
        # Generate all 2^k truth assignments
        for values in itertools.product([False, True], repeat=k):
            assumptions = []
            true_count = 0
            for lit, val in zip(lits, values):
                if val:
                    assumptions.append(lit)
                    true_count += 1
                else:
                    assumptions.append(-lit)

            is_sat = solver.solve(assumptions=assumptions)
            
            if true_count <= 1:
                assert is_sat is True, f"Failed for {name} with k={k}, true_count={true_count}, assumptions={assumptions}"
            else:
                assert is_sat is False, f"Failed for {name} with k={k}, true_count={true_count}, assumptions={assumptions}"

@pytest.mark.parametrize("name, amo_func", AMO_FUNCTIONS)
@pytest.mark.parametrize("n", [4, 8])
def test_nqueens_encoding(name, amo_func, n):
    """
    Test generating and solving N-Queens encoding for n=4 and n=8 using each AMO function.
    """
    pool = IDPool()
    clauses = nqueens_encoding(n, amo_func, pool)

    with Minisat22(bootstrap_with=clauses) as solver:
        assert solver.solve() is True, f"N-Queens n={n} with {name} encoding should be SAT"
        model = solver.get_model()
        assert verify(model, n=n) is True, f"N-Queens n={n} model verification failed for {name}"

def test_pairwise_formula_and_single_group_counts():
    """
    Validates:
    1. Pairwise total clause formula: 2*N*C(N,2) + 2*(2*C(N,3) + C(N,2)) + N for N in {8, 16, 64}.
    2. Single-group formulas for binary (n*ceil(log2 n) clauses, ceil(log2 n) aux vars)
       and sequential counter (3n - 4 clauses, n - 1 aux vars).
    """
    import math

    def c(n, k):
        return math.comb(n, k)

    # 1. Pairwise N-Queens total clauses
    for n in [8, 16, 64]:
        expected_clauses = 2 * n * c(n, 2) + 2 * (2 * c(n, 3) + c(n, 2)) + n
        pool = IDPool()
        clauses = nqueens_encoding(n, amo_pairwise, pool)
        assert len(clauses) == expected_clauses, f"Pairwise N={n}: got {len(clauses)}, expected {expected_clauses}"

    # 2. Single-group binary and sequential formulas
    for k in [2, 3, 5, 8, 10, 16]:
        # Binary
        pool_bin = IDPool()
        lits_bin = [pool_bin.id() for _ in range(k)]
        clauses_bin = amo_binary(lits_bin, pool_bin)
        m = math.ceil(math.log2(k))
        assert len(clauses_bin) == k * m
        assert pool_bin.top - k == m

        # Sequential
        pool_seq = IDPool()
        lits_seq = [pool_seq.id() for _ in range(k)]
        clauses_seq = amo_sequential(lits_seq, pool_seq)
        assert len(clauses_seq) == 3 * k - 4
        assert pool_seq.top - k == k - 1

