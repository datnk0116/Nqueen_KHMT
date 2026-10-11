"""
At-Most-One (AMO) constraint encodings for SAT solvers.
"""

import math
from pysat.formula import IDPool

def amo_pairwise(lits: list[int], pool: IDPool) -> list[list[int]]:
    """
    Pairwise AMO encoding: adds clauses [-u, -v] for all pairs of distinct literals.
    """
    clauses = []
    n = len(lits)
    for i in range(n):
        for j in range(i + 1, n):
            clauses.append([-lits[i], -lits[j]])
    return clauses

def amo_binary(lits: list[int], pool: IDPool) -> list[list[int]]:
    """
    Binary (bit-wise) AMO encoding using log2(n) auxiliary variables.
    """
    clauses = []
    n = len(lits)
    if n <= 1:
        return clauses
    m = math.ceil(math.log2(n))
    if m == 0:
        m = 1
    y = [pool.id() for _ in range(m)]
    for i, lit in enumerate(lits):
        for j in range(m):
            bit = (i >> j) & 1
            if bit:
                clauses.append([-lit, y[j]])
            else:
                clauses.append([-lit, -y[j]])
    return clauses

def amo_sequential(lits: list[int], pool: IDPool) -> list[list[int]]:
    """
    Sequential Counter (Sinz) AMO encoding.
    """
    clauses = []
    n = len(lits)
    if n <= 1:
        return clauses
    s = [pool.id() for _ in range(n - 1)]
    clauses.append([-lits[0], s[0]])
    for i in range(1, n - 1):
        clauses.append([-lits[i], s[i]])
        clauses.append([-s[i-1], s[i]])
        clauses.append([-lits[i], -s[i-1]])
    clauses.append([-lits[n-1], -s[n-2]])
    return clauses

def amo_commander(lits: list[int], pool: IDPool) -> list[list[int]]:
    """
    Commander AMO encoding.
    """
    clauses = []
    n = len(lits)
    if n <= 1:
        return clauses
    if n <= 3:
        return amo_pairwise(lits, pool)
    
    block_size = 3
    groups = [lits[i:i + block_size] for i in range(0, n, block_size)]
    commanders = []
    for group in groups:
        if len(group) == 1:
            commanders.append(group[0])
            continue
        clauses.extend(amo_pairwise(group, pool))
        c = pool.id()
        commanders.append(c)
        for lit in group:
            clauses.append([-lit, c])
            
    clauses.extend(amo_commander(commanders, pool))
    return clauses

def amo_product(lits: list[int], pool: IDPool) -> list[list[int]]:
    """
    Product AMO encoding (Bailleux & Boufkhad).
    """
    clauses = []
    n = len(lits)
    if n <= 1:
        return clauses
    p = math.ceil(math.sqrt(n))
    q = math.ceil(n / p)
    rows = [pool.id() for _ in range(p)]
    cols = [pool.id() for _ in range(q)]
    
    clauses.extend(amo_binary(rows, pool))
    clauses.extend(amo_binary(cols, pool))
    
    for i, lit in enumerate(lits):
        r = i // q
        c = i % q
        if r < p and c < q:
            clauses.append([-lit, rows[r]])
            clauses.append([-lit, cols[c]])
            
    return clauses

def nqueens_encoding(n: int, amo_func, pool: IDPool) -> list[list[int]]:
    """
    Generates CNF clauses for the n-queens problem using the specified AMO encoding function.
    """
    if pool.top < n * n:
        pool.top = n * n
    try:
        from .board import get_row_groups, get_col_groups, get_diag1_groups, get_diag2_groups, var
    except ImportError:
        from board import get_row_groups, get_col_groups, get_diag1_groups, get_diag2_groups, var
    clauses = []
    
    # Exactly one queen per row (At-Least-One + At-Most-One)
    for row in get_row_groups(n):
        clauses.append(row)  # At least one
        clauses.extend(amo_func(row, pool))
        
    # At most one queen per column
    for col in get_col_groups(n):
        clauses.extend(amo_func(col, pool))
        
    # At most one queen per diagonal
    for d1 in get_diag1_groups(n):
        if len(d1) > 1:
            clauses.extend(amo_func(d1, pool))
    for d2 in get_diag2_groups(n):
        if len(d2) > 1:
            clauses.extend(amo_func(d2, pool))
            
    return clauses
