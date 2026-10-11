"""
Board representation and helper functions for N-Queens SAT and constraint programming benchmarks.
"""

def var(i: int, j: int, n: int) -> int:
    """
    Returns the 1-based SAT variable index for cell (i, j) on an n x n board.
    Assumes 0-indexed i (row) and j (col) in range [0, n-1].
    var(i, j, n) = i * n + j + 1
    """
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError(f"Cell ({i}, {j}) out of bounds for {n}x{n} board.")
    return i * n + j + 1

def cell_from_var(v: int, n: int) -> tuple[int, int]:
    """
    Returns (i, j) 0-indexed coordinates for a 1-based variable index v.
    """
    if not (1 <= v <= n * n):
        raise ValueError(f"Variable index {v} out of bounds for {n}x{n} board.")
    v0 = v - 1
    return v0 // n, v0 % n

def get_row_groups(n: int) -> list[list[int]]:
    """Returns list of row groups, where each group is a list of variable indices."""
    return [[var(i, j, n) for j in range(n)] for i in range(n)]

def get_col_groups(n: int) -> list[list[int]]:
    """Returns list of column groups, where each group is a list of variable indices."""
    return [[var(i, j, n) for i in range(n)] for j in range(n)]

def get_diag1_groups(n: int) -> list[list[int]]:
    """
    Returns list of diagonal groups (main diagonals / difference i - j).
    For each k in [-(n-1), n-1], cells (i, j) where i - j == k.
    """
    diags = {}
    for i in range(n):
        for j in range(n):
            k = i - j
            diags.setdefault(k, []).append(var(i, j, n))
    return list(diags.values())

def get_diag2_groups(n: int) -> list[list[int]]:
    """
    Returns list of anti-diagonal groups (anti-diagonals / sum i + j).
    For each k in [0, 2n-2], cells (i, j) where i + j == k.
    """
    diags = {}
    for i in range(n):
        for j in range(n):
            k = i + j
            diags.setdefault(k, []).append(var(i, j, n))
    return list(diags.values())

def verify(queens, n: int = None) -> bool:
    """
    Verifies if the given queen configuration is a valid solution for the n-queens problem.
    
    Supported representations for `queens`:
    1. List of column indices per row (length n, e.g., [4, 0, 3, 5, 7, 1, 6, 2]).
    2. List of (row, col) tuples or lists (length n).
    3. List of integers (SAT model/assignment or variable indices).
    """
    if queens is None:
        return False
        
    coords = []
    
    if all(isinstance(x, int) for x in queens):
        if n is None and all(0 <= x < len(queens) for x in queens):
            n = len(queens)
            coords = [(r, c) for r, c in enumerate(queens)]
        elif any(x < 0 for x in queens) or (n and len(queens) >= n * n):
            active_vars = [x for x in queens if x > 0]
            if n is None:
                import math
                n = int(math.isqrt(max(active_vars))) if active_vars else 0
            for v in active_vars:
                try:
                    r, c = cell_from_var(v, n)
                    coords.append((r, c))
                except ValueError:
                    pass
        else:
            if n is not None and len(queens) == n and all(0 <= x < n for x in queens):
                coords = [(r, c) for r, c in enumerate(queens)]
            else:
                if n is None:
                    import math
                    n = int(math.isqrt(max(queens))) if queens else 0
                for v in queens:
                    try:
                        r, c = cell_from_var(v, n)
                        coords.append((r, c))
                    except ValueError:
                        pass
    elif all(isinstance(item, (tuple, list)) and len(item) == 2 for item in queens):
        coords = list(queens)
    else:
        coords = [(r, c) for r, c in enumerate(queens)]
        
    if n is None:
        n = len(coords)
        
    if len(coords) != n:
        return False
        
    rows = set()
    cols = set()
    diag1s = set()
    diag2s = set()
    
    for r, c in coords:
        if not (0 <= r < n and 0 <= c < n):
            return False
        if r in rows or c in cols:
            return False
        d1 = r - c
        d2 = r + c
        if d1 in diag1s or d2 in diag2s:
            return False
        rows.add(r)
        cols.add(c)
        diag1s.add(d1)
        diag2s.add(d2)
        
    return len(rows) == n
