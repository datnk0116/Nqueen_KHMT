"""Kiểm tra CPLEX MIP và CPLEX CP Optimizer qua N-Queens.

Chạy:  python check_cplex.py [N]      (mặc định N = 40)
N = 40 -> 1600 biến, vượt giới hạn 1000 của bản Community Edition.
Nếu CPLEX là bản đầy đủ, MIP phải giải được mà không báo lỗi 1016.
"""
import sys
import time
import os

from docplex.mp.model import Model
from docplex.cp.model import CpoModel
from docplex.cp.config import context

cp_path = os.environ.get("CPLEX_CP_EXECUTABLE")
if cp_path and os.path.exists(cp_path):
    context.solver.local.execfile = cp_path


def is_valid(queens, n):
    """Kiểm tra n hậu: khác cột, khác chéo i-j, khác chéo i+j (mỗi hàng 1 hậu)."""
    return (len(queens) == n
            and len({c for _, c in queens}) == n
            and len({r - c for r, c in queens}) == n
            and len({r + c for r, c in queens}) == n)


def solve_mip(n):
    m = Model("nqueens_mip")
    x = m.binary_var_matrix(n, n, "x")
    for i in range(n):
        m.add_constraint(m.sum(x[i, j] for j in range(n)) == 1)  # mỗi hàng
        m.add_constraint(m.sum(x[j, i] for j in range(n)) == 1)  # mỗi cột
    for d in range(-(n - 1), n):                                  # chéo i - j = d
        cells = [x[i, i - d] for i in range(n) if 0 <= i - d < n]
        if len(cells) > 1:
            m.add_constraint(m.sum(cells) <= 1)
    for s in range(2 * n - 1):                                    # chéo i + j = s
        cells = [x[i, s - i] for i in range(n) if 0 <= s - i < n]
        if len(cells) > 1:
            m.add_constraint(m.sum(cells) <= 1)
    start = time.time()
    sol = m.solve()
    elapsed = time.time() - start
    if sol is None:
        return None, elapsed
    queens = [(i, j) for i in range(n) for j in range(n)
              if sol.get_value(x[i, j]) > 0.5]
    return queens, elapsed


def solve_cp(n):
    m = CpoModel("nqueens_cp")
    q = [m.integer_var(0, n - 1, f"q{i}") for i in range(n)]
    m.add(m.all_diff(q))
    m.add(m.all_diff([q[i] + i for i in range(n)]))
    m.add(m.all_diff([q[i] - i for i in range(n)]))
    start = time.time()
    sol = m.solve(TimeLimit=60, LogVerbosity="Quiet")
    elapsed = time.time() - start
    if not sol.is_solution():
        return None, elapsed
    return [(i, sol[q[i]]) for i in range(n)], elapsed


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    for name, fn in (("CPLEX MIP", solve_mip), ("CPLEX CP", solve_cp)):
        try:
            queens, t = fn(n)
            status = "OK, nghiem hop le" if queens and is_valid(queens, n) else "KHONG co nghiem hop le"
            print(f"{name:10s} N={n}: {status} ({t:.3f}s)")
        except Exception as e:  # in nguyên lỗi để biết nguyên nhân (vd. mã 1016)
            print(f"{name:10s} N={n}: LOI -> {type(e).__name__}: {e}")
