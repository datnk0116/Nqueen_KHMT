"""Kiểm tra Gurobi qua N-Queens (mô hình nhị phân N*N biến).

Chạy:  python check_gurobi.py [N]      (mặc định N = 40)
- Bản gurobipy miễn phí: giải được khi số biến <= 2000 (N <= 44).
- N = 50 (2500 biến) chỉ chạy được khi đã có license đầy đủ.
"""
import sys
import time

import gurobipy as gp
from gurobipy import GRB


def is_valid(queens, n):
    """n hậu: khác cột, khác chéo i-j, khác chéo i+j (mỗi hàng 1 hậu)."""
    return (len(queens) == n
            and len({c for _, c in queens}) == n
            and len({r - c for r, c in queens}) == n
            and len({r + c for r, c in queens}) == n)


def solve_gurobi(n):
    m = gp.Model("nqueens")
    m.Params.OutputFlag = 0
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    for i in range(n):
        m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)  # mỗi hàng
        m.addConstr(gp.quicksum(x[j, i] for j in range(n)) == 1)  # mỗi cột
    for d in range(-(n - 1), n):                                   # chéo i - j = d
        cells = [x[i, i - d] for i in range(n) if 0 <= i - d < n]
        if len(cells) > 1:
            m.addConstr(gp.quicksum(cells) <= 1)
    for s in range(2 * n - 1):                                     # chéo i + j = s
        cells = [x[i, s - i] for i in range(n) if 0 <= s - i < n]
        if len(cells) > 1:
            m.addConstr(gp.quicksum(cells) <= 1)
    start = time.time()
    m.optimize()
    elapsed = time.time() - start
    if m.Status != GRB.OPTIMAL:
        return None, elapsed
    queens = [(i, j) for i in range(n) for j in range(n) if x[i, j].X > 0.5]
    return queens, elapsed


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print("gurobipy", ".".join(map(str, gp.gurobi.version())))
    try:
        queens, t = solve_gurobi(n)
        status = "OK, nghiem hop le" if queens and is_valid(queens, n) else "KHONG co nghiem hop le"
        print(f"Gurobi N={n}: {status} ({t:.3f}s)")
    except gp.GurobiError as e:
        print(f"Gurobi N={n}: LOI -> {e}")
