# N-Queens Multi-Paradigm Benchmark Framework

This project provides an exhaustive empirical benchmarking and analysis framework for solving the classic $N$-Queens problem across six major solver groups and modeling paradigms:

1. **Propositional Satisfiability (SAT)**: Pairwise, Binary, Sequential Counter, Commander, and Product encodings solved with `cadical`, `glucose3`, `minisat22`, `kissat`, and `maplecm`.
2. **Constraint Programming (CP-SAT)**: OR-Tools CP-SAT (`ortools-cpsat`) using Integer, Boolean, and Allowed assignments formulations with global constraints (`AllDifferent`).
3. **Open-Source Mixed-Integer Programming (MIP)**: Binary and Integer linear formulations solved with `cbc` and `scip`.
4. **Commercial MIP (Gurobi)**: High-performance mathematical programming via `gurobipy`.
5. **Commercial CPLEX (MIP & CP)**: IBM ILOG CPLEX Optimizer using both Mathematical Programming (`cplex` MIP) and Constraint Programming (`cplex` CP) engines.

---

## Repository Structure

- [`src/`](nqueens-sat-benchmark/src/): Core source code and model implementations.
  - [`board.py`](nqueens-sat-benchmark/src/board.py): Board representation and solution verification.
  - [`encodings.py`](nqueens-sat-benchmark/src/encodings.py): SAT encoding algorithms (pairwise, binary, sequential, commander, product).
  - [`sat_model.py`](nqueens-sat-benchmark/src/sat_model.py): SAT model builder and multi-solver integration.
  - [`cpsat_model.py`](nqueens-sat-benchmark/src/cpsat_model.py): OR-Tools CP-SAT model builder and solver.
  - [`mip_cp_models.py`](nqueens-sat-benchmark/src/mip_cp_models.py): Open-source MIP model builder (CBC, SCIP).
  - [`gurobi_model.py`](nqueens-sat-benchmark/src/gurobi_model.py): Gurobi optimization model builder and solver.
  - [`cplex_mip_model.py`](nqueens-sat-benchmark/src/cplex_mip_model.py): CPLEX MIP model builder and solver.
  - [`cplex_cp_model.py`](nqueens-sat-benchmark/src/cplex_cp_model.py): CPLEX CP model builder and solver.
  - [`benchmark.py`](nqueens-sat-benchmark/src/benchmark.py): Main benchmarking orchestrator with subprocess timeouts and dataset scaling up to $N = 128$.
  - [`generate_report.py`](nqueens-sat-benchmark/src/generate_report.py): Result aggregation, charting (matplotlib), and LaTeX report compilation.
- [`scripts/`](nqueens-sat-benchmark/scripts/): Diagnostic and license verification scripts.
  - [`check_gurobi.py`](nqueens-sat-benchmark/scripts/check_gurobi.py): Gurobi license & installation check.
  - [`check_cplex.py`](nqueens-sat-benchmark/scripts/check_cplex.py): CPLEX license & installation check.
- [`tests/`](nqueens-sat-benchmark/tests/): Comprehensive unit test suite covering encodings, board validation, and solver models.
- [`results/`](results/): Raw CSV results (`benchmark_results.csv`, `sat_results.csv`, `cpsat_results.csv`, `mip_results.csv`, `gurobi_results.csv`, `cplex_results.csv`).
- [`report/`](nqueens-sat-benchmark/report/): Academic LaTeX report (`report.tex`, `report.pdf`), styling classes (`cas-sc.cls`), and generated performance plots.

---

## Getting Started & Usage

### 1. Environment Setup
Activate the virtual environment:
```bash
source SAT/bin/activate
```

### 2. Commercial Solvers Setup (Gurobi & CPLEX)
To enable Gurobi and CPLEX benchmarking, ensure valid licenses are configured:
- **Gurobi**: Run license check script:
  ```bash
  python nqueens-sat-benchmark/scripts/check_gurobi.py
  ```
- **CPLEX**: Run license check script:
  ```bash
  python nqueens-sat-benchmark/scripts/check_cplex.py
  ```

### 3. Running Unit Tests
Execute pytest across the test suite:
```bash
pytest nqueens-sat-benchmark/tests
```

### 4. Running Benchmarks
Run the full benchmarking suite across all solvers and board sizes up to $N = 128$:
```bash
python nqueens-sat-benchmark/src/benchmark.py
```

### 5. Generating the Academic Report & Visualizations
Aggregate results, generate performance scaling charts, and compile the LaTeX report into `report.pdf`:
```bash
python nqueens-sat-benchmark/src/generate_report.py
```
The compiled PDF report is available at [`report/report.pdf`](nqueens-sat-benchmark/report/report.pdf).
