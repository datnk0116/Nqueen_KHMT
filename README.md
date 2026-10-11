# N-Queens SAT, CP-SAT, and MIP Benchmark Project

This project provides a comprehensive benchmarking framework for solving the classic $N$-Queens problem across three major constraint-solving paradigms:
1. **Propositional Satisfiability (SAT)**: Pairwise, Binary, Sequential Counter, Commander, and Product encodings solved with `glucose3`, `minisat22`, and `cadical`.
2. **Constraint Programming (CP-SAT)**: Alldiff, Allowed assignments, Explicit assignments, Boolean, Integer, and Matrix formulations solved with OR-Tools CP-SAT (`ortools-cpsat`).
3. **Mixed-Integer Programming (MIP)**: Binary, Integer, and Matrix formulations solved with `cbc` and `scip`.

---

## Project Structure

- [`src/`](nqueens-sat-benchmark/src/): Source code containing model definitions, board representations, encodings, and benchmarking logic.
  - [`board.py`](nqueens-sat-benchmark/src/board.py): Board representation and solution validation.
  - [`encodings.py`](nqueens-sat-benchmark/src/encodings.py): SAT encoding algorithms (pairwise, binary, sequential, commander, product).
  - [`sat_model.py`](nqueens-sat-benchmark/src/sat_model.py): SAT model builder and solver integration.
  - [`cpsat_model.py`](nqueens-sat-benchmark/src/cpsat_model.py): CP-SAT model builder and OR-Tools solver integration.
  - [`mip_cp_models.py`](nqueens-sat-benchmark/src/mip_cp_models.py): MIP model builder and solver integration (CBC, SCIP).
  - [`benchmark.py`](nqueens-sat-benchmark/src/benchmark.py): Main benchmarking script executing runs across $N$ and saving CSV results.
  - [`generate_report.py`](nqueens-sat-benchmark/src/generate_report.py): Script to process CSV results, generate summary charts/tables, and build the LaTeX report.
- [`tests/`](nqueens-sat-benchmark/tests/): Unit test suite covering all models, encodings, and board validation.
- [`results/`](results/): Benchmark output CSV files (`benchmark_results.csv`, `sat_results.csv`, `cpsat_results.csv`, `mip_results.csv`).
- [`report/`](nqueens-sat-benchmark/report/): Generated LaTeX report (`report.tex`), tables, and performance plots.

---

## Getting Started & Usage

### 1. Environment Setup
Activate the virtual environment located in the root directory:
```bash
source SAT/bin/activate
```

### 2. Running Unit Tests
Execute the pytest test suite to verify correctness across all implementations:
```bash
./SAT/bin/pytest nqueens-sat-benchmark/tests
```

### 3. Running Benchmarks
To run the benchmark suite and generate/update CSV results in [`results/`](results/):
```bash
./SAT/bin/python nqueens-sat-benchmark/src/benchmark.py
```

### 4. Generating the LaTeX Report & Visualizations
To aggregate the benchmark results, generate performance scaling charts, and compile the LaTeX report in [`report/`](nqueens-sat-benchmark/report/):
```bash
./SAT/bin/python nqueens-sat-benchmark/src/generate_report.py
```
The resulting LaTeX document is saved at [`report/report.tex`](nqueens-sat-benchmark/report/report.tex) along with generated charts (`sat_time_cadical.png`, `paradigm_comparison.png`, `variables_scaling.png`).
