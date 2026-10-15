# N-Queens Multi-Paradigm Benchmark Framework

This project provides an empirical benchmarking and analysis framework for solving the classic $N$-Queens problem across six major solver groups and modeling paradigms:

1. **Propositional Satisfiability (SAT)**: Pairwise, Binary, Sequential Counter, Commander, and Product encodings solved with `cadical`, `glucose3`, `minisat22`, `kissat`, and `maplecm`.
2. **Constraint Programming (CP-SAT)**: OR-Tools CP-SAT (`ortools-cpsat`) using Integer, Boolean, and Allowed assignments formulations with global constraints (`AllDifferent`).
3. **Open-Source Mixed-Integer Programming (MIP)**: Binary and Integer linear formulations solved with `cbc` and `scip`.
4. **Commercial MIP (Gurobi)**: High-performance mathematical programming via `gurobipy`.
5. **Commercial CPLEX MIP**: IBM ILOG CPLEX Optimizer using the Mathematical Programming (`cplex` MIP) engine.
6. **Commercial CPLEX CP**: IBM ILOG CPLEX Optimizer using the Constraint Programming (`cplex-cp`) engine.

---

## Repository Structure

- [`src/`](src/): Core source code and model implementations.
  - [`board.py`](src/board.py:1): Board representation and solution verification.
  - [`encodings.py`](src/encodings.py:1): SAT encoding algorithms (pairwise, binary, sequential, commander, product).
  - [`sat_model.py`](src/sat_model.py:1): SAT model builder and multi-solver integration.
  - [`cpsat_model.py`](src/cpsat_model.py:1): OR-Tools CP-SAT model builder and solver.
  - [`mip_cp_models.py`](src/mip_cp_models.py:1): Open-source MIP model builder (CBC, SCIP).
  - [`gurobi_model.py`](src/gurobi_model.py:1): Gurobi optimization model builder and solver.
  - [`cplex_mip_model.py`](src/cplex_mip_model.py:1): CPLEX MIP model builder and solver.
  - [`cplex_cp_model.py`](src/cplex_cp_model.py:1): CPLEX CP model builder and solver.
  - [`benchmark.py`](src/benchmark.py:1): Main benchmarking orchestrator with subprocess timeouts and dataset scaling up to $N = 128$.
  - [`generate_report.py`](src/generate_report.py:1): Result aggregation, charting (matplotlib), and LaTeX report compilation.
- [`scripts/`](scripts/): Diagnostic and license verification scripts.
  - [`check_gurobi.py`](scripts/check_gurobi.py:1): Gurobi license & installation check.
  - [`check_cplex.py`](scripts/check_cplex.py:1): CPLEX license & installation check.
- [`tests/`](tests/): Comprehensive unit test suite covering encodings, board validation, and solver models.
- [`results/`](results/): Raw CSV results (`benchmark_results.csv`, `sat_results.csv`, `cpsat_results.csv`, `mip_results.csv`, `gurobi_results.csv`, `cplex_results.csv`).
- [`report/`](report/): Academic LaTeX report (`report.tex`, `report.pdf`), styling classes (`cas-sc.cls`), and generated performance plots.

---

## Timing Protocol

The benchmarking framework enforces a unified, rigorous timing and timeout protocol across all paradigms:
- **High-Resolution Clocks**: All elapsed time measurements consistently utilize `time.perf_counter()`.
- **Phase Breakdown & Metric Definitions**:
  - `import_time`: Time taken to import and load paradigm/solver modules prior to model construction.
  - `build_time`: Time taken to construct model variables and constraints (excluding module import overhead).
  - `solve_time`: Active solving time measured elapsed around solver invocation.
  - `time_seconds`: Cumulative child process phase duration, defined strictly as `import_time + build_time + solve_time` (or `time_limit` on hard timeout).
  - `wall_time`: Parent-measured wall-clock elapsed time for the child process execution (`subprocess.run`), capturing OS process launch, execution, and IPC.
  - `solver_reported_time`: Internal solve time reported directly by solvers when available (e.g., CPLEX CP via `get_solve_time()`, CPLEX MIP via `solve_details.time`, Gurobi via `Runtime`).
  - `model_time`: Solver-relevant duration defined per run as `build_time + solve_time` (excluding module import overhead), used in cross-paradigm comparisons to eliminate Python module import skew at small $N$. When aggregating over repetitions, reported `model_time` is the median of per-run `(build_time + solve_time)` values; hence, mathematically it need not equal `median(build_time) + median(solve_time)`.
- **Dual Timeout Architecture**:
  - **Solver-Level Time Limit (`TIME_LIMIT`)**: Inside the child process, the remaining time budget `max(0.001, time_limit - (import_time + build_time))` is computed and passed to the solver. For backends with cooperative interrupt support (such as PySAT `Glucose3`, `Glucose4`, `Minisat22`, `MapleCM`, OR-Tools CP-SAT, and MIP solvers), the solver terminates gracefully when the limit is reached without killing the process.
  - **Process-Level Hard Timeout (`TIMEOUT`)**: The parent process wraps child execution with `subprocess.run(timeout=time_limit)`. Backends lacking internal interrupt support in PySAT (such as CaDiCaL and Kissat) rely on this parent-side hard timeout. When a hard timeout occurs, status is recorded as `TIMEOUT`, phase times are recorded as `NaN`, and `time_seconds` is set to `time_limit`.
  - **Status Classification Layer**: Employs a dedicated classification layer (`classify_status()`) that cleanly categorizes solver outcomes into `SUCCESS`, `TIMEOUT`, `MEMOUT`, `LICENSE_LIMIT`, `INFEASIBLE`, `ERROR`, and `UNKNOWN`:
    - Successful statuses (`OPTIMAL`, `FEASIBLE`, `SAT`, `SUBOPTIMAL`) are accepted as success only when their solution validity requirements are satisfied (`is_sat = True` and `verified = True`). Unverified or unvalidated outputs degrade to `ERROR`.
    - Normalizes case variations across solvers (e.g. CPLEX CP `Feasible` is normalized to `FEASIBLE`) while preserving solver-specific status semantics (e.g. Gurobi `SUBOPTIMAL`).
    - Explicitly identifies proven unsatisfiability (`UNSAT`, `INFEASIBLE`) with verified status tracking, while solver model rejections (`MODEL_INVALID`) are classified under `ERROR` rather than treated as proven mathematical infeasibility.
    - Soft timeouts (`TIME_LIMIT`) are treated as timeouts rather than normal success (`is_sat = False`).
- **Repetitions & Mixed-Run Resolution**:
  - Each configuration executes 3 repetitions (`--repetitions 3`). For deterministic conditions (`LICENSE_LIMIT`), redundant runs are short-circuited by padding identical records. For non-deterministic conditions (`MEMOUT`, timeouts, errors), all repetitions are executed.
  - Granular repetition metrics are tracked per benchmark:
    - `runs_total`: Total number of repetition runs executed (default: 3).
    - `runs_success`: Count of successful runs (`SAT`, `OPTIMAL`, `FEASIBLE`, `SUBOPTIMAL`).
    - `runs_timeout`: Count of timed out runs (`TIMEOUT`, `TIME_LIMIT`).
    - `runs_error`: Count of genuine execution errors (`ERROR`). Non-error failures such as `LICENSE_LIMIT`, `MEMOUT`, `INFEASIBLE`, or `UNKNOWN` resolve to their respective aggregate status without inflating `runs_error`.
  - Aggregation Policy:
    - **Success Majority**: A configuration is classified as successful only if successful runs form a strict majority (`runs_success > runs_total / 2`) and their solutions are verified. The aggregate status is the most common status among successful repetitions (mode), `is_sat = True`, `verified = True`, and non-success occurrences (`TIMEOUT`, `MEMOUT`, `ERROR`, `LICENSE_LIMIT`, `INFEASIBLE`, `UNKNOWN`) are recorded in the `error` column (e.g., `"1/3 runs timed out"`). Phase times (`import_time`, `build_time`, `solve_time`, `model_time`) are medians across successful runs.
    - **Failure Majority & Indeterminate Outcomes**: If successes do not form a strict majority, the outcome is never classified as successful (`is_sat = False`).
      - If a failure category (`TIMEOUT`/`TIME_LIMIT`, `MEMOUT`, `ERROR`, `LICENSE_LIMIT`, `INFEASIBLE`, `UNKNOWN`) forms a strict majority (`count > runs_total / 2`), the aggregate status resolves to that failure mode.
      - If no category achieves a strict majority (e.g. ties in even repetitions or split failures across modes), the status resolves to the most frequent failure mode among categories with `count > 0`, breaking ties deterministically by severity hierarchy (`LICENSE_LIMIT` > `MEMOUT` > `TIMEOUT` > `INFEASIBLE` > `ERROR` > `UNKNOWN`). Crucially, `LICENSE_LIMIT` is selected only when evidence of license limits exists (`runs_license > 0`), ensuring unknown or unclassified statuses never fall through to license limits.
      - Diagnostic notes describe the primary failure and detail all observed outcomes (e.g., `"1/2 runs timed out; 1/2 runs succeeded"`). Phase times are recorded as `NaN` to prevent misleading solve-time metrics on failed or indeterminate configurations.
  - Medians across repetitions are reported for all timing metrics, including `solver_reported_time` and `wall_time`.

> **Note on Benchmark Results Data**:
> The 170-row dataset currently committed in `results/` reflects complete benchmark runs executed under the standardized P4 protocol with subprocess isolation, 3 repetitions, and granular tracking metrics (`runs_total`, `runs_success`, `runs_timeout`, `runs_error`, `model_time`, and `wall_time`). Optional future benchmark reruns can be executed using the command in Step 4 below.

---

## Getting Started & Usage

### 1. Environment Setup & Requirements
Create and activate the virtual environment, then install required dependencies:
```bash
python -m venv SAT
source SAT/bin/activate
pip install -r requirements.txt
```

### 2. Commercial Solvers Setup (Gurobi & CPLEX)
To enable Gurobi and CPLEX benchmarking, ensure valid licenses are configured:
- **Gurobi**: Run license check script:
  ```bash
  python scripts/check_gurobi.py
  ```
- **CPLEX**: Run license check script:
  ```bash
  python scripts/check_cplex.py
  ```

### 3. Running Unit Tests
Execute pytest across the test suite:
```bash
pytest tests
```

### 4. Running Benchmarks
Run the full benchmarking suite across all solvers and board sizes up to $N = 128$ with customized time limit and repetitions:
```bash
python src/benchmark.py --time-limit 30.0 --repetitions 3 --output-dir results
```
*(Note: Commercial solvers Gurobi and CPLEX require valid licenses or trial constraints; ensure license availability before execution).*

### 5. LaTeX Requirements (for Report Compilation)
Compiling the academic report (`report/report.tex`) requires a **full TeX Live installation** (or Overleaf) with the following LaTeX packages:

`footmisc`, `makecell`, `moreverb`, `multirow`, `stfloats`, `wrapfig`, `xstring`, `booktabs`, `natbib`, `hyperref`, `amsmath`, `graphicx`, `geometry`, `float`, `xcolor`, `colortbl`, `dcolumn`, `array`

If any package is missing, install it via:
```bash
tlmgr install footmisc makecell moreverb multirow stfloats wrapfig xstring
```

> **Note**: Do not vendor (copy) `.sty` files into `report/`. The `cas-sc.cls` document class loads these packages from the TeX system distribution. Vendored copies from a different TeX Live version can conflict and cause compilation failures.

### 6. Generating the Academic Report & Visualizations
Aggregate results, generate performance scaling charts, and compile the LaTeX report into `report.pdf`:
```bash
python src/generate_report.py
```
The compiled PDF report is available at [`report/report.pdf`](report/report.pdf).
