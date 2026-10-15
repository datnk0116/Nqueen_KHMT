import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import math
import pandas as pd
from benchmark import aggregate_repetition_records, run_benchmarks_with_repetitions

def make_record(status="SAT", time_seconds=0.1, is_sat=True, verified=True,
                import_time=0.01, build_time=0.01, solve_time=0.08,
                model_time=0.09, solver_reported_time=0.05,
                variables_count=64, clauses_constraints_count=100,
                error="", wall_time=0.15, n=8, paradigm="SAT",
                model_variant="sequential", solver="cadical"):
    return {
        "n": n,
        "paradigm": paradigm,
        "model_variant": model_variant,
        "solver": solver,
        "status": status,
        "is_sat": is_sat,
        "variables_count": variables_count,
        "clauses_constraints_count": clauses_constraints_count,
        "import_time": import_time,
        "build_time": build_time,
        "solve_time": solve_time,
        "model_time": model_time,
        "time_seconds": time_seconds,
        "solver_reported_time": solver_reported_time,
        "wall_time": wall_time,
        "verified": verified,
        "error": error
    }

def test_all_repetitions_succeed_same_status():
    recs = [
        make_record(status="SAT", time_seconds=0.10, solve_time=0.08),
        make_record(status="SAT", time_seconds=0.12, solve_time=0.10),
        make_record(status="SAT", time_seconds=0.11, solve_time=0.09),
    ]
    res = aggregate_repetition_records(recs, time_limit=30.0)
    assert res["status"] == "SAT"
    assert res["is_sat"] is True
    assert res["verified"] is True
    assert res["runs_total"] == 3
    assert res["runs_success"] == 3
    assert res["runs_timeout"] == 0
    assert res["runs_error"] == 0
    assert res["error"] == ""
    assert res["time_seconds"] == pytest.approx(0.11)
    assert res["solve_time"] == pytest.approx(0.09)

def test_successful_repetitions_different_statuses():
    # 2 FEASIBLE, 1 OPTIMAL
    recs = [
        make_record(status="FEASIBLE", paradigm="CPLEX-CP", solver="cplex-cp"),
        make_record(status="OPTIMAL", paradigm="CPLEX-CP", solver="cplex-cp"),
        make_record(status="FEASIBLE", paradigm="CPLEX-CP", solver="cplex-cp"),
    ]
    res = aggregate_repetition_records(recs, time_limit=30.0)
    assert res["status"] == "FEASIBLE"
    assert res["is_sat"] is True
    assert res["verified"] is True
    assert res["runs_total"] == 3
    assert res["runs_success"] == 3
    assert res["runs_timeout"] == 0
    assert res["runs_error"] == 0
    assert res["error"] == ""

    # Even tie: 1 OPTIMAL, 1 FEASIBLE in 2 repetitions
    tie_recs = [
        make_record(status="OPTIMAL"),
        make_record(status="FEASIBLE"),
    ]
    res_tie = aggregate_repetition_records(tie_recs, time_limit=30.0)
    assert res_tie["status"] in ("OPTIMAL", "FEASIBLE")
    assert res_tie["is_sat"] is True
    assert res_tie["runs_success"] == 2

def test_success_majority_minority_timeout():
    recs = [
        make_record(status="SAT", time_seconds=0.10, solve_time=0.05),
        make_record(status="SAT", time_seconds=0.12, solve_time=0.07),
        make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0,
                    solve_time=float('nan'), build_time=float('nan'), import_time=float('nan'),
                    model_time=float('nan'), error="Hard timeout expired"),
    ]
    res = aggregate_repetition_records(recs, time_limit=30.0)
    assert res["status"] == "SAT"
    assert res["is_sat"] is True
    assert res["verified"] is True
    assert res["runs_total"] == 3
    assert res["runs_success"] == 2
    assert res["runs_timeout"] == 1
    assert res["runs_error"] == 0
    assert "1/3 runs timed out" in res["error"]
    assert res["time_seconds"] == pytest.approx(0.12)
    assert res["solve_time"] == pytest.approx(0.06)

def test_timeout_majority_minority_success():
    recs = [
        make_record(status="SAT", time_seconds=0.10, solve_time=0.05),
        make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0,
                    solve_time=float('nan'), build_time=float('nan'), import_time=float('nan'),
                    model_time=float('nan'), error="Hard timeout expired"),
        make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0,
                    solve_time=float('nan'), build_time=float('nan'), import_time=float('nan'),
                    model_time=float('nan'), error="Hard timeout expired"),
    ]
    res = aggregate_repetition_records(recs, time_limit=30.0)
    assert res["status"] == "TIMEOUT"
    assert res["is_sat"] is False
    assert res["verified"] is False
    assert res["runs_total"] == 3
    assert res["runs_success"] == 1
    assert res["runs_timeout"] == 2
    assert res["runs_error"] == 0
    assert "2/3 runs timed out" in res["error"]
    assert "1/3 runs succeeded" in res["error"]
    assert math.isnan(res["solve_time"])
    assert res["time_seconds"] == pytest.approx(30.0)

def test_mixture_success_and_memout():
    # 2 SAT, 1 MEMOUT (success majority)
    recs_succ = [
        make_record(status="SAT", time_seconds=0.10, solve_time=0.05),
        make_record(status="SAT", time_seconds=0.12, solve_time=0.07),
        make_record(status="MEMOUT", is_sat=False, verified=False, time_seconds=30.0,
                    solve_time=float('nan'), error="Memory limit exceeded (MEMOUT)"),
    ]
    res_succ = aggregate_repetition_records(recs_succ, time_limit=30.0)
    assert res_succ["status"] == "SAT"
    assert res_succ["is_sat"] is True
    assert res_succ["verified"] is True
    assert res_succ["runs_total"] == 3
    assert res_succ["runs_success"] == 2
    assert res_succ["runs_timeout"] == 0
    assert res_succ["runs_error"] == 0
    assert "1/3 runs ran out of memory" in res_succ["error"]
    assert res_succ["solve_time"] == pytest.approx(0.06)

    # 1 SAT, 2 MEMOUT (MEMOUT majority)
    recs_mem = [
        make_record(status="SAT", time_seconds=0.10, solve_time=0.05),
        make_record(status="MEMOUT", is_sat=False, verified=False, time_seconds=30.0,
                    solve_time=float('nan'), error="Memory limit exceeded (MEMOUT)"),
        make_record(status="MEMOUT", is_sat=False, verified=False, time_seconds=30.0,
                    solve_time=float('nan'), error="Memory limit exceeded (MEMOUT)"),
    ]
    res_mem = aggregate_repetition_records(recs_mem, time_limit=30.0)
    assert res_mem["status"] == "MEMOUT"
    assert res_mem["is_sat"] is False
    assert res_mem["runs_total"] == 3
    assert res_mem["runs_success"] == 1
    assert "2/3 runs ran out of memory" in res_mem["error"]
    assert "1/3 runs succeeded" in res_mem["error"]
    assert math.isnan(res_mem["solve_time"])

def test_mixture_timeout_and_memout():
    # 2 TIMEOUT, 1 MEMOUT (timeout majority)
    recs_to = [
        make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan')),
        make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan')),
        make_record(status="MEMOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan'),
                    error="Memory limit exceeded (MEMOUT)"),
    ]
    res_to = aggregate_repetition_records(recs_to, time_limit=30.0)
    assert res_to["status"] == "TIMEOUT"
    assert res_to["runs_total"] == 3
    assert res_to["runs_timeout"] == 2
    assert "2/3 runs timed out" in res_to["error"]
    assert "1/3 runs ran out of memory" in res_to["error"]

    # 1 TIMEOUT, 2 MEMOUT (MEMOUT majority)
    recs_mem = [
        make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan')),
        make_record(status="MEMOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan'),
                    error="Memory limit exceeded (MEMOUT)"),
        make_record(status="MEMOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan'),
                    error="Memory limit exceeded (MEMOUT)"),
    ]
    res_mem = aggregate_repetition_records(recs_mem, time_limit=30.0)
    assert res_mem["status"] == "MEMOUT"
    assert res_mem["runs_total"] == 3
    assert res_mem["runs_timeout"] == 1
    assert "2/3 runs ran out of memory" in res_mem["error"]
    assert "1/3 runs timed out" in res_mem["error"]

    # 1 TIMEOUT, 1 MEMOUT (tied, R=2)
    recs_tie = [
        make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan')),
        make_record(status="MEMOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan'),
                    error="Memory limit exceeded (MEMOUT)"),
    ]
    res_tie = aggregate_repetition_records(recs_tie, time_limit=30.0)
    assert res_tie["status"] == "MEMOUT"
    assert "1/2 runs ran out of memory" in res_tie["error"]
    assert "1/2 runs timed out" in res_tie["error"]

def test_mixture_success_and_error():
    # 2 SAT, 1 ERROR (success majority)
    recs_succ = [
        make_record(status="SAT", time_seconds=0.10, solve_time=0.05),
        make_record(status="SAT", time_seconds=0.12, solve_time=0.07),
        make_record(status="ERROR", is_sat=False, verified=False, time_seconds=0.05,
                    solve_time=float('nan'), error="Unexpected solver crash"),
    ]
    res_succ = aggregate_repetition_records(recs_succ, time_limit=30.0)
    assert res_succ["status"] == "SAT"
    assert res_succ["is_sat"] is True
    assert res_succ["runs_total"] == 3
    assert res_succ["runs_success"] == 2
    assert res_succ["runs_error"] == 1
    assert "1/3 runs errored" in res_succ["error"]

    # 1 SAT, 2 ERROR (error majority)
    recs_err = [
        make_record(status="SAT", time_seconds=0.10, solve_time=0.05),
        make_record(status="ERROR", is_sat=False, verified=False, time_seconds=0.05,
                    solve_time=float('nan'), error="Crash A"),
        make_record(status="ERROR", is_sat=False, verified=False, time_seconds=0.05,
                    solve_time=float('nan'), error="Crash B"),
    ]
    res_err = aggregate_repetition_records(recs_err, time_limit=30.0)
    assert res_err["status"] == "ERROR"
    assert res_err["is_sat"] is False
    assert res_err["runs_total"] == 3
    assert res_err["runs_success"] == 1
    assert res_err["runs_error"] == 2
    assert "2/3 runs errored: Crash A" in res_err["error"]
    assert "1/3 runs succeeded" in res_err["error"]

def test_license_limit_behavior_and_short_circuit(monkeypatch):
    call_count = 0
    def mock_run_subprocess(paradigm, n, variant, solver, time_limit):
        nonlocal call_count
        call_count += 1
        return make_record(status="LICENSE_LIMIT", is_sat=False, verified=False,
                           variables_count=0, clauses_constraints_count=0,
                           solve_time=float('nan'), error="Trial license limited: 2000 vars exceeded")

    import benchmark
    monkeypatch.setattr(benchmark, "run_subprocess_solver", mock_run_subprocess)
    results = benchmark.run_benchmarks_with_repetitions("Gurobi", [128], ["binary"], ["gurobi"], 30.0, repetitions=3)
    assert len(results) == 1
    rec = results[0]
    # Verify short-circuiting: solver subprocess was invoked exactly once!
    assert call_count == 1
    assert rec["status"] == "LICENSE_LIMIT"
    assert rec["runs_total"] == 3
    assert rec["runs_success"] == 0
    assert rec["runs_timeout"] == 0
    assert rec["runs_error"] == 0
    assert "Trial license limited" in rec["error"]

    # Synthetic mixed record tests: 2 SAT, 1 LICENSE_LIMIT
    recs_mix = [
        make_record(status="SAT", time_seconds=0.1),
        make_record(status="SAT", time_seconds=0.12),
        make_record(status="LICENSE_LIMIT", is_sat=False, verified=False, error="Token expired"),
    ]
    res_mix = aggregate_repetition_records(recs_mix, time_limit=30.0)
    assert res_mix["status"] == "SAT"
    assert res_mix["runs_total"] == 3
    assert res_mix["runs_success"] == 2
    assert "1/3 runs hit license limits" in res_mix["error"]

    # 2 LICENSE_LIMIT, 1 SAT
    recs_lic_maj = [
        make_record(status="LICENSE_LIMIT", is_sat=False, verified=False, error="Token expired"),
        make_record(status="LICENSE_LIMIT", is_sat=False, verified=False, error="Token expired"),
        make_record(status="SAT", time_seconds=0.1),
    ]
    res_lic_maj = aggregate_repetition_records(recs_lic_maj, time_limit=30.0)
    assert res_lic_maj["status"] == "LICENSE_LIMIT"
    assert res_lic_maj["runs_total"] == 3
    assert res_lic_maj["runs_success"] == 1
    assert "2/3 runs hit license limits: Token expired" in res_lic_maj["error"]
    assert "1/3 runs succeeded" in res_lic_maj["error"]

def test_all_repetitions_fail():
    # 3 TIMEOUT
    recs_to = [make_record(status="TIMEOUT", is_sat=False, verified=False, solve_time=float('nan')) for _ in range(3)]
    res_to = aggregate_repetition_records(recs_to, time_limit=30.0)
    assert res_to["status"] == "TIMEOUT"
    assert res_to["runs_total"] == 3
    assert res_to["runs_timeout"] == 3
    assert res_to["runs_success"] == 0

    # 3 MEMOUT
    recs_mem = [make_record(status="MEMOUT", is_sat=False, verified=False, solve_time=float('nan'),
                            error="Memory limit exceeded (MEMOUT)") for _ in range(3)]
    res_mem = aggregate_repetition_records(recs_mem, time_limit=30.0)
    assert res_mem["status"] == "MEMOUT"
    assert res_mem["runs_total"] == 3
    assert "Memory limit exceeded (MEMOUT)" in res_mem["error"]

    # 3 ERROR
    recs_err = [make_record(status="ERROR", is_sat=False, verified=False, solve_time=float('nan'),
                            error="Syntax error") for _ in range(3)]
    res_err = aggregate_repetition_records(recs_err, time_limit=30.0)
    assert res_err["status"] == "ERROR"
    assert res_err["runs_total"] == 3
    assert res_err["runs_error"] == 3
    assert "3/3 runs errored: Syntax error" in res_err["error"]

    # 3 LICENSE_LIMIT
    recs_lic = [make_record(status="LICENSE_LIMIT", is_sat=False, verified=False, solve_time=float('nan'),
                            error="License size limit") for _ in range(3)]
    res_lic = aggregate_repetition_records(recs_lic, time_limit=30.0)
    assert res_lic["status"] == "LICENSE_LIMIT"
    assert res_lic["runs_total"] == 3
    assert "License size limit" in res_lic["error"]

def test_counters_and_diagnostic_notes_match_records():
    # R=4: 1 SAT, 1 TIMEOUT, 1 MEMOUT, 1 ERROR
    recs = [
        make_record(status="SAT", time_seconds=0.1, solve_time=0.08),
        make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan')),
        make_record(status="MEMOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan'),
                    error="OOM"),
        make_record(status="ERROR", is_sat=False, verified=False, time_seconds=0.05, solve_time=float('nan'),
                    error="Fatal exception"),
    ]
    res = aggregate_repetition_records(recs, time_limit=30.0)
    assert res["runs_total"] == 4
    assert res["runs_success"] == 1
    assert res["runs_timeout"] == 1
    assert res["runs_error"] == 1
    assert res["status"] == "MEMOUT"  # Severity tie-breaker: MEMOUT > TIMEOUT > ERROR
    assert "1/4 runs ran out of memory" in res["error"]
    assert "1/4 runs timed out" in res["error"]
    assert "1/4 runs errored" in res["error"]
    assert "1/4 runs succeeded" in res["error"]

def test_edge_cases_single_and_even_repetitions():
    # R = 1: Single repetition
    rec_sat = [make_record(status="SAT", time_seconds=0.05)]
    res_sat = aggregate_repetition_records(rec_sat, time_limit=30.0)
    assert res_sat["status"] == "SAT"
    assert res_sat["runs_total"] == 1
    assert res_sat["runs_success"] == 1
    assert res_sat["error"] == ""

    rec_to = [make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan'))]
    res_to = aggregate_repetition_records(rec_to, time_limit=30.0)
    assert res_to["status"] == "TIMEOUT"
    assert res_to["runs_total"] == 1
    assert res_to["runs_timeout"] == 1
    assert res_to["error"] == "1/1 runs timed out"

    rec_mem = [make_record(status="MEMOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan'), error="OOM kill")]
    res_mem = aggregate_repetition_records(rec_mem, time_limit=30.0)
    assert res_mem["status"] == "MEMOUT"
    assert res_mem["runs_total"] == 1
    assert "OOM kill" in res_mem["error"]

    rec_err = [make_record(status="ERROR", is_sat=False, verified=False, time_seconds=0.05, solve_time=float('nan'), error="Parse fail")]
    res_err = aggregate_repetition_records(rec_err, time_limit=30.0)
    assert res_err["status"] == "ERROR"
    assert res_err["runs_total"] == 1
    assert res_err["runs_error"] == 1
    assert "1/1 runs errored: Parse fail" in res_err["error"]

    rec_lic = [make_record(status="LICENSE_LIMIT", is_sat=False, verified=False, time_seconds=0.05, solve_time=float('nan'), error="Community edition")]
    res_lic = aggregate_repetition_records(rec_lic, time_limit=30.0)
    assert res_lic["status"] == "LICENSE_LIMIT"
    assert res_lic["runs_total"] == 1
    assert "Community edition" in res_lic["error"]

    # R = 2: Even repetition count
    # 2 successes
    recs_2sat = [make_record(status="SAT"), make_record(status="SAT")]
    res_2sat = aggregate_repetition_records(recs_2sat, time_limit=30.0)
    assert res_2sat["status"] == "SAT"
    assert res_2sat["runs_success"] == 2
    assert res_2sat["error"] == ""

    # 1 success, 1 timeout -> Indeterminate, MUST NOT be classified as success!
    recs_1s_1to = [
        make_record(status="SAT"),
        make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan'))
    ]
    res_1s_1to = aggregate_repetition_records(recs_1s_1to, time_limit=30.0)
    assert res_1s_1to["status"] == "TIMEOUT"
    assert res_1s_1to["is_sat"] is False
    assert res_1s_1to["runs_total"] == 2
    assert res_1s_1to["runs_success"] == 1
    assert res_1s_1to["runs_timeout"] == 1
    assert "1/2 runs timed out" in res_1s_1to["error"]
    assert "1/2 runs succeeded" in res_1s_1to["error"]

    # 1 success, 1 MEMOUT -> Indeterminate, MUST NOT be classified as success!
    recs_1s_1mem = [
        make_record(status="SAT"),
        make_record(status="MEMOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan'), error="OOM")
    ]
    res_1s_1mem = aggregate_repetition_records(recs_1s_1mem, time_limit=30.0)
    assert res_1s_1mem["status"] == "MEMOUT"
    assert res_1s_1mem["is_sat"] is False
    assert "1/2 runs ran out of memory" in res_1s_1mem["error"]
    assert "1/2 runs succeeded" in res_1s_1mem["error"]

    # 1 success, 1 ERROR -> Indeterminate, MUST NOT be classified as success!
    recs_1s_1err = [
        make_record(status="SAT"),
        make_record(status="ERROR", is_sat=False, verified=False, time_seconds=0.05, solve_time=float('nan'), error="Crash")
    ]
    res_1s_1err = aggregate_repetition_records(recs_1s_1err, time_limit=30.0)
    assert res_1s_1err["status"] == "ERROR"
    assert res_1s_1err["is_sat"] is False
    assert "1/2 runs errored: Crash" in res_1s_1err["error"]
    assert "1/2 runs succeeded" in res_1s_1err["error"]

    # R = 4: Even repetition count
    # 3 successes, 1 timeout -> strict majority (> 2.0)
    recs_3s_1to = [
        make_record(status="SAT", time_seconds=0.10),
        make_record(status="SAT", time_seconds=0.12),
        make_record(status="SAT", time_seconds=0.11),
        make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan'))
    ]
    res_3s_1to = aggregate_repetition_records(recs_3s_1to, time_limit=30.0)
    assert res_3s_1to["status"] == "SAT"
    assert res_3s_1to["is_sat"] is True
    assert res_3s_1to["runs_total"] == 4
    assert res_3s_1to["runs_success"] == 3
    assert "1/4 runs timed out" in res_3s_1to["error"]

    # 2 successes, 2 timeouts -> 50/50 tie (not strict majority), indeterminate
    recs_2s_2to = [
        make_record(status="SAT", time_seconds=0.10),
        make_record(status="SAT", time_seconds=0.12),
        make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan')),
        make_record(status="TIMEOUT", is_sat=False, verified=False, time_seconds=30.0, solve_time=float('nan'))
    ]
    res_2s_2to = aggregate_repetition_records(recs_2s_2to, time_limit=30.0)
    assert res_2s_2to["status"] == "TIMEOUT"
    assert res_2s_2to["is_sat"] is False
    assert res_2s_2to["runs_total"] == 4
    assert res_2s_2to["runs_success"] == 2
    assert res_2s_2to["runs_timeout"] == 2
    assert "2/4 runs timed out" in res_2s_2to["error"]
    assert "2/4 runs succeeded" in res_2s_2to["error"]
