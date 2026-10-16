import sys
import os
import re
import pytest
import pandas as pd
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from generate_report import load_combined, timeout_table, make_sat_table_rows, make_sat_size_table_rows, make_table_rows, format_sat_time, build_discussion_stats, generate_tex, compile_pdf, main

def test_timeout_table():
    df = load_combined("results")
    to_df = timeout_table(df)
    assert len(to_df) == 5

    row1 = to_df[(to_df['paradigm'] == 'CP-SAT') & (to_df['solver'] == 'ortools-cpsat') & (to_df['n'] == 128)]
    assert len(row1) == 1
    assert row1.iloc[0]['timeout_count'] == 1

    row2 = to_df[(to_df['paradigm'] == 'SAT') & (to_df['solver'] == 'glucose3') & (to_df['n'] == 64)]
    assert len(row2) == 1
    assert row2.iloc[0]['timeout_count'] == 3

    row3 = to_df[(to_df['paradigm'] == 'SAT') & (to_df['solver'] == 'glucose3') & (to_df['n'] == 128)]
    assert len(row3) == 1
    assert row3.iloc[0]['timeout_count'] == 4

    row4 = to_df[(to_df['paradigm'] == 'SAT') & (to_df['solver'] == 'maplecm') & (to_df['n'] == 128)]
    assert len(row4) == 1
    assert row4.iloc[0]['timeout_count'] == 3

    row5 = to_df[(to_df['paradigm'] == 'SAT') & (to_df['solver'] == 'minisat22') & (to_df['n'] == 128)]
    assert len(row5) == 1
    assert row5.iloc[0]['timeout_count'] == 3

def test_load_combined_raises_on_duplicates(tmp_path):
    d = {
        'n': [8, 8],
        'paradigm': ['SAT', 'SAT'],
        'model_variant': ['pairwise', 'pairwise'],
        'solver': ['cadical', 'cadical'],
        'status': ['SUCCESS', 'SUCCESS']
    }
    test_df = pd.DataFrame(d)
    csv_path = tmp_path / "benchmark_results.csv"
    test_df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Duplicate rows detected"):
        load_combined(str(tmp_path))

def test_sat_table_rows_cells():
    df = load_combined("results")
    sat_tex = make_sat_table_rows(df)
    lines = sat_tex.strip().split("\n")
    assert len(lines) == 25

    sat_df = df[df['paradigm'] == 'SAT']
    solvers = ['cadical', 'glucose3', 'minisat22', 'kissat', 'maplecm']

    sorted_df = sat_df.sort_values(by=['n', 'model_variant']).drop_duplicates(subset=['n', 'model_variant'])
    for idx, (_, r) in enumerate(sorted_df.iterrows()):
        n = r['n']
        variant = r['model_variant']
        line_parts = [p.strip() for p in lines[idx].split("&")]
        assert int(line_parts[0]) == n
        assert line_parts[1] == variant
        for s_idx, solver in enumerate(solvers):
            sub = sat_df[(sat_df['n'] == n) & (sat_df['model_variant'] == variant) & (sat_df['solver'] == solver)]
            if sub.empty:
                expected = "-"
            else:
                expected = format_sat_time(sub.iloc[0])
            actual = line_parts[2 + s_idx].rstrip('\\').strip()
            assert actual == expected

def test_sat_size_table_rows():
    df = load_combined("results")
    size_tex = make_sat_size_table_rows(df)
    lines = size_tex.strip().split("\n")
    assert len(lines) == 25

    text_content = size_tex
    assert "64 & commander & 11958 & 47096" in text_content
    assert "64 & sequential & 20098 & 47692" in text_content

def test_make_table_rows_timeout():
    df_timeout = pd.DataFrame([{
        'n': 128,
        'model_variant': 'allowed',
        'solver': 'ortools-cpsat',
        'variables_count': 0,
        'constraints_count': 0,
        'time_seconds': 30.0,
        'status': 'TIMEOUT'
    }])
    res = make_table_rows(df_timeout)
    assert "- & - & T/O" in res
    assert "0 & 0" not in res

def test_build_discussion_stats():
    df = load_combined("results")
    stats = build_discussion_stats(df)

    assert abs(stats['cplex_cp_8_ms'] - 161.11) < 1.0
    assert abs(stats['cplex_cp_128_ms'] - 169.60) < 1.0
    assert abs(stats['cplex_cp_128_solve_s'] - 0.0194) < 0.005
    assert abs(stats['cplex_cp_128_build_ms'] - 0.533) < 0.05
    assert abs(stats['cpsat_int_128_s'] - 0.647) < 0.05
    assert abs(stats['cpsat_bool_128_s'] - 0.720) < 0.05
    assert abs(stats['cadical_max'] - 8.64) < 0.1
    assert abs(stats['kissat_max'] - 5.93) < 0.1
    assert abs(stats['maplecm_max'] - 26.00) < 0.1
    assert stats['maplecm_max_variant'] == "product"
    assert stats['sat_128_timeouts'] == 10
    assert stats['sat_128_total'] == 25
    assert stats['g3_to'] == 4
    assert stats['m22_to'] == 3
    assert stats['mcm_to'] == 3

def test_build_discussion_stats_empty_and_missing_data():
    empty_df = pd.DataFrame()
    stats_empty = build_discussion_stats(empty_df)
    assert pd.isna(stats_empty['cadical_max'])
    assert pd.isna(stats_empty['kissat_max'])
    assert pd.isna(stats_empty['maplecm_max'])
    assert stats_empty['maplecm_max_variant'] == "N/A"
    assert stats_empty['sat_128_timeouts'] == 0
    assert stats_empty['sat_128_total'] == 0
    assert stats_empty['g3_to'] == 0
    assert stats_empty['m22_to'] == 0
    assert stats_empty['mcm_to'] == 0
    assert pd.isna(stats_empty['cplex_cp_8_s'])
    assert pd.isna(stats_empty['cplex_cp_128_s'])
    assert stats_empty['cplex_cp_solve_times_str'] == "N/A"
    assert pd.isna(stats_empty['cpsat_int_128_s'])
    assert pd.isna(stats_empty['cpsat_bool_128_s'])
    assert not stats_empty['has_solver_reported_time']
    assert pd.isna(stats_empty['cplex_cp_128_reported_s'])

    # Ensure no hardcoded fallback numbers appear when data is empty
    for val in [8.60, 6.03, 26.22, 0.000264, 0.010534, 0.71326, 0.73134]:
        for k, v in stats_empty.items():
            if isinstance(v, (int, float)) and not pd.isna(v):
                assert abs(v - val) > 1e-6

def test_domain_regex_matching():
    old_str = r"$q_i \in \{1, \dots, N\}$"
    assert re.search(r"\\\{1,\s*\\dots,\s*N\\\}", old_str) is not None

def test_tmp_path_report_generation(tmp_path, monkeypatch):
    results_src = "results"
    results_dst = tmp_path / "results"
    shutil.copytree(results_src, results_dst)

    monkeypatch.chdir(tmp_path)
    generate_tex("results", "report")

    report_tex_path = "report/report.tex"
    assert os.path.exists(report_tex_path)
    with open(report_tex_path, "r") as f:
        content = f.read()

    assert "10/25" in content
    assert "Glucose3: 4" in content
    assert "MiniSat22: 3" in content
    assert "MapleCM: 3" in content

    assert "0.02 to 1.3" not in content
    assert "under 26s" not in content
    assert "0.012 to 0.019" not in content
    assert "has not been isolated" not in content
    assert "130.7~million at $N = 128$" in content
    assert "tab:sat_encodings_complexity" in content
    assert r"\Theta(N^3)" in content
    assert "428,800 clauses at $N = 64$" in content
    assert re.search(r"\\\{1,\s*\\dots,\s*N\\\}", content) is None

def test_compile_pdf_missing_pdflatex(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda *_: None)
    res = compile_pdf("report")
    assert res is False

def test_load_combined_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required benchmark results file not found"):
        load_combined(str(tmp_path))

def test_generate_tex_empty_dataset_raises(tmp_path):
    empty_csv = tmp_path / "benchmark_results.csv"
    empty_csv.write_text("")
    with pytest.raises(ValueError, match="contains no valid records"):
        generate_tex(str(tmp_path), str(tmp_path / "report"))

def test_generate_tex_missing_sat_128_data(tmp_path):
    df = pd.DataFrame([{
        'n': 8, 'paradigm': 'SAT', 'model_variant': 'pairwise', 'solver': 'cadical',
        'status': 'SUCCESS', 'time_seconds': 0.05, 'build_time': 0.01, 'solve_time': 0.04,
        'variables_count': 64, 'clauses_constraints_count': 100
    }])
    df.to_csv(tmp_path / "benchmark_results.csv", index=False)
    tex_path = generate_tex(str(tmp_path), str(tmp_path / "report"))
    with open(tex_path, "r") as f:
        content = f.read()
    assert "0/0" not in content
    assert "insufficient data" in content
    assert "specifically Glucose3: 0" not in content

def test_generate_tex_sat_128_zero_timeouts(tmp_path):
    df = pd.DataFrame([{
        'n': 128, 'paradigm': 'SAT', 'model_variant': 'pairwise', 'solver': 'cadical',
        'status': 'SUCCESS', 'time_seconds': 0.5, 'build_time': 0.1, 'solve_time': 0.4,
        'variables_count': 16384, 'clauses_constraints_count': 50000
    }])
    df.to_csv(tmp_path / "benchmark_results.csv", index=False)
    tex_path = generate_tex(str(tmp_path), str(tmp_path / "report"))
    with open(tex_path, "r") as f:
        content = f.read()
    assert "0/0" not in content
    assert "no timeouts were encountered across tested SAT configurations at $N = 128$ (0/1 timed out)" in content

def test_generate_tex_missing_cplex_cp(tmp_path):
    df = pd.DataFrame([{
        'n': 8, 'paradigm': 'SAT', 'model_variant': 'pairwise', 'solver': 'cadical',
        'status': 'SUCCESS', 'time_seconds': 0.05, 'build_time': 0.01, 'solve_time': 0.04,
        'variables_count': 64, 'clauses_constraints_count': 100
    }])
    df.to_csv(tmp_path / "benchmark_results.csv", index=False)
    tex_path = generate_tex(str(tmp_path), str(tmp_path / "report"))
    with open(tex_path, "r") as f:
        content = f.read()
    assert "CP Optimizer experimental data is not available" in content
    assert "N/A to N/A" not in content

def test_generate_tex_missing_optional_columns(tmp_path):
    df = pd.DataFrame([{
        'n': 8, 'paradigm': 'SAT', 'model_variant': 'pairwise', 'solver': 'cadical',
        'status': 'SUCCESS', 'time_seconds': 0.05
    }])
    df.to_csv(tmp_path / "benchmark_results.csv", index=False)
    tex_path = generate_tex(str(tmp_path), str(tmp_path / "report"))
    assert os.path.exists(tex_path)

def test_generate_tex_no_timeouts(tmp_path):
    df = pd.DataFrame([{
        'n': 8, 'paradigm': 'SAT', 'model_variant': 'pairwise', 'solver': 'cadical',
        'status': 'SUCCESS', 'time_seconds': 0.05, 'build_time': 0.01, 'solve_time': 0.04,
        'variables_count': 64, 'clauses_constraints_count': 100
    }])
    df.to_csv(tmp_path / "benchmark_results.csv", index=False)
    tex_path = generate_tex(str(tmp_path), str(tmp_path / "report"))
    with open(tex_path, "r") as f:
        content = f.read()
    assert "No timeouts recorded" in content

def test_generate_tex_missing_cpsat(tmp_path):
    df = pd.DataFrame([{
        'n': 8, 'paradigm': 'SAT', 'model_variant': 'pairwise', 'solver': 'cadical',
        'status': 'SUCCESS', 'time_seconds': 0.05, 'build_time': 0.01, 'solve_time': 0.04,
        'variables_count': 64, 'clauses_constraints_count': 100
    }])
    df.to_csv(tmp_path / "benchmark_results.csv", index=False)
    tex_path = generate_tex(str(tmp_path), str(tmp_path / "report"))
    with open(tex_path, "r") as f:
        content = f.read()
    assert "CP-SAT Allowed Assignment Timeout" not in content

def test_generate_tex_preserves_error_rows(tmp_path):
    df = pd.DataFrame([
        {
            'n': 8, 'paradigm': 'SAT', 'model_variant': 'pairwise', 'solver': 'cadical',
            'status': 'ERROR', 'time_seconds': 0.0, 'build_time': float('nan'), 'solve_time': float('nan'),
            'variables_count': 0, 'clauses_constraints_count': 0
        },
        {
            'n': 8, 'paradigm': 'CP-SAT', 'model_variant': 'integer', 'solver': 'ortools-cpsat',
            'status': 'ERROR', 'time_seconds': 0.0, 'build_time': float('nan'), 'solve_time': float('nan'),
            'variables_count': 0, 'clauses_constraints_count': 0
        }
    ])
    df.to_csv(tmp_path / "benchmark_results.csv", index=False)
    loaded = load_combined(str(tmp_path))
    assert len(loaded) == 2
    assert (loaded['status'] == 'ERROR').all()
    tex_path = generate_tex(str(tmp_path), str(tmp_path / "report"))
    with open(tex_path, "r") as f:
        content = f.read()
    assert "8 & pairwise & ERR & - & - & - & - \\\\" in content
    assert "- & - & ERR" in content

def test_experimental_setup_content(tmp_path):
    df = pd.DataFrame([{
        'n': 8, 'paradigm': 'SAT', 'model_variant': 'pairwise', 'solver': 'cadical',
        'status': 'SUCCESS', 'time_seconds': 0.05, 'build_time': 0.01, 'solve_time': 0.04,
        'variables_count': 64, 'clauses_constraints_count': 100
    }])
    df.to_csv(tmp_path / "benchmark_results.csv", index=False)
    tex_path = generate_tex(str(tmp_path), str(tmp_path / "report"))
    with open(tex_path, "r") as f:
        content = f.read()

    assert "Apple M4" in content
    assert "macOS 15.7.4" in content
    assert "Python 3.14.7" in content
    assert "tab:software_versions" in content
    assert "1.9.dev15" in content
    assert "9.15.6755" in content
    assert "13.0.3" in content
    assert "22.2.0.1" in content
    assert "2.32.264" in content
    assert "benchmark.py --time-limit 30.0 --repetitions 3" in content
    assert "gurobi_sizelimit" in content



