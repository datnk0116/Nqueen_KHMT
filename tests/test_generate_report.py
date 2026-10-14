import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from generate_report import load_combined, timeout_table, make_sat_table_rows, make_sat_size_table_rows, make_table_rows, format_sat_time

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
