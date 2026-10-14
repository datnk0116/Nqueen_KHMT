import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from generate_report import load_combined, timeout_table, make_sat_table_rows

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

def test_sat_table_rows():
    df = load_combined("results")
    sat_tex = make_sat_table_rows(df)
    lines = sat_tex.strip().split("\n")
    # 25 rows (5 N values x 5 encodings)
    assert len(lines) == 25
    
    # Contains all N values from 8 to 128
    for n_val in [8, 16, 32, 64, 128]:
        assert f"{n_val}" in sat_tex
        
    # Contains "T/O" exactly 13 times
    assert sat_tex.count("T/O") == 13
    
    # Check total data cells across 25 rows x 5 solvers = 125 cells
    total_cells = 0
    for line in lines:
        parts = [p.strip() for p in line.split("&")]
        # parts: [N, encoding, solver1, solver2, solver3, solver4, solver5\\ ]
        # Last part ends with \\
        data_cells = parts[2:]
        total_cells += len(data_cells)
    assert total_cells == 125
