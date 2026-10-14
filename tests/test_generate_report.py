import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from generate_report import load_combined, timeout_table

def test_timeout_table():
    df = load_combined("results")
    to_df = timeout_table(df)
    assert len(to_df) == 5
    
    # Check expected counts
    # CP-SAT, ortools-cpsat, N=128 -> 1
    row1 = to_df[(to_df['paradigm'] == 'CP-SAT') & (to_df['solver'] == 'ortools-cpsat') & (to_df['n'] == 128)]
    assert len(row1) == 1
    assert row1.iloc[0]['timeout_count'] == 1

    # SAT, glucose3, N=64 -> 3
    row2 = to_df[(to_df['paradigm'] == 'SAT') & (to_df['solver'] == 'glucose3') & (to_df['n'] == 64)]
    assert len(row2) == 1
    assert row2.iloc[0]['timeout_count'] == 3

    # SAT, glucose3, N=128 -> 4
    row3 = to_df[(to_df['paradigm'] == 'SAT') & (to_df['solver'] == 'glucose3') & (to_df['n'] == 128)]
    assert len(row3) == 1
    assert row3.iloc[0]['timeout_count'] == 4

    # SAT, maplecm, N=128 -> 3
    row4 = to_df[(to_df['paradigm'] == 'SAT') & (to_df['solver'] == 'maplecm') & (to_df['n'] == 128)]
    assert len(row4) == 1
    assert row4.iloc[0]['timeout_count'] == 3

    # SAT, minisat22, N=128 -> 3
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
