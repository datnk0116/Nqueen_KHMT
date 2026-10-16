import os
import subprocess
import pandas as pd
import numpy as np

def load_combined(results_dir="results"):
    path = os.path.join(results_dir, "benchmark_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required benchmark results file not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    
    if not df.empty:
        keys = ['n', 'paradigm', 'model_variant', 'solver']
        if all(k in df.columns for k in keys):
            dups = df.duplicated(subset=keys, keep=False)
            dup_count = dups.sum()
            if dup_count > 0:
                raise ValueError(f"Duplicate rows detected in {path}: {dup_count} duplicate rows found.")
    return df

def timeout_table(df):
    if df.empty or 'status' not in df.columns:
        return pd.DataFrame(columns=['paradigm', 'solver', 'n', 'timeout_count'])
    timeouts_df = df[df['status'].isin(['TIMEOUT', 'TIME_LIMIT'])]
    if timeouts_df.empty:
        return pd.DataFrame(columns=['paradigm', 'solver', 'n', 'timeout_count'])
    grouped_to = timeouts_df.groupby(['paradigm', 'solver', 'n']).size().reset_index(name='timeout_count')
    return grouped_to

def format_sat_time(row):
    raw_status = row.get('status')
    status = str(raw_status).upper() if (raw_status is not None and not pd.isna(raw_status)) else 'SUCCESS'
    if status in ('TIMEOUT', 'TIME_LIMIT'):
        return "T/O"
    elif status in ('ERROR', 'MEMOUT', 'LICENSE_LIMIT', 'MODEL_INVALID'):
        return "ERR"
    elif status in ('SAT', 'OPTIMAL', 'FEASIBLE', 'SUCCESS', 'SUBOPTIMAL') or (row.get('is_sat') is True and status not in ('TIMEOUT', 'TIME_LIMIT')):
        t = row.get('time_seconds', 0.0)
        if pd.isna(t):
            return "N/A"
        if t < 0.001:
            return f"{t:.2e}"
        else:
            return f"{t:.3f}"
    else:
        return "ERR"

def format_adaptive_time(val_seconds):
    if val_seconds is None or pd.isna(val_seconds):
        return "N/A"
    if val_seconds < 1.0:
        val_ms = val_seconds * 1000
        return f"{val_ms:.1f} ms"
    else:
        return f"{val_seconds:.2f} s"

def make_sat_table_rows(df):
    if df.empty:
        return "No data available & - & - & - & - & - & - \\\\"
    sat_df = df[df['paradigm'] == 'SAT'] if 'paradigm' in df.columns else df
    if sat_df.empty:
        return "No data available & - & - & - & - & - & - \\\\"
    
    solvers = ['cadical', 'glucose3', 'minisat22', 'kissat', 'maplecm']
    sorted_df = sat_df.sort_values(by=['n', 'model_variant']).drop_duplicates(subset=['n', 'model_variant'])
    
    rows = []
    for _, r in sorted_df.iterrows():
        n = r['n']
        variant = r['model_variant']
        row_cells = [str(n), str(variant)]
        for solver in solvers:
            sub = sat_df[(sat_df['n'] == n) & (sat_df['model_variant'] == variant) & (sat_df['solver'] == solver)]
            if sub.empty:
                row_cells.append("-")
            else:
                row = sub.iloc[0]
                row_cells.append(format_sat_time(row))
        rows.append(" & ".join(row_cells) + " \\\\")
    return "\n".join(rows)

def make_sat_size_table_rows(df):
    if df.empty:
        return "No data available & - & - & - \\\\"
    sat_df = df[df['paradigm'] == 'SAT'] if 'paradigm' in df.columns else df
    if sat_df.empty:
        return "No data available & - & - & - \\\\"
    
    sorted_df = sat_df.sort_values(by=['n', 'model_variant']).drop_duplicates(subset=['n', 'model_variant'])
    rows = []
    for _, r in sorted_df.iterrows():
        n = r['n']
        variant = r['model_variant']
        sub = sat_df[(sat_df['n'] == n) & (sat_df['model_variant'] == variant)]
        if 'variables_count' in sub.columns:
            valid_sub = sub[~sub['status'].isin(['TIMEOUT', 'TIME_LIMIT', 'ERROR', 'MEMOUT', 'LICENSE_LIMIT', 'MODEL_INVALID']) & (sub['variables_count'] > 0)]
        else:
            valid_sub = pd.DataFrame()
        if valid_sub.empty:
            rows.append(f"{n} & {variant} & - & - \\\\")
        else:
            var_nunique = valid_sub['variables_count'].nunique()
            clause_col = 'clauses_constraints_count' if 'clauses_constraints_count' in valid_sub.columns else 'constraints_count'
            clause_nunique = valid_sub[clause_col].nunique()
            if var_nunique > 1 or clause_nunique > 1:
                raise ValueError(f"Non-unique variable or clause counts found for n={n}, variant={variant}")
            v_count = valid_sub['variables_count'].iloc[0]
            c_count = valid_sub[clause_col].iloc[0]
            rows.append(f"{n} & {variant} & {v_count} & {c_count} \\\\")
    return "\n".join(rows)

def make_table_rows(subset_df, max_rows=None):
    if subset_df.empty:
        return "No data available & - & - & - & - & - \\\\"
    sorted_df = subset_df.sort_values(by=['n', 'model_variant', 'solver'])
    rows = []
    count = 0
    for _, row in sorted_df.iterrows():
        if max_rows is not None and count >= max_rows:
            break
        raw_status = row.get('status')
        status = str(raw_status).upper() if (raw_status is not None and not pd.isna(raw_status)) else 'SUCCESS'
        if status in ('TIMEOUT', 'TIME_LIMIT'):
            v_str = "-"
            c_str = "-"
            t_str = "T/O"
        elif status in ('ERROR', 'MEMOUT', 'LICENSE_LIMIT', 'MODEL_INVALID'):
            v_str = "-"
            c_str = "-"
            t_str = "ERR"
        elif status in ('SAT', 'OPTIMAL', 'FEASIBLE', 'SUCCESS', 'SUBOPTIMAL') or (row.get('is_sat') is True and status not in ('TIMEOUT', 'TIME_LIMIT')):
            v_str = str(row.get('variables_count', 0))
            constraints = row.get('clauses_constraints_count', row.get('constraints_count', 0))
            c_str = str(constraints)
            t_sec = row.get('time_seconds', 0.0)
            t_str = f"{t_sec:.6f}" if not pd.isna(t_sec) else "N/A"
        else:
            v_str = "-"
            c_str = "-"
            t_str = "ERR"
        rows.append(f"{row.get('n', 0)} & {row.get('model_variant', '')} & {row.get('solver', '')} & {v_str} & {c_str} & {t_str} \\\\")
        count += 1
    return "\n".join(rows)

def build_discussion_stats(combined_df):
    sat_df = combined_df[combined_df['paradigm'] == 'SAT'] if (not combined_df.empty and 'paradigm' in combined_df.columns) else pd.DataFrame()
    cpsat_df = combined_df[combined_df['paradigm'] == 'CP-SAT'] if (not combined_df.empty and 'paradigm' in combined_df.columns) else pd.DataFrame()
    cplex_cp_df = combined_df[combined_df['paradigm'] == 'CPLEX-CP'] if (not combined_df.empty and 'paradigm' in combined_df.columns) else pd.DataFrame()

    sat_128 = sat_df[sat_df['n'] == 128] if (not sat_df.empty and 'n' in sat_df.columns) else pd.DataFrame()
    if not sat_128.empty and 'status' in sat_128.columns:
        sat_128_timeouts = int(len(sat_128[sat_128['status'].isin(['TIMEOUT', 'TIME_LIMIT'])]))
        sat_128_total = int(len(sat_128))
        g3_to = int(len(sat_128[(sat_128['solver'] == 'glucose3') & (sat_128['status'].isin(['TIMEOUT', 'TIME_LIMIT']))])) if 'solver' in sat_128.columns else 0
        m22_to = int(len(sat_128[(sat_128['solver'] == 'minisat22') & (sat_128['status'].isin(['TIMEOUT', 'TIME_LIMIT']))])) if 'solver' in sat_128.columns else 0
        mcm_to = int(len(sat_128[(sat_128['solver'] == 'maplecm') & (sat_128['status'].isin(['TIMEOUT', 'TIME_LIMIT']))])) if 'solver' in sat_128.columns else 0
    else:
        sat_128_timeouts = 0
        sat_128_total = 0
        g3_to = 0
        m22_to = 0
        mcm_to = 0

    cplex_cp_8_df = cplex_cp_df[cplex_cp_df['n'] == 8] if (not cplex_cp_df.empty and 'n' in cplex_cp_df.columns) else pd.DataFrame()
    med_8 = cplex_cp_8_df['time_seconds'].median() if (not cplex_cp_8_df.empty and 'time_seconds' in cplex_cp_8_df.columns) else float('nan')
    cplex_cp_8_s = med_8 if (med_8 is not None and not pd.isna(med_8)) else float('nan')
    cplex_cp_8_ms = cplex_cp_8_s * 1000 if not pd.isna(cplex_cp_8_s) else float('nan')

    cplex_cp_128_df = cplex_cp_df[cplex_cp_df['n'] == 128] if (not cplex_cp_df.empty and 'n' in cplex_cp_df.columns) else pd.DataFrame()
    med_128 = cplex_cp_128_df['time_seconds'].median() if (not cplex_cp_128_df.empty and 'time_seconds' in cplex_cp_128_df.columns) else float('nan')
    cplex_cp_128_s = med_128 if (med_128 is not None and not pd.isna(med_128)) else float('nan')
    cplex_cp_128_ms = cplex_cp_128_s * 1000 if not pd.isna(cplex_cp_128_s) else float('nan')

    cplex_cp_128_solve_s = cplex_cp_128_df['solve_time'].median() if (not cplex_cp_128_df.empty and 'solve_time' in cplex_cp_128_df.columns and not pd.isna(cplex_cp_128_df['solve_time'].median())) else float('nan')
    cplex_cp_128_build_ms = cplex_cp_128_df['build_time'].median() * 1000 if (not cplex_cp_128_df.empty and 'build_time' in cplex_cp_128_df.columns and not pd.isna(cplex_cp_128_df['build_time'].median())) else float('nan')

    if not cplex_cp_df.empty and 'solve_time' in cplex_cp_df.columns:
        unique_st = sorted(cplex_cp_df['solve_time'].dropna().unique())
        if len(unique_st) > 1:
            cplex_cp_solve_times_str = ", ".join([f"{v:.2f}" for v in unique_st[:-1]]) + " and " + f"{unique_st[-1]:.2f}"
        elif len(unique_st) == 1:
            cplex_cp_solve_times_str = f"{unique_st[0]:.2f}"
        else:
            cplex_cp_solve_times_str = "N/A"
    else:
        cplex_cp_solve_times_str = "N/A"

    cpsat_int_sub = cpsat_df[(cpsat_df['n'] == 128) & (cpsat_df['model_variant'] == 'integer')] if (not cpsat_df.empty and 'n' in cpsat_df.columns and 'model_variant' in cpsat_df.columns) else pd.DataFrame()
    cpsat_int_128 = cpsat_int_sub['time_seconds'].median() if (not cpsat_int_sub.empty and 'time_seconds' in cpsat_int_sub.columns and not pd.isna(cpsat_int_sub['time_seconds'].median())) else float('nan')

    cpsat_bool_sub = cpsat_df[(cpsat_df['n'] == 128) & (cpsat_df['model_variant'] == 'boolean')] if (not cpsat_df.empty and 'n' in cpsat_df.columns and 'model_variant' in cpsat_df.columns) else pd.DataFrame()
    cpsat_bool_128 = cpsat_bool_sub['time_seconds'].median() if (not cpsat_bool_sub.empty and 'time_seconds' in cpsat_bool_sub.columns and not pd.isna(cpsat_bool_sub['time_seconds'].median())) else float('nan')

    def non_failure(df):
        return df[~df['status'].isin(['TIMEOUT', 'TIME_LIMIT', 'ERROR', 'MEMOUT', 'LICENSE_LIMIT', 'MODEL_INVALID'])] if ('status' in df.columns) else df

    cadical_sub = non_failure(sat_df[sat_df['solver'] == 'cadical']) if (not sat_df.empty and 'solver' in sat_df.columns) else pd.DataFrame()
    kissat_sub = non_failure(sat_df[sat_df['solver'] == 'kissat']) if (not sat_df.empty and 'solver' in sat_df.columns) else pd.DataFrame()
    maplecm_sub = non_failure(sat_df[sat_df['solver'] == 'maplecm']) if (not sat_df.empty and 'solver' in sat_df.columns) else pd.DataFrame()

    cadical_max = cadical_sub['time_seconds'].max() if (not cadical_sub.empty and 'time_seconds' in cadical_sub.columns and not pd.isna(cadical_sub['time_seconds'].max())) else float('nan')
    kissat_max = kissat_sub['time_seconds'].max() if (not kissat_sub.empty and 'time_seconds' in kissat_sub.columns and not pd.isna(kissat_sub['time_seconds'].max())) else float('nan')

    if not maplecm_sub.empty and 'time_seconds' in maplecm_sub.columns and not maplecm_sub['time_seconds'].dropna().empty:
        max_idx = maplecm_sub['time_seconds'].idxmax()
        maplecm_max = maplecm_sub.loc[max_idx, 'time_seconds']
        maplecm_max_variant = str(maplecm_sub.loc[max_idx, 'model_variant']) if 'model_variant' in maplecm_sub.columns else "N/A"
    else:
        maplecm_max = float('nan')
        maplecm_max_variant = "N/A"

    has_solver_reported_time = not combined_df.empty and 'solver_reported_time' in combined_df.columns and combined_df['solver_reported_time'].notna().any()
    cplex_cp_128_reported_s = cplex_cp_128_df['solver_reported_time'].median() if (not cplex_cp_128_df.empty and 'solver_reported_time' in cplex_cp_128_df.columns and not pd.isna(cplex_cp_128_df['solver_reported_time'].median())) else float('nan')

    return {
        'cplex_cp_8_s': cplex_cp_8_s,
        'cplex_cp_8_ms': cplex_cp_8_ms,
        'cplex_cp_128_s': cplex_cp_128_s,
        'cplex_cp_128_ms': cplex_cp_128_ms,
        'cplex_cp_128_solve_s': cplex_cp_128_solve_s,
        'cplex_cp_128_build_ms': cplex_cp_128_build_ms,
        'cplex_cp_solve_times_str': cplex_cp_solve_times_str,
        'cpsat_int_128_s': cpsat_int_128,
        'cpsat_bool_128_s': cpsat_bool_128,
        'cadical_max': cadical_max,
        'kissat_max': kissat_max,
        'maplecm_max': maplecm_max,
        'maplecm_max_variant': maplecm_max_variant,
        'sat_128_timeouts': sat_128_timeouts,
        'sat_128_total': sat_128_total,
        'g3_to': g3_to,
        'm22_to': m22_to,
        'mcm_to': mcm_to,
        'has_solver_reported_time': has_solver_reported_time,
        'cplex_cp_128_reported_s': cplex_cp_128_reported_s,
    }

def generate_tex(results_dir="results", out_dir="report"):
    os.makedirs(out_dir, exist_ok=True)
    
    # Ensure style files and directories are present in report directory
    src_report_dir = "report"
    parent_report = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../report")
    if os.path.exists(parent_report) and parent_report != os.path.abspath(out_dir):
        src_report_dir = parent_report
    if os.path.exists(src_report_dir) and src_report_dir != os.path.abspath(out_dir):
        for root, dirs, files in os.walk(src_report_dir):
            rel = os.path.relpath(root, src_report_dir)
            target_sub = os.path.join(out_dir, rel) if rel != "." else out_dir
            os.makedirs(target_sub, exist_ok=True)
            for f in files:
                src_p = os.path.join(root, f)
                dst_p = os.path.join(target_sub, f)
                if not os.path.exists(dst_p):
                    try:
                        import shutil
                        shutil.copy(src_p, dst_p)
                    except Exception:
                        pass
    
    combined_df = load_combined(results_dir)
    if combined_df.empty:
        raise ValueError(f"Benchmark results dataset in '{results_dir}' contains no valid records.")
    stats = build_discussion_stats(combined_df)
    
    sat_df = combined_df[combined_df['paradigm'] == 'SAT'] if not combined_df.empty else pd.DataFrame()
    cpsat_df = combined_df[combined_df['paradigm'] == 'CP-SAT'] if not combined_df.empty else pd.DataFrame()
    mip_df = combined_df[combined_df['paradigm'] == 'MIP'] if not combined_df.empty else pd.DataFrame()
    gurobi_df = combined_df[combined_df['paradigm'] == 'Gurobi'] if not combined_df.empty else pd.DataFrame()
    cplex_df = combined_df[combined_df['paradigm'].isin(['CPLEX-MIP', 'CPLEX-CP'])] if not combined_df.empty else pd.DataFrame()
    cplex_cp_df = combined_df[combined_df['paradigm'] == 'CPLEX-CP'] if not combined_df.empty else pd.DataFrame()

    has_mpl = False
    try:
        import matplotlib.pyplot as plt
        has_mpl = True
    except ImportError:
        pass

    if has_mpl and not combined_df.empty:
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        # Plot 1: SAT Solvers time vs n by encoding (CaDiCaL)
        if not sat_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for variant in sat_df['model_variant'].unique():
                sub = sat_df[(sat_df['model_variant'] == variant) & (sat_df['solver'] == 'cadical')]
                if not sub.empty:
                    grouped = sub.groupby('n')['time_seconds'].mean()
                    ax.plot(grouped.index, grouped.values, marker='o', label=f"SAT (CaDiCaL) - {variant}")
            ax.set_xlabel("Board Size ($N$)")
            ax.set_ylabel("Time (seconds)")
            ax.set_title("SAT Solvers Performance by Encoding Variant (CaDiCaL)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "sat_time_cadical.png"), dpi=300)
            plt.close()

        # Plot 2: Paradigm comparison across all solver groups (model time: build + solve, timeouts counted as 30 s, excluding LICENSE_LIMIT/MEMOUT/ERROR)
        fig, ax = plt.subplots(figsize=(8, 5))
        valid_paradigm_df = combined_df[~combined_df['status'].isin(['LICENSE_LIMIT', 'MEMOUT', 'ERROR', 'MODEL_INVALID'])].copy() if not combined_df.empty else combined_df
        has_paradigm_lines = False
        if not valid_paradigm_df.empty:
            if 'model_time' in valid_paradigm_df.columns:
                # If model_time has NaN but build and solve exist, fill them
                nan_mask = valid_paradigm_df['model_time'].isna() & valid_paradigm_df['build_time'].notna() & valid_paradigm_df['solve_time'].notna()
                valid_paradigm_df.loc[nan_mask, 'model_time'] = valid_paradigm_df.loc[nan_mask, 'build_time'] + valid_paradigm_df.loc[nan_mask, 'solve_time']
            elif 'build_time' in valid_paradigm_df.columns and 'solve_time' in valid_paradigm_df.columns:
                valid_paradigm_df['model_time'] = valid_paradigm_df['build_time'] + valid_paradigm_df['solve_time']
            else:
                valid_paradigm_df['model_time'] = valid_paradigm_df['time_seconds']
            valid_paradigm_df.loc[valid_paradigm_df['status'].isin(['TIMEOUT', 'TIME_LIMIT']), 'model_time'] = 30.0
            for paradigm in valid_paradigm_df['paradigm'].unique():
                sub = valid_paradigm_df[valid_paradigm_df['paradigm'] == paradigm]
                if not sub.empty:
                    grouped = sub.groupby('n')['model_time'].mean()
                    ax.plot(grouped.index, grouped.values, marker='s', label=paradigm)
                    has_paradigm_lines = True
        ax.set_yscale('log')
        ax.set_xlabel("Board Size ($N$)")
        ax.set_ylabel("Mean Model Time (seconds, log scale, build + solve, timeouts counted as 30 s)")
        ax.set_title("Paradigm Comparison Across Board Sizes (model time: build + solve, timeouts counted as 30 s)")
        if has_paradigm_lines:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "paradigm_comparison.png"), dpi=300)
        plt.close()

        # Plot 3: Variable count scaling (exclude TIMEOUT/TIME_LIMIT/MEMOUT/LICENSE_LIMIT/ERROR)
        fig, ax = plt.subplots(figsize=(8, 5))
        valid_var_df = combined_df[~combined_df['status'].isin(['TIMEOUT', 'TIME_LIMIT', 'MEMOUT', 'LICENSE_LIMIT', 'ERROR', 'MODEL_INVALID'])]
        has_var_lines = False
        for paradigm in valid_var_df['paradigm'].unique():
            sub = valid_var_df[valid_var_df['paradigm'] == paradigm]
            if not sub.empty and 'variables_count' in sub.columns:
                grouped = sub.groupby('n')['variables_count'].mean()
                if not grouped.empty:
                    ax.plot(grouped.index, grouped.values, marker='^', label=paradigm)
                    has_var_lines = True
        ax.set_yscale('log')
        ax.set_xlabel("Board Size ($N$)")
        ax.set_ylabel("Variable Count (log scale)")
        ax.set_title("Variable Scaling Across Paradigms")
        if has_var_lines:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "variables_scaling.png"), dpi=300)
        plt.close()

        # Plot 4: Build vs Solve time (exclude TIMEOUT/TIME_LIMIT/MEMOUT/LICENSE_LIMIT/ERROR, reindex)
        if 'build_time' in combined_df.columns and 'solve_time' in combined_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            valid_bs_df = combined_df[~combined_df['status'].isin(['TIMEOUT', 'TIME_LIMIT', 'MEMOUT', 'LICENSE_LIMIT', 'ERROR', 'MODEL_INVALID'])]
            build_grouped = valid_bs_df.groupby('paradigm')['build_time'].mean()
            solve_grouped = valid_bs_df.groupby('paradigm')['solve_time'].mean()
            all_paradigms = sorted(list(set(build_grouped.index).union(set(solve_grouped.index))))
            build_grouped = build_grouped.reindex(all_paradigms, fill_value=0)
            solve_grouped = solve_grouped.reindex(all_paradigms, fill_value=0)
            
            x = np.arange(len(all_paradigms))
            width = 0.35
            ax.bar(x - width/2, build_grouped.values, width, label='Build Time')
            ax.bar(x + width/2, solve_grouped.values, width, label='Solve Time')
            ax.set_xticks(x)
            ax.set_xticklabels(all_paradigms, rotation=15)
            ax.set_yscale('log')
            ax.set_ylabel("Seconds (log scale)")
            ax.set_title("Model Construction vs Active Solving Time by Paradigm")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "build_vs_solve.png"), dpi=300)
            plt.close()

    sat_table_tex = make_sat_table_rows(sat_df)
    sat_size_table_tex = make_sat_size_table_rows(sat_df)
    cpsat_table_tex = make_table_rows(cpsat_df)
    mip_table_tex = make_table_rows(mip_df)
    gurobi_table_tex = make_table_rows(gurobi_df)
    cplex_table_tex = make_table_rows(cplex_df)

    # Generate timeout analysis table data using timeout_table helper
    timeout_rows = []
    to_df = timeout_table(combined_df)
    if not to_df.empty:
        for _, row in to_df.iterrows():
            timeout_rows.append(f"{row['paradigm']} & {row['solver']} & {row['n']} & {row['timeout_count']} \\\\")
    else:
        timeout_rows.append("No timeouts recorded & - & - & - \\\\")
    timeout_table_tex = "\n".join(timeout_rows)

    cplex_cp_8_str = format_adaptive_time(stats['cplex_cp_8_s'])
    cplex_cp_128_str = format_adaptive_time(stats['cplex_cp_128_s'])
    cpsat_int_128_str = f"{stats['cpsat_int_128_s']:.3f}" if not pd.isna(stats['cpsat_int_128_s']) else "N/A"
    cadical_max_str = f"{stats['cadical_max']:.2f}" if not pd.isna(stats['cadical_max']) else "N/A"
    kissat_max_str = f"{stats['kissat_max']:.2f}" if not pd.isna(stats['kissat_max']) else "N/A"
    maplecm_max_str = f"{stats['maplecm_max']:.2f}" if not pd.isna(stats['maplecm_max']) else "N/A"
    maplecm_variant_str = stats['maplecm_max_variant']
    cplex_cp_128_build_str = f"{stats['cplex_cp_128_build_ms']:.1f}" if not pd.isna(stats['cplex_cp_128_build_ms']) else "N/A"

    if not pd.isna(stats['cplex_cp_128_s']) and not pd.isna(stats['cpsat_int_128_s']):
        comp_runtime_text = f"At $N = 128$, recorded runtimes range from approximately {cplex_cp_128_str} for CPLEX CP to {cpsat_int_128_str} s for CP-SAT"
    elif not pd.isna(stats['cplex_cp_128_s']):
        comp_runtime_text = f"At $N = 128$, CPLEX CP records a runtime of approximately {cplex_cp_128_str}"
    elif not pd.isna(stats['cpsat_int_128_s']):
        comp_runtime_text = f"At $N = 128$, CP-SAT records a runtime of approximately {cpsat_int_128_str} s"
    else:
        comp_runtime_text = "At $N = 128$, experimental runtime data for CPLEX CP and CP-SAT is not available"

    if not pd.isna(stats['cadical_max']) and not pd.isna(stats['kissat_max']):
        cadical_kissat_text = f"CaDiCaL and Kissat handle large clause databases efficiently up to $N = 128$ (with maximum recorded runtimes of approximately {cadical_max_str} s for CaDiCaL and {kissat_max_str} s for Kissat)"
    elif not pd.isna(stats['cadical_max']):
        cadical_kissat_text = f"CaDiCaL handles large clause databases efficiently up to $N = 128$ (with a maximum recorded runtime of approximately {cadical_max_str} s)"
    elif not pd.isna(stats['kissat_max']):
        cadical_kissat_text = f"Kissat handles large clause databases efficiently up to $N = 128$ (with a maximum recorded runtime of approximately {kissat_max_str} s)"
    else:
        cadical_kissat_text = "CaDiCaL and Kissat experimental data is not available at $N = 128$"

    if not pd.isna(stats['maplecm_max']):
        maplecm_text = f", whereas MapleCM reaches a maximum recorded runtime of approximately {maplecm_max_str} seconds under {maplecm_variant_str} encoding at $N = 128$"
    else:
        maplecm_text = ""

    if stats['sat_128_total'] > 0:
        if stats['sat_128_timeouts'] > 0:
            sat_timeout_text = (
                f", and Glucose3, MiniSat22, and MapleCM encounter frequent timeouts "
                f"({stats['sat_128_timeouts']}/{stats['sat_128_total']} cases timed out at $N = 128$, "
                f"specifically Glucose3: {stats['g3_to']}, MiniSat22: {stats['m22_to']}, MapleCM: {stats['mcm_to']})"
            )
        else:
            sat_timeout_text = f", and no timeouts were encountered across tested SAT configurations at $N = 128$ (0/{stats['sat_128_total']} timed out)"
    else:
        sat_timeout_text = ", whereas no experimental SAT timeout data is available at $N = 128$ (insufficient data)"

    latex_content = r"""\documentclass[a4paper,fleqn]{cas-sc}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{hyperref}
\usepackage{float}
\usepackage[numbers]{natbib}

\ifdefined\DeclareUnicodeCharacter
  \DeclareUnicodeCharacter{1EC5}{\~{\^e}}
  \DeclareUnicodeCharacter{1EA3}{\v{a}}
  \DeclareUnicodeCharacter{0110}{\DJ}
  \DeclareUnicodeCharacter{1EA1}{\d{a}}
  \DeclareUnicodeCharacter{0111}{\dj}
\fi

\begin{document}

\title{\textbf{Comprehensive Benchmark Report: N-Queens Problem Across Six Solver Groups and Paradigms}}
\shorttitle{N-Queens Benchmark Across Six Solver Groups}
\shortauthors{Nguyễn Khả Đạt}
\author[1]{Nguyễn Khả Đạt}
\ead{24021404@vnu.edu.vn}
\fntext[1]{Course: Modern Topics in Computer Science (2627I\_INT3011E\_4) -- Student ID: 24021404}

\affiliation[1]{organization={Faculty of Information Technology, University of Engineering and Technology, Vietnam National University},
                city={Hanoi},
                country={Vietnam}}

\begin{abstract}
This report presents an empirical evaluation of solving the classic $N$-Queens problem across six major solver groups and modeling paradigms: Propositional Satisfiability (SAT with CaDiCaL, Glucose3, MiniSat22, Kissat, and MapleCM), Constraint Programming via CP-SAT (OR-Tools), Mixed-Integer Programming (MIP with CBC, SCIP), High-Performance Commercial MIP (Gurobi), Commercial CPLEX MIP, and Commercial CPLEX CP. We evaluate encoding efficiency, variable scaling, constraint footprint, model construction overhead, and execution times across evaluated board sizes up to $N = 128$ under standardized 3-repetition benchmark runs with hard timeouts. Detailed performance metrics and visualizations highlight the trade-offs between expressive global constraints and low-level clausal representations, alongside timeout analyses across evaluated instances, CP solve overhead management, and solution status semantics (FEASIBLE vs OPTIMAL).
\end{abstract}

\begin{keywords}
N-Queens \sep Propositional Satisfiability \sep Constraint Programming \sep Mixed-Integer Programming \sep Benchmarking \sep PySAT
\end{keywords}

\maketitle

\section{Introduction}
The $N$-Queens problem serves as a canonical benchmark in artificial intelligence, automated reasoning, constraint satisfaction, and combinatorial optimization. Placing $N$ non-attacking queens on an $N \times N$ chessboard tests the scalability, constraint propagation, and search pruning power of modern discrete solvers. 

This study expands the evaluation framework to cover six distinct solver groups:
\begin{itemize}
    \item \textbf{SAT Solvers}: CaDiCaL \citep{biere2020cadical}, Glucose3, MiniSat22, Kissat, and MapleCM evaluated across Pairwise, Binary, Sequential Counter, Commander, and Product cardinality encodings (implemented using the PySAT toolkit \citep{ignatiev2018pysat}).
    \item \textbf{Constraint Programming (CP-SAT)}: OR-Tools CP-SAT using Integer, Boolean, and Allowed assignments configurations \citep{perron2021ortools}.
    \item \textbf{Open-Source MIP}: CBC and SCIP solvers using Binary and Integer linear formulations.
    \item \textbf{Commercial MIP (Gurobi)}: High-performance Gurobi optimization engine with advanced presolve and branch-and-cut routines (noting trial license bounds at $N \le 44$).
    \item \textbf{Commercial CPLEX MIP}: IBM ILOG CPLEX Optimizer using the Mathematical Programming (MIP) engine.
    \item \textbf{Commercial CPLEX CP}: IBM ILOG CPLEX Optimizer using the Constraint Programming (CP) engine, scaling up to $N = 128$.
\end{itemize}

\section{Encodings and Mathematical Formulations}
Different paradigms map the combinatorial structure of the $N$-Queens problem into mathematical constraints in distinct ways:

\subsection{Propositional Satisfiability (SAT) Encodings}
Boolean decision variables $x_{i,j}$ denote whether a queen is placed at row $i$, column $j$ ($1 \le i, j \le N$), requiring $N^2$ primary decision variables. As implemented in \texttt{nqueens\_encoding} (\texttt{src/encodings.py}), the problem is formulated via four distinct constraint families:
\begin{enumerate}
    \item \textbf{At-Least-One (ALO) per row}: $\bigvee_{j=1}^N x_{i,j}$ for each row $1 \le i \le N$, contributing $N$ clauses.
    \item \textbf{At-Most-One (AMO) per row}: $\text{AMO}(x_{i,1}, \dots, x_{i,N})$ for each row $1 \le i \le N$ ($N$ groups of size $N$).
    \item \textbf{At-Most-One (AMO) per column}: $\text{AMO}(x_{1,j}, \dots, x_{N,j})$ for each column $1 \le j \le N$ ($N$ groups of size $N$).
    \item \textbf{At-Most-One (AMO) per diagonal}: $\text{AMO}(x_{i,j} : (i,j) \in \text{diag})$ for each major and minor diagonal of length $k \ge 2$, comprising $2(2N - 3)$ groups of sizes $k \in \{2, \dots, N\}$.
\end{enumerate}
Crucially, all three AMO constraint families (rows, columns, and diagonals) invoke the identical modular AMO encoding function. Five representative AMO encodings are evaluated:
\begin{itemize}
    \item \textbf{Pairwise}: Adds negative binary clauses $(\neg u \lor \neg v)$ for each distinct pair. Per AMO group of size $n$, it generates $\binom{n}{2} = \frac{n(n-1)}{2}$ clauses without auxiliary variables. Summed over all rows, columns, and diagonals, the total clause count is $2N\binom{N}{2} + 2\left(2\binom{N}{3} + \binom{N}{2}\right) + N = \frac{5}{3}N^3 - 2N^2 + \frac{4}{3}N = \Theta(N^3)$ (yielding 428,800 clauses at $N = 64$ and 3,462,656 at $N = 128$).
    \item \textbf{Binary (Bitwise)}: Assigns binary index vectors to literals using $\lceil \log_2 n \rceil$ auxiliary variables and $n \lceil \log_2 n \rceil$ clauses per group of size $n$, resulting in $\Theta(N^2 \log N)$ total clauses and $\Theta(N \log N)$ auxiliary variables.
    \item \textbf{Sequential Counter (Sinz)} \citep{sinz2005}: Chains partial sums with $n - 1$ auxiliary variables and $3n - 4$ clauses per group ($n \ge 2$), achieving $\Theta(N^2)$ total clauses and $\Theta(N^2)$ auxiliary variables.
    \item \textbf{Commander (Klieber \& Kwon)} \citep{klieber2007efficient}: Recursively partitions literals into blocks of size 3 and elects commander variables, posting $\Theta(n)$ clauses and $\Theta(n)$ auxiliary variables per group, yielding $\Theta(N^2)$ total clauses and $\Theta(N^2)$ auxiliary variables.
    \item \textbf{Product (Chen)} \citep{chen2010new}: Embeds literals into a 2D grid of size $p \times q$ with $p = \lceil \sqrt{n} \rceil$ and $q = \lceil n/p \rceil$, enforcing binary AMO on row and column variables alongside $2n$ assignment clauses, yielding $\Theta(n)$ clauses and $\Theta(\sqrt{n})$ auxiliary variables per group ($\Theta(N^2)$ total clauses and $\Theta(N^{1.5})$ auxiliary variables).
\end{itemize}

\begin{table}[htbp]
\centering
\caption{Asymptotic Size of the SAT Encodings} \label{tab:sat_encodings_complexity}
\begin{tabular}{lllll}
\toprule
Encoding & Clauses per Group (Size $n$) & Aux. Variables per Group & Total Clauses & Total Aux. Variables \\
\midrule
Pairwise & $\binom{n}{2} = \frac{n(n-1)}{2}$ & $0$ & $\Theta(N^3)$ & $0$ \\
Binary & $n \lceil \log_2 n \rceil$ & $\lceil \log_2 n \rceil$ & $\Theta(N^2 \log N)$ & $\Theta(N \log N)$ \\
Sequential & $3n - 4$ & $n - 1$ & $\Theta(N^2)$ & $\Theta(N^2)$ \\
Commander & $\Theta(n)$ & $\Theta(n)$ & $\Theta(N^2)$ & $\Theta(N^2)$ \\
Product & $\Theta(n)$ & $\Theta(\sqrt{n})$ & $\Theta(N^2)$ & $\Theta(N^{1.5})$ \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Constraint Programming (CP-SAT) \& Global Constraints}
OR-Tools CP-SAT utilizes integer decision variables $q_i \in \{0, 1, \dots, N-1\}$ representing the column of the queen in row $i$. Global constraints enforce:
\begin{enumerate}
    \item $\text{AllDifferent}(q_1, \dots, q_N)$ (columns)
    \item $\text{AllDifferent}(q_1 + 1, q_2 + 2, \dots, q_N + N)$ (main diagonals)
    \item $\text{AllDifferent}(q_1 - 1, q_2 - 2, \dots, q_N - N)$ (anti-diagonals)
\end{enumerate}
This formulation utilizes $N$ primary integer decision variables, alongside auxiliary variables introduced during model construction (resulting in 383 variables for $N = 128$ in the integer configuration proto), and 3 global constraints, pruning infeasible branches effectively.

\subsection{Mixed-Integer Programming (MIP)}
MIP models formulate the problem as linear inequalities over binary variables $x_{i,j} \in \{0, 1\}$ or integer variables $q_i \in \{0, 1, \dots, N-1\}$, coupled with linear diagonal separation cuts:
$$\sum_{i=1}^N x_{i,j} = 1 \quad (\forall j), \quad \sum_{j=1}^N x_{i,j} = 1 \quad (\forall i), \quad \sum_{(i,j) \in \text{diag}} x_{i,j} \le 1$$

\section{Experimental Setup}

\subsection{Hardware and Software Environment}
All benchmarks were conducted on a MacBook Air equipped with an Apple M4 processor and 16~GB of unified memory, operating under macOS 15.7.4 (Sequoia). The evaluation runtime was isolated within a dedicated Python 3.14.7 virtual environment (\texttt{SAT}). Table~\ref{tab:software_versions} lists the exact software library and solver package versions employed in the reference benchmark. The SAT solvers are directly bundled within PySAT 1.9.dev15 as compiled C/C++ libraries, corresponding to the concrete solver classes instantiated in \texttt{src/sat\_model.py}: CaDiCaL 1.5.3 (\texttt{Cadical153}), Kissat 4.0.4 (\texttt{Kissat404}), Glucose 3.0 (\texttt{Glucose3}), MiniSat 2.2 (\texttt{Minisat22}), and MapleCOMSPS/MapleCM (\texttt{MapleCM}).

\begin{table}[htbp]
\centering
\caption{Software and Solver Package Versions} \label{tab:software_versions}
\begin{tabular}{lll}
\toprule
Package / Tool & Version & Solvers / Modules Included \\
\midrule
PySAT & 1.9.dev15 & CaDiCaL 1.5.3, Kissat 4.0.4, Glucose 3.0, MiniSat 2.2, MapleCM \\
OR-Tools & 9.15.6755 & CP-SAT solver, Linear Solver (CBC, SCIP interfaces) \\
Gurobi (\texttt{gurobipy}) & 13.0.3 & Gurobi Optimizer (size-limited trial license) \\
IBM ILOG CPLEX (\texttt{cplex}) & 22.2.0.1 & CPLEX Mathematical Programming (MIP) engine \\
DOcplex (\texttt{docplex}) & 2.32.264 & CPLEX CP Optimizer API (\texttt{cpoptimizer}) \\
Python & 3.14.7 & Execution runtime (virtualenv \texttt{SAT}, macOS 15.7.4) \\
\bottomrule
\end{tabular}
\end{table}

Solver configurations, worker allocations, and timeout parameters are specified as follows:
\begin{itemize}
    \item \textbf{SAT Solvers (PySAT)}: Solvers run as single-threaded C++ processes using deterministic default search heuristics without explicit random seeds. For solvers with cooperative interrupt support (\texttt{Glucose3}, \texttt{Minisat22}, \texttt{MapleCM}), internal soft timeouts are handled via a Python \texttt{threading.Timer} invoking \texttt{solve\_limited(expect\_interrupt=True)}. Solvers without native interrupt mechanisms in PySAT (\texttt{Cadical153}, \texttt{Kissat404}) rely on parent process hard timeouts.
    \item \textbf{OR-Tools CP-SAT}: Executes with default multithreaded worker allocation (the number of search workers is not explicitly pinned in code, adhering to OR-Tools heuristics based on host hardware; thus solving efficiency reflects available CPU cores), default random seed, and remaining time budget passed via \texttt{parameters.max\_time\_in\_seconds}.
    \item \textbf{MIP Solvers (CBC, SCIP)}: Invoked through OR-Tools \texttt{pywraplp.Solver} with default single-threaded parameters and time limits specified via \texttt{set\_time\_limit()} in milliseconds.
    \item \textbf{Gurobi}: Commercial solver Gurobi operated under trial license limitations ($N \le 44$) with default multithreading (\texttt{Threads = 0}), default random seed (\texttt{Seed = 0}), and remaining time budget forwarded via \texttt{Params.TimeLimit}.
    \item \textbf{CPLEX MIP and CP Optimizer}: CPLEX operated under full license support allowing scaling up to $N = 128$. CPLEX MIP uses default thread allocation and random seeds, with time limits passed via \texttt{set\_time\_limit()}. CP Optimizer (\texttt{cplex-cp}) executes via the external \texttt{cpoptimizer} binary with default worker allocation, default seed, \texttt{LogVerbosity = Quiet}, and time limits configured via \texttt{params["TimeLimit"]}.
\end{itemize}

\subsection{Benchmark Instances and Scaling}
Evaluated board sizes span the progression $N \in \{8, 16, 32, 44, 64, 128\}$. The committed benchmark dataset comprises 170 distinct configuration records, where each record represents a unique combination of $(N, \text{model variant}, \text{solver})$ evaluated over 3 complete repetitions, rather than an exhaustive Cartesian product across all solvers and scales. Open-source MIP solvers (CBC, SCIP) were evaluated up to $N = 64$, whereas SAT, CP-SAT, and CPLEX configurations scaled up to $N = 128$.

The intermediate instance size $N = 44$ was evaluated exclusively for Gurobi to analyze solver behavior at the boundary of its license capacity: the restricted license supports at most 2,000 variables and 2,000 linear constraints \citep{gurobi_sizelimit}. Under the binary matrix formulation with $N^2$ binary decision variables and row, column, and diagonal cuts, $N = 44$ produces $44^2 = 1,936$ variables ($\le 2,000$) and 258 linear constraints ($\le 2,000$). At $N = 45$, the model requires $45^2 = 2,025$ variables, and at $N = 64$ requires 4,096 variables; because both exceed the 2,000-variable limit, these dimensions were not evaluated with Gurobi.

\subsection{Reproducibility and Metric Definitions}
Each benchmark run executes inside an isolated Python child process with a strict 30.0-second timeout across 3 repetitions without early termination. Benchmarking runs can be reproduced using:
\begin{quote}
\texttt{python src/benchmark.py --time-limit 30.0 --repetitions 3 --output-dir results}
\end{quote}
Aggregated metrics, comparison tables, and report figures are generated via:
\begin{quote}
\texttt{python src/generate\_report.py}
\end{quote}
Because multithreaded solvers (CP-SAT, Gurobi, CPLEX) utilize host-dependent thread pools, exact execution runtimes depend on host CPU architecture, clock frequencies, and core availability. The benchmark records the following primary metrics:
\begin{itemize}
    \item \texttt{import\_time}: Duration (seconds) to load paradigm and solver modules in the child process prior to model building.
    \item \texttt{build\_time}: Time (seconds) elapsed during variable instantiation and constraint posting.
    \item \texttt{solve\_time}: Active execution duration (seconds) spent inside the solver's solve method.
    \item \texttt{model\_time}: Sum of \texttt{build\_time + solve\_time} per repetition (reported as the median of per-run sums), isolating model construction and solving performance from module import latency.
    \item \texttt{time\_seconds}: Total child process phase duration (\texttt{import\_time + build\_time + solve\_time}), capped at 30.0~s upon timeout.
    \item \texttt{wall\_time}: Parent-measured wall-clock elapsed time (via \texttt{time.perf\_counter()}) spanning child process launch, IPC, and termination.
    \item \texttt{runs\_total}, \texttt{runs\_success}, \texttt{runs\_timeout}, \texttt{runs\_error}: Granular counters tracking outcomes across all 3 repetitions to ensure failure and timeout distributions are captured without masking.
\end{itemize}

\section{Empirical Results and Performance Analysis}

\subsection{SAT Solvers Performance}
Table~\ref{tab:sat} summarizes SAT benchmark results across diverse encodings and solvers (total time in seconds: import + build + solve).

\begin{table*}[htbp]
\centering
\caption{SAT Solvers Performance Metrics (Total Time in Seconds: Import + Build + Solve)} \label{tab:sat}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrr}
\toprule
$N$ & Encoding & CaDiCaL & Glucose3 & MiniSat22 & Kissat & MapleCM \\
\midrule
% {sat_table_tex}
\bottomrule
\tabularnewline
\end{tabular}%
}
\end{table*}

Table~\ref{tab:satsize} presents the model size (variable and clause counts) across board sizes $N$ and encoding variants.

\begin{table}[htbp]
\centering
\caption{SAT Model Size Metrics by Encoding Variant} \label{tab:satsize}
\begin{tabular}{llrr}
\toprule
$N$ & Encoding & Variables & Clauses \\
\midrule
% {sat_size_table_tex}
\bottomrule
\tabularnewline
\end{tabular}
\end{table}

\subsection{CP-SAT Performance}
Table~\ref{tab:cpsat} outlines OR-Tools CP-SAT performance.

\begin{table}[htbp]
\centering
\caption{CP-SAT Performance Metrics} \label{tab:cpsat}
\begin{tabular}{llrrrr}
\toprule
$N$ & Model Variant & Solver & Variables & Constraints & Total Time (s) \\
\midrule
% {cpsat_table_tex}
\bottomrule
\tabularnewline
\end{tabular}
\end{table}

\subsection{Open-Source MIP Performance}
Table~\ref{tab:mip} summarizes CBC and SCIP performance.

\begin{table}[htbp]
\centering
\caption{MIP (CBC/SCIP) Performance Metrics} \label{tab:mip}
\begin{tabular}{llrrrr}
\toprule
$N$ & Model Variant & Solver & Variables & Constraints & Total Time (s) \\
\midrule
% {mip_table_tex}
\bottomrule
\tabularnewline
\end{tabular}
\end{table}

\subsection{Commercial Gurobi and CPLEX Performance}
Tables~\ref{tab:gurobi} and \ref{tab:cplex} present Gurobi and CPLEX results.

\begin{table}[htbp]
\centering
\caption{Gurobi Performance Metrics} \label{tab:gurobi}
\begin{tabular}{llrrrr}
\toprule
$N$ & Model Variant & Solver & Variables & Constraints & Total Time (s) \\
\midrule
% {gurobi_table_tex}
\bottomrule
\tabularnewline
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{CPLEX Performance Metrics} \label{tab:cplex}
\begin{tabular}{llrrrr}
\toprule
$N$ & Model Variant & Solver & Variables & Constraints & Total Time (s) \\
\midrule
% {cplex_table_tex}
\bottomrule
\tabularnewline
\end{tabular}
\end{table}

\section{Timeout Analysis}
Table~\ref{tab:timeouts} details the timeout counts encountered across solver paradigms and board sizes under the strict 30-second limit.

\begin{table}[htbp]
\centering
\caption{Timeout Analysis Across Solvers and Board Sizes} \label{tab:timeouts}
\begin{tabular}{llrr}
\toprule
Paradigm & Solver & Board Size ($N$) & Timeout Count \\
\midrule
% {timeout_table_tex}
\bottomrule
\tabularnewline
\end{tabular}
\end{table}

\section{Visualizations}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{sat_time_cadical.png}
\caption{SAT (CaDiCaL) total time scaling across board sizes $N$.}
\label{fig:sat_time}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{paradigm_comparison.png}
\caption{Comparison of mean model time across all solver paradigms (build + solve; log scale; timeouts counted as 30 s, so means are lower bounds).}
\label{fig:paradigm_time}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{variables_scaling.png}
\caption{Variable count scaling across different solving paradigms and solvers.}
\label{fig:var_scaling}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{build_vs_solve.png}
\caption{Model construction overhead versus active solving time by paradigm.}
\label{fig:build_solve}
\end{figure}

\section{Discussion and Comprehensive Solver Comparison}
The empirical results reveal distinct performance characteristics and limitations across all six solver groups (SAT, CP-SAT, open-source MIP, Gurobi, CPLEX MIP, and CPLEX CP) when scaling across the $N$ range ($N = 8$ to $128$):
\begin{itemize}
    \item \textbf{Comparative Performance Across Six Solver Groups}: Across moderate to large scales ($N = 8$ to $128$), commercial engines (CPLEX MIP, CPLEX CP, Gurobi) and high-performance solvers (CP-SAT, CaDiCaL) demonstrate exceptional scalability. """ + comp_runtime_text + r""" (noting that measurement methodologies differ across solvers, where reported \texttt{time\_seconds} is the median over repetitions of each run's \texttt{import\_time + build\_time + solve\_time} — with build and solve medians reported separately so they may not sum exactly — excluding parent-side process launch overhead; similarly, \texttt{model\_time} represents the median across repetitions of per-run \texttt{build\_time + solve\_time} values, which need not equal \texttt{median(build\_time) + median(solve\_time)}). Crucially, \texttt{import\_time} is library-dependent and constant with respect to $N$ (for instance, OR-Tools CP-SAT incurs $\approx 0.18$~s of Python module import overhead regardless of $N$), meaning cross-paradigm runtime comparisons at small $N$ primarily reflect library import latency rather than algorithmic solving speed. To isolate solver efficiency from module import overhead, Figure~\ref{fig:paradigm_time} (Plot~2) evaluates \emph{model time} (\texttt{build\_time + solve\_time}), with timeouts counted as 30~s, whereas many SAT combinations take tens of seconds or timeout.
    \item \textbf{FEASIBLE vs. OPTIMAL Status Semantics}: For pure satisfaction problems (such as $N$-Queens without an objective function), finding any valid non-attacking configuration successfully solves the problem. Solution status labels are solver-specific and API-dependent: CPLEX CP reports \texttt{FEASIBLE}, whereas CP-SAT reports \texttt{OPTIMAL} and MIP solvers may report \texttt{OPTIMAL} for satisfaction models. Solution validity in our framework is independently verified using the benchmark's verification module rather than relying solely on solver status strings.
    {cp_overhead_text}
    {gurobi_license_text}
    \item \textbf{SAT Scaling and Encodings}: """ + cadical_kissat_text + maplecm_text + sat_timeout_text + r""". Compact cardinality encodings such as Commander (47,096 clauses at $N = 64$) and Sequential Counter (47,692 clauses at $N = 64$) dramatically reduce clause footprints compared to naive Pairwise encodings (428,800 clauses at $N = 64$).
    {cpsat_allowed_text}
\end{itemize}

\section{Conclusion}
This benchmark evaluation demonstrates that while modern SAT, CP-SAT, and commercial CP/MIP engines solve moderate instances efficiently, performance varies significantly at $N = 128$. Performance is heavily dictated by solver sensitivity, global constraint propagation, and encoding efficiency: CaDiCaL, Kissat, CPLEX CP, and CP-SAT handle large scales effectively, whereas clausal blow-up and solver heuristics cause frequent timeouts for older SAT engines.

\section{References}
\begin{thebibliography}{10}
\bibitem{biere2020cadical}
A. Biere, ``CaDiCaL at SAT Race 2019,'' in \emph{Proceedings of SAT Race 2019} (Pragmatics of SAT, Vol. B), pp. 1--2, Department of Computer Science, University of Helsinki, 2019. Available: \url{http://www.satcompetition.org/2019/proceedings.pdf}

\bibitem{sinz2005}
C. Sinz, ``Towards an Optimal CNF Encoding of Boolean Cardinality Constraints,'' In \emph{Principles and Practice of Constraint Programming (CP 2005)}, pp. 827--831, 2005.

\bibitem{klieber2007efficient}
W. Klieber and G. Kwon, ``Efficient CNF Encoding for Selecting 1 from n Objects,'' In \emph{Constraints in Formal Verification (CFV)}, 2007.

\bibitem{ignatiev2018pysat}
A. Ignatiev, A. Morgado, and J. Marques-Silva, ``PySAT: A Python Toolkit for Prototyping with SAT Oracles,'' In \emph{International Conference on Theory and Applications of Satisfiability Testing (SAT)}, pp. 428--437, 2018.

\bibitem{chen2010new}
J. Chen, ``A New SAT Encoding for the At-Most-One Constraint,'' In \emph{International Workshop on Modelling and Reformulation (ModRef)}, 2010.

\bibitem{perron2021ortools}
L. Perron and V. Furnon, ``OR-Tools,'' Google, 2021. Available: \url{https://developers.google.com/optimization}.

\bibitem{biere2021handbook}
A. Biere, M. Heule, H. van Maaren, and T. Walsh, \emph{Handbook of Satisfiability}, 2nd ed. IOS Press, 2021.

\bibitem{gurobi_sizelimit}
Gurobi Optimization, LLC, ``How do I resolve a `Model too large for size-limited Gurobi license' error?'', \emph{Gurobi Help Center}. Available: \url{https://support.gurobi.com/hc/en-us/articles/360051597492} (accessed 16 October 2026).
\end{thebibliography}

\section*{CRediT Authorship Contribution Statement}
\textbf{Nguyễn Khả Đạt}: Conceptualization, Methodology, Software, Validation, Investigation, Data curation, Visualization, Writing - original draft, Writing - review \& editing.

\section*{Declaration of Competing Interest}
The author declares that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Declaration of Generative AI and AI-Assisted Technologies in the Manuscript Preparation}
During the preparation of this work, the author used Claude (Anthropic) for manuscript review and text editing, and Gemini (Google) for code drafting and testing. All experimental evaluations were executed independently by the author on personal hardware, and the author reviewed, verified, and takes full responsibility for the contents of this publication.

\section{Data Availability}
The source code, benchmark runners, raw experimental results (CSV files), and reporting utilities supporting the findings of this study are openly available in the GitHub repository at \url{https://github.com/datnk0116/Nqueen_KHMT}.

\clearpage
\addtocounter{page}{-1}
\end{document}
"""

    if gurobi_df.empty:
        gurobi_license_text = r"""\item \textbf{Gurobi Trial License Limitations}: Gurobi optimization benchmark data is not present in the current dataset. When evaluated under trial licenses, Gurobi restricts model dimensions ($N \le 44$)."""
    else:
        gurobi_license_text = r"""\item \textbf{Gurobi Trial License Limitations}: Gurobi's trial license restricts model dimensions for larger instances, capping trials at $N \le 44$. For $N > 44$, full commercial licenses or alternative solvers (CPLEX, CP-SAT, CaDiCaL) are required."""

    cpsat_allowed_128 = cpsat_df[(cpsat_df['n'] == 128) & (cpsat_df['model_variant'] == 'allowed')] if (not cpsat_df.empty and 'n' in cpsat_df.columns and 'model_variant' in cpsat_df.columns) else pd.DataFrame()
    if not cpsat_allowed_128.empty and 'status' in cpsat_allowed_128.columns and cpsat_allowed_128.iloc[0]['status'] in ('TIMEOUT', 'TIME_LIMIT'):
        cpsat_allowed_text = r"""\item \textbf{CP-SAT Allowed Assignment Timeout at $N = 128$}: The \texttt{allowed} assignment variant posts $\binom{N}{2}$ pairwise binary table constraints whose cumulative allowed tuple count scales as $\mathcal{O}(N^4)$, surging from 7.96~million tuples at $N = 64$ to 130.7~million at $N = 128$. Measured build and solve times at $N = 32$ and $N = 64$ scale by approximately $16\times$ (0.060~s to 0.963~s) and $12\times$ (0.169~s to 2.105~s) per doubling of $N$. Extrapolating these scaling trends from $N = 32$ and $N = 64$ (rather than direct measurement at $N = 128$) projects an estimated build time of $\sim 15.6$~s and solve time of $\sim 26.2$~s (totalling $\sim 41.8$~s, exceeding the 30~s limit), indicating that the timeout is likely driven by table constraint size rather than search difficulty."""
    else:
        cpsat_allowed_text = ""

    has_reported = stats.get('has_solver_reported_time', False)
    reported_s = stats.get('cplex_cp_128_reported_s', float('nan'))
    if cplex_cp_df.empty:
        cp_overhead_text = r"""\item \textbf{CP Optimizer Process Startup Overhead}: CP Optimizer experimental data is not available in the current dataset. When CP Optimizer is evaluated, spawning its external binary executable (\texttt{cpoptimizer}) incurs inter-process communication and process startup overhead which should be distinguished from native API solvers."""
    elif has_reported and not pd.isna(reported_s):
        cp_overhead_text = r"""\item \textbf{CP Optimizer Process Startup Overhead}: CP Optimizer relies on an external binary executable (\texttt{cpoptimizer}). Spawning this subprocess incurs inter-process communication and process startup overhead; actual total measured runtimes for CPLEX CP increase from approximately """ + cplex_cp_8_str + r""" at $N = 8$ to """ + cplex_cp_128_str + r""" at $N = 128$. Here, \texttt{solve\_time} measures the wall-clock duration around the solve call (including cpoptimizer process startup), while \texttt{solver\_reported\_time} is the number reported by the solver (median of """ + format_adaptive_time(reported_s) + r""" at $N = 128$); thus, measured runtimes include process startup and inter-process communication, and should not be directly compared on equal footing with CP-SAT's wall-clock measurement."""
    else:
        cpsat_comp = f"~{cpsat_int_128_str} s" if cpsat_int_128_str != "N/A" else "N/A"
        build_comp = f"~{cplex_cp_128_build_str} ms" if cplex_cp_128_build_str != "N/A" else "N/A"
        cp_overhead_text = r"""\item \textbf{CP Optimizer Process Startup Overhead}: CP Optimizer relies on an external binary executable (\texttt{cpoptimizer}). Spawning this subprocess incurs inter-process communication and process startup overhead; actual total measured runtimes for CPLEX CP increase from approximately """ + cplex_cp_8_str + r""" at $N = 8$ to """ + cplex_cp_128_str + r""" at $N = 128$. Note that solver-reported solve time via \texttt{get\_solve\_time()} has a reported resolution of approximately 10 ms, whereas \texttt{solve\_time} measures the wall-clock duration around the solve call (including \texttt{cpoptimizer} process startup and IPC); thus, the ~""" + cplex_cp_128_str + r""" recorded at $N = 128$ includes external process startup plus """ + build_comp + r""" model build time, and should not be directly compared on equal footing with CP-SAT's """ + cpsat_comp + r""" wall-clock measurement."""
    latex_content = latex_content.replace("% {sat_table_tex}", sat_table_tex)
    latex_content = latex_content.replace("% {sat_size_table_tex}", sat_size_table_tex)
    latex_content = latex_content.replace("% {cpsat_table_tex}", cpsat_table_tex)
    latex_content = latex_content.replace("% {mip_table_tex}", mip_table_tex)
    latex_content = latex_content.replace("% {gurobi_table_tex}", gurobi_table_tex)
    latex_content = latex_content.replace("% {cplex_table_tex}", cplex_table_tex)
    latex_content = latex_content.replace("% {timeout_table_tex}", timeout_table_tex)
    latex_content = latex_content.replace("{cp_overhead_text}", cp_overhead_text)
    latex_content = latex_content.replace("{gurobi_license_text}", gurobi_license_text)
    latex_content = latex_content.replace("{cpsat_allowed_text}", cpsat_allowed_text)

    latex_path = os.path.join(out_dir, "report.tex")
    with open(latex_path, "w") as f:
        f.write(latex_content)
    print(f"LaTeX report generated successfully at {latex_path}")
    return latex_path

def compile_pdf(out_dir="report"):
    import shutil
    if shutil.which("pdflatex") is None:
        print("Warning: pdflatex not found in PATH. Skipping PDF compilation.")
        return False
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], cwd=out_dir, check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], cwd=out_dir, check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], cwd=out_dir, check=True)
        pdf_path = os.path.join(out_dir, "report.pdf")
        if os.path.exists(pdf_path):
            print(f"Successfully compiled {pdf_path}!")
            return True
        else:
            raise RuntimeError("report.pdf was not generated.")
    except Exception as e:
        print(f"Error during pdflatex compilation: {e}")
        raise

def main():
    generate_tex(results_dir="results", out_dir="report")
    compile_pdf(out_dir="report")

if __name__ == "__main__":
    main()
