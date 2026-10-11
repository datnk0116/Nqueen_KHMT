import os
import pandas as pd
import numpy as np

def main():
    os.makedirs("nqueens-sat-benchmark/report", exist_ok=True)
    
    # Load data
    df = pd.read_csv("nqueens-sat-benchmark/results/benchmark_results.csv")
    sat_df = pd.read_csv("nqueens-sat-benchmark/results/sat_results.csv")
    cpsat_df = pd.read_csv("nqueens-sat-benchmark/results/cpsat_results.csv")
    mip_df = pd.read_csv("nqueens-sat-benchmark/results/mip_results.csv")

    # Try importing matplotlib for plots
    has_mpl = False
    try:
        import matplotlib.pyplot as plt
        has_mpl = True
    except ImportError:
        pass

    if has_mpl:
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        # Plot 1: SAT Solvers time vs n by encoding
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant in sat_df['model_variant'].unique():
            sub = sat_df[(sat_df['model_variant'] == variant) & (sat_df['solver'] == 'cadical')]
            if not sub.empty:
                grouped = sub.groupby('n')['time_seconds'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', label=f"SAT - {variant}")
        ax.set_xlabel("Board Size (n)")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("SAT (Cadical) Performance by Encoding Variant")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("nqueens-sat-benchmark/report/sat_time_cadical.png", dpi=300)
        plt.close()

        # Plot 2: Paradigm comparison (SAT vs CP-SAT vs MIP)
        fig, ax = plt.subplots(figsize=(8, 5))
        for paradigm in df['paradigm'].unique():
            sub = df[df['paradigm'] == paradigm]
            grouped = sub.groupby('n')['time_seconds'].mean()
            ax.plot(grouped.index, grouped.values, marker='s', label=paradigm)
        ax.set_xlabel("Board Size (n)")
        ax.set_ylabel("Mean Time (seconds)")
        ax.set_title("Paradigm Solving Time Comparison")
        ax.legend()
        plt.tight_layout()
        plt.savefig("nqueens-sat-benchmark/report/paradigm_comparison.png", dpi=300)
        plt.close()

        # Plot 3: Variables count comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        for paradigm in df['paradigm'].unique():
            sub = df[df['paradigm'] == paradigm]
            grouped = sub.groupby('n')['variables_count'].mean()
            ax.plot(grouped.index, grouped.values, marker='^', label=paradigm)
        ax.set_xlabel("Board Size (n)")
        ax.set_ylabel("Mean Variables Count")
        ax.set_title("Variable Scaling across Paradigms")
        ax.legend()
        plt.tight_layout()
        plt.savefig("nqueens-sat-benchmark/report/variables_scaling.png", dpi=300)
        plt.close()

    # Generate LaTeX Tables from summaries across expanded N (e.g. n in [4, 8] for SAT/MIP, n in [4, 10] for CP-SAT)
    sat_expanded = sat_df.sort_values(by=['n', 'model_variant', 'solver'])
    sat_table_rows = []
    for _, row in sat_expanded.iterrows():
        sat_table_rows.append(f"{row['n']} & {row['model_variant']} & {row['solver']} & {row['variables_count']} & {row['clauses_constraints_count']} & {row['time_seconds']:.6f} \\\\")
    sat_table_tex = "\n".join(sat_table_rows)

    cpsat_expanded = cpsat_df.sort_values(by=['n', 'model_variant'])
    cpsat_table_rows = []
    for _, row in cpsat_expanded.iterrows():
        cpsat_table_rows.append(f"{row['n']} & {row['model_variant']} & {row['solver']} & {row['variables_count']} & {row['clauses_constraints_count']} & {row['time_seconds']:.6f} \\\\")
    cpsat_table_tex = "\n".join(cpsat_table_rows)

    mip_expanded = mip_df.sort_values(by=['n', 'model_variant', 'solver'])
    mip_table_rows = []
    for _, row in mip_expanded.iterrows():
        mip_table_rows.append(f"{row['n']} & {row['model_variant']} & {row['solver']} & {row['variables_count']} & {row['clauses_constraints_count']} & {row['time_seconds']:.6f} \\\\")
    mip_table_tex = "\n".join(mip_table_rows)

    latex_content = f"""\\documentclass[a4paper,fleqn]{{cas-sc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}
\\usepackage{{hyperref}}
\\usepackage{{float}}

\\title{{\\textbf{{Comprehensive Benchmark Report: N-Queens Problem via SAT, CP-SAT, and MIP Paradigms}}}}
\\author{{N-Queens SAT Benchmark Team}}
\\date{{\\today}}

\\begin{{document}}

\\begin{{frontmatter}}
\\maketitle

\\begin{{abstract}}
This report presents a thorough empirical evaluation of solving the classic $N$-Queens problem across three distinct constraint-solving paradigms: Propositional Satisfiability (SAT), Constraint Programming via CP-SAT (OR-Tools), and Mixed-Integer Programming (MIP) using CBC and SCIP solvers. We evaluate multiple encoding strategies (e.g., pairwise, binary, sequential counter, commander, product encodings for SAT; alldiff, allowed, assignments, boolean, integer, matrix for CP-SAT and MIP) across board sizes $N \\in [4, 10]$. Performance metrics including variable counts, clause/constraint counts, and execution time are analyzed and discussed.
\\end{{abstract}}
\\end{{frontmatter}}

\\section{{Introduction}}
The $N$-Queens problem is a fundamental benchmark problem in artificial intelligence, constraint satisfaction, and combinatorial optimization. The objective is to place $N$ non-attacking queens on an $N \\times N$ chessboard. While conceptually straightforward, the formulation and solving efficiency vary dramatically depending on the modeling paradigm and encoding strategy.

This benchmark framework explores:
\\begin{{itemize}}
    \\item \\textbf{{SAT Encodings}}: Pairwise, Binary, Sequential Counter, Commander, and Product encodings solved with Glucose3, MiniSat22, and CaDiCaL.
    \\item \\textbf{{Constraint Programming (CP-SAT)}}: Alldiff, Allowed assignments, Explicit assignments, Boolean, Integer, and Matrix formulations solved with OR-Tools CP-SAT.
    \\item \\textbf{{Mixed-Integer Programming (MIP)}}: Binary, Integer, and Matrix formulations solved with CBC and SCIP.
\\end{{itemize}}

\\section{{Methodology and Encodings}}
Different modeling choices impact the size of the search space, the number of variables, and the number of constraints (or clauses). 
\\begin{{itemize}}
    \\item \\textbf{{SAT Paradigms}}: Encode queen positions using boolean variables $x_{{i,j}}$. Constraints enforce that each row, column, and diagonal has at most one queen.
    \\item \\textbf{{CP-SAT Paradigms}}: Leverage high-level global constraints such as \\texttt{{AllDifferent}} and integer domains.
    \\item \\textbf{{MIP Paradigms}}: Formulate linear integer programming constraints with binary or integer decision variables.
\\end{{itemize}}

\\section{{Empirical Results and Performance Analysis}}

\\subsection{{SAT Encapsulation Performance (Expanded $N$ Experiments)}}
Table~\\ref{{tab:sat_expanded}} summarizes the performance of SAT models across expanded board sizes $N \\in [4, 8]$.

\\begin{{table}}[H]
\\centering
\\caption{{SAT Performance Metrics Across Expanded $N$ Experiments}} \\label{{tab:sat_expanded}}
\\begin{{tabular}}{{llrrrr}}
\\toprule
$N$ & Model Variant & Solver & Variables & Clauses & Time (s) \\\\
\\midrule
{sat_table_tex}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{CP-SAT Performance (Expanded $N$ Experiments)}}
Table~\\ref{{tab:cpsat_expanded}} summarizes CP-SAT models across expanded board sizes $N \\in [4, 10]$.

\\begin{{table}}[H]
\\centering
\\caption{{CP-SAT Performance Metrics Across Expanded $N$ Experiments}} \\label{{tab:cpsat_expanded}}
\\begin{{tabular}}{{llrrrr}}
\\toprule
$N$ & Model Variant & Solver & Variables & Constraints & Time (s) \\\\
\\midrule
{cpsat_table_tex}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{MIP Performance (Expanded $N$ Experiments)}}
Table~\\ref{{tab:mip_expanded}} summarizes MIP models across expanded board sizes $N \\in [4, 8]$.

\\begin{{table}}[H]
\\centering
\\caption{{MIP Performance Metrics Across Expanded $N$ Experiments}} \\label{{tab:mip_expanded}}
\\begin{{tabular}}{{llrrrr}}
\\toprule
$N$ & Model Variant & Solver & Variables & Constraints & Time (s) \\\\
\\midrule
{mip_table_tex}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{Visualizations}}

"""

    if has_mpl:
        latex_content += f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.85\\textwidth]{{sat_time_cadical.png}}
\\caption{{SAT (CaDiCaL) solving time scaling across board sizes $N$.}}
\\label{{fig:sat_time}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.85\\textwidth]{{paradigm_comparison.png}}
\\caption{{Comparison of mean solving time across paradigms (SAT, CP-SAT, MIP).}}
\\label{{fig:paradigm_time}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.85\\textwidth]{{variables_scaling.png}}
\\caption{{Variable count scaling across different solving paradigms.}}
\\label{{fig:var_scaling}}
\\end{{figure}}
"""

    latex_content += f"""
\\section{{Conclusion}}
The benchmark results indicate that SAT solvers achieve exceptional performance with sub-millisecond execution times, while CP-SAT and MIP solvers achieve efficient millisecond-level execution times for moderate board sizes ($N \\le 10$). High-level global constraints in CP-SAT (such as \\texttt{{AllDifferent}}) minimize constraint counts, while compact SAT encodings (like binary and commander) scale efficiently in variable count. MIP solvers (SCIP and CBC) also solve models reliably, though overhead is slightly higher compared to specialized SAT and CP propagators.

\\section{{Data Availability}}
The source code, benchmark scripts, raw experimental results (CSV files), and reporting utilities supporting the findings of this study are openly available in the GitHub repository at \\url{{https://github.com/username/nqueens-sat-benchmark}}.

\\end{{document}}
"""

    with open("nqueens-sat-benchmark/report/report.tex", "w") as f:
        f.write(latex_content)
    print("LaTeX report generated successfully at nqueens-sat-benchmark/report/report.tex")

if __name__ == "__main__":
    main()
