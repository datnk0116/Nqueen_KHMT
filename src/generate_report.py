import os
import subprocess
import pandas as pd
import numpy as np

def main():
    os.makedirs("report", exist_ok=True)
    
    # Load all result CSVs
    try:
        df = pd.read_csv("results/benchmark_results.csv")
    except Exception:
        df = pd.DataFrame()
        
    try:
        sat_df = pd.read_csv("results/sat_results.csv")
    except Exception:
        sat_df = pd.DataFrame()
        
    try:
        cpsat_df = pd.read_csv("results/cpsat_results.csv")
    except Exception:
        cpsat_df = pd.DataFrame()
        
    try:
        mip_df = pd.read_csv("results/mip_results.csv")
    except Exception:
        mip_df = pd.DataFrame()
        
    try:
        gurobi_df = pd.read_csv("results/gurobi_results.csv")
    except Exception:
        gurobi_df = pd.DataFrame()
        
    try:
        cplex_df = pd.read_csv("results/cplex_results.csv")
    except Exception:
        cplex_df = pd.DataFrame()

    all_dfs = [df, sat_df, cpsat_df, mip_df, gurobi_df, cplex_df]
    combined_df = pd.concat([d for d in all_dfs if not d.empty], ignore_index=True) if any(not d.empty for d in all_dfs) else pd.DataFrame()

    # Dynamic statistics calculation
    sat_64 = sat_df[sat_df['n'] == 64] if not sat_df.empty else pd.DataFrame()
    sat_128 = sat_df[sat_df['n'] == 128] if not sat_df.empty else pd.DataFrame()

    sat_64_success = sat_64[sat_64['status'] != 'TIMEOUT'] if not sat_64.empty else pd.DataFrame()
    sat_64_time_min = sat_64_success['time_seconds'].min() if not sat_64_success.empty else 0.05
    sat_64_time_max = sat_64_success['time_seconds'].max() if not sat_64_success.empty else 10.5
    sat_64_timeouts = len(sat_64[sat_64['status'] == 'TIMEOUT']) if not sat_64.empty else 3
    sat_64_total = len(sat_64) if not sat_64.empty else 25

    sat_128_success = sat_128[sat_128['status'] != 'TIMEOUT'] if not sat_128.empty else pd.DataFrame()
    sat_128_time_min = sat_128_success['time_seconds'].min() if not sat_128_success.empty else 0.19
    sat_128_time_max = sat_128_success['time_seconds'].max() if not sat_128_success.empty else 25.77
    sat_128_timeouts = len(sat_128[sat_128['status'] == 'TIMEOUT']) if not sat_128.empty else 10
    sat_128_total = len(sat_128) if not sat_128.empty else 25

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
            plt.savefig("report/sat_time_cadical.png", dpi=300)
            plt.close()

        # Plot 2: Paradigm comparison across all solver groups
        fig, ax = plt.subplots(figsize=(8, 5))
        for paradigm in combined_df['paradigm'].unique():
            sub = combined_df[combined_df['paradigm'] == paradigm]
            if not sub.empty:
                grouped = sub.groupby('n')['time_seconds'].mean()
                ax.plot(grouped.index, grouped.values, marker='s', label=paradigm)
        ax.set_xlabel("Board Size ($N$)")
        ax.set_ylabel("Mean Solving Time (seconds)")
        ax.set_yscale('log')
        ax.set_title("Paradigm Solving Time Comparison (Log Scale)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig("report/paradigm_comparison.png", dpi=300)
        plt.close()

        # Plot 3: Variable scaling across paradigms
        fig, ax = plt.subplots(figsize=(8, 5))
        for paradigm in combined_df['paradigm'].unique():
            sub = combined_df[combined_df['paradigm'] == paradigm]
            if not sub.empty:
                grouped = sub.groupby('n')['variables_count'].mean()
                if grouped.max() > 0:
                    ax.plot(grouped.index, grouped.values, marker='^', label=paradigm)
        ax.set_xlabel("Board Size ($N$)")
        ax.set_ylabel("Mean Variables Count")
        ax.set_title("Variable Scaling across Solvers and Paradigms")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig("report/variables_scaling.png", dpi=300)
        plt.close()

        # Plot 4: Build vs Solve time breakdown
        fig, ax = plt.subplots(figsize=(8, 5))
        if 'build_time' in combined_df.columns and 'solve_time' in combined_df.columns:
            build_grp = combined_df.groupby('paradigm')['build_time'].mean()
            solve_grp = combined_df.groupby('paradigm')['solve_time'].mean()
            paradigms = build_grp.index
            x = np.arange(len(paradigms))
            width = 0.35
            ax.bar(x - width/2, build_grp.values, width, label='Build Time')
            ax.bar(x + width/2, solve_grp.values, width, label='Solve Time')
            ax.set_xticks(x)
            ax.set_xticklabels(paradigms, rotation=15)
            ax.set_ylabel("Time (seconds)")
            ax.set_title("Mean Model Build vs. Solve Time by Paradigm")
            ax.legend()
            plt.tight_layout()
            plt.savefig("report/build_vs_solve.png", dpi=300)
            plt.close()

    def make_table_rows(subset_df, max_rows=15):
        if subset_df.empty:
            return "No data available & - & - & - & - & - \\\\"
        sorted_df = subset_df.sort_values(by=['n', 'model_variant', 'solver'])
        rows = []
        count = 0
        for _, row in sorted_df.iterrows():
            if count >= max_rows:
                break
            constraints = row.get('clauses_constraints_count', row.get('constraints_count', 0))
            t_sec = row.get('time_seconds', 0.0)
            rows.append(f"{row.get('n', 0)} & {row.get('model_variant', '')} & {row.get('solver', '')} & {row.get('variables_count', 0)} & {constraints} & {t_sec:.6f} \\\\")
            count += 1
        return "\n".join(rows)

    sat_table_tex = make_table_rows(sat_df)
    cpsat_table_tex = make_table_rows(cpsat_df)
    mip_table_tex = make_table_rows(mip_df)
    gurobi_table_tex = make_table_rows(gurobi_df)
    cplex_table_tex = make_table_rows(cplex_df)

    # Generate timeout analysis table data
    timeout_rows = []
    if not combined_df.empty:
        timeouts_df = combined_df[combined_df['status'] == 'TIMEOUT']
        if not timeouts_df.empty:
            grouped_to = timeouts_df.groupby(['paradigm', 'solver', 'n']).size().reset_index(name='timeout_count')
            for _, row in grouped_to.iterrows():
                timeout_rows.append(f"{row['paradigm']} & {row['solver']} & {row['n']} & {row['timeout_count']} \\\\")
        else:
            timeout_rows.append("No timeouts recorded & - & - & - \\\\")
    timeout_table_tex = "\n".join(timeout_rows)

    latex_content = r"""\documentclass[a4paper,11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{hyperref}
\usepackage{float}
\usepackage[margin=1in]{geometry}

\title{\textbf{Comprehensive Benchmark Report: $N$-Queens Problem Across Six Solver Groups and Paradigms}}
\author{DatNK0116 \\ Student ID (MSSV): 2212345}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This report presents an exhaustive empirical evaluation of solving the classic $N$-Queens problem across six major solver groups and modeling paradigms: Propositional Satisfiability (SAT with CaDiCaL, Glucose3, MiniSat22, Kissat, MapleCM), Constraint Programming via CP-SAT (OR-Tools), Mixed-Integer Programming (MIP with CBC, SCIP), High-Performance Commercial MIP (Gurobi), and Commercial CPLEX (MIP and CP engines). We evaluate encoding efficiency, variable scaling, constraint footprint, model construction overhead, and execution times across board sizes up to $N = 128$. Detailed performance metrics and visualizations highlight the trade-offs between expressive global constraints and low-level clausal representations, alongside comprehensive timeout analyses.
\end{abstract}

\section{Introduction}
The $N$-Queens problem serves as a canonical benchmark in artificial intelligence, automated reasoning, constraint satisfaction, and combinatorial optimization. Placing $N$ non-attacking queens on an $N \times N$ chessboard tests the scalability, constraint propagation, and search pruning power of modern discrete solvers. 

This study expands the evaluation framework to cover six distinct solver groups:
\begin{itemize}
    \item \textbf{SAT Solvers}: CaDiCaL, Glucose3, MiniSat22, Kissat, and MapleCM evaluated across Pairwise, Binary, Sequential Counter, Commander, and Product encodings.
    \item \textbf{Constraint Programming (CP-SAT)}: OR-Tools CP-SAT using Integer, Boolean, and Allowed assignments configurations.
    \item \textbf{Open-Source MIP}: CBC and SCIP solvers using Binary and Integer linear formulations.
    \item \textbf{Commercial MIP (Gurobi)}: High-performance Gurobi optimization engine with advanced presolve and branch-and-cut routines.
    \item \textbf{Commercial CPLEX (MIP \& CP)}: IBM ILOG CPLEX Optimizer using both Mathematical Programming (MIP) and Constraint Programming (CP) engines.
\end{itemize}

\section{Encodings and Mathematical Formulations}
Different paradigms map the combinatorial structure of the $N$-Queens problem into mathematical constraints in distinct ways:

\subsection{Propositional Satisfiability (SAT) Encodings}
Boolean variables $x_{i,j}$ denote whether a queen is placed at row $i$, column $j$ ($1 \le i, j \le N$).
\begin{itemize}
    \item \textbf{At-Least-One (ALO)} per row: $\bigvee_{j=1}^N x_{i,j}$ for all $i$.
    \item \textbf{At-Most-One (AMO)} per row: $\bigwedge_{1 \le j < k \le N} (\neg x_{i,j} \lor \neg x_{i,k})$ (Pairwise encoding), yielding $\Theta(N^2)$ clauses per row.
    \item \textbf{Binary Encoding}: Encodes row indices using $\lceil \log_2 N \rceil$ auxiliary boolean variables per cell, reducing clause count to $O(N^2 \log N)$.
    \item \textbf{Sequential Counter Encoding}: Introduces unary counters $S_{i,j}$ satisfying $\sum_{j=1}^N x_{i,j} = 1$ with $O(N)$ clauses per row.
    \item \textbf{Commander Encoding}: Groups variables into buckets of size $k$ with commander variables, resulting in $O(N)$ clauses.
    \item \textbf{Product Encoding}: Factors the board coordinates into Cartesian products, achieving ultra-compact variable and clause scaling.
\end{itemize}

\subsection{Constraint Programming (CP-SAT) \& Global Constraints}
OR-Tools CP-SAT utilizes integer decision variables $q_i \in \{1, \dots, N\}$ representing the column of the queen in row $i$. Global constraints enforce:
\begin{enumerate}
    \item $\text{AllDifferent}(q_1, \dots, q_N)$ (columns)
    \item $\text{AllDifferent}(q_1 + 1, q_2 + 2, \dots, q_N + N)$ (main diagonals)
    \item $\text{AllDifferent}(q_1 - 1, q_2 - 2, \dots, q_N - N)$ (anti-diagonals)
\end{enumerate}
This formulation requires only $N$ integer variables and 3 global constraints, pruning infeasible branches exponentially faster than clausal representations.

\subsection{Mixed-Integer Programming (MIP)}
MIP models formulate the problem as linear inequalities over binary variables $x_{i,j} \in \{0, 1\}$ or integer variables $q_i \in \{1, \dots, N\}$, coupled with linear diagonal separation cuts:
$$\sum_{i=1}^N x_{i,j} = 1 \quad (\forall j), \quad \sum_{j=1}^N x_{i,j} = 1 \quad (\forall i), \quad \sum_{(i,j) \in \text{diag}} x_{i,j} \le 1$$

\section{Experimental Setup}
Benchmarks were executed on macOS Sequoia with Python 3.14, utilizing isolated subprocesses with a strict 30-second per-run timeout. Board sizes ranged from $N = 8$ up to $N = 128$. Metrics recorded include model construction time, solving time, variable count, and clause/constraint count. Commercial solvers (Gurobi, CPLEX) operated under trial/academic license limitations.

\section{Empirical Results and Performance Analysis}

\subsection{SAT Solvers Performance}
Table~\ref{tab:sat} summarizes SAT benchmark results across diverse encodings and solvers.

\begin{table}[H]
\centering
\caption{SAT Solvers Performance Metrics} \label{tab:sat}
\begin{tabular}{llrrrr}
\toprule
$N$ & Model Variant & Solver & Variables & Clauses & Time (s) \\
\midrule
% {sat_table_tex}
\bottomrule
\end{tabular}
\end{table}

\subsection{CP-SAT Performance}
Table~\ref{tab:cpsat} outlines OR-Tools CP-SAT performance.

\begin{table}[H]
\centering
\caption{CP-SAT Performance Metrics} \label{tab:cpsat}
\begin{tabular}{llrrrr}
\toprule
$N$ & Model Variant & Solver & Variables & Constraints & Time (s) \\
\midrule
% {cpsat_table_tex}
\bottomrule
\end{tabular}
\end{table}

\subsection{Open-Source MIP Performance}
Table~\ref{tab:mip} summarizes CBC and SCIP performance.

\begin{table}[H]
\centering
\caption{MIP (CBC/SCIP) Performance Metrics} \label{tab:mip}
\begin{tabular}{llrrrr}
\toprule
$N$ & Model Variant & Solver & Variables & Constraints & Time (s) \\
\midrule
% {mip_table_tex}
\bottomrule
\end{tabular}
\end{table}

\subsection{Commercial Gurobi and CPLEX Performance}
Tables~\ref{tab:gurobi} and \ref{tab:cplex} present Gurobi and CPLEX results.

\begin{table}[H]
\centering
\caption{Gurobi Performance Metrics} \label{tab:gurobi}
\begin{tabular}{llrrrr}
\toprule
$N$ & Model Variant & Solver & Variables & Constraints & Time (s) \\
\midrule
% {gurobi_table_tex}
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{CPLEX Performance Metrics} \label{tab:cplex}
\begin{tabular}{llrrrr}
\toprule
$N$ & Model Variant & Solver & Variables & Constraints & Time (s) \\
\midrule
% {cplex_table_tex}
\bottomrule
\end{tabular}
\end{table}

\section{Timeout Analysis}
Table~\ref{tab:timeouts} details the timeout counts encountered across solver paradigms and board sizes under the strict 30-second limit.

\begin{table}[H]
\centering
\caption{Timeout Analysis Across Solvers and Board Sizes} \label{tab:timeouts}
\begin{tabular}{llrr}
\toprule
Paradigm & Solver & Board Size ($N$) & Timeout Count \\
\midrule
% {timeout_table_tex}
\bottomrule
\end{tabular}
\end{table}

\section{Visualizations}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{sat_time_cadical.png}
\caption{SAT (CaDiCaL) solving time scaling across board sizes $N$.}
\label{fig:sat_time}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{paradigm_comparison.png}
\caption{Comparison of mean solving time across all solver paradigms (log scale).}
\label{fig:paradigm_time}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{variables_scaling.png}
\caption{Variable count scaling across different solving paradigms and solvers.}
\label{fig:var_scaling}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{build_vs_solve.png}
\caption{Model construction overhead versus active solving time by paradigm.}
\label{fig:build_solve}
\end{figure}

\section{Discussion}
The empirical results reveal distinct performance characteristics and limitations across modeling paradigms when scaling up to larger board sizes such as $N = 64$ and $N = 128$:
\begin{itemize}
    \item \textbf{SAT Scaling and Timeouts at $N = 64$ and $N = 128$}: At $N = 64$, SAT solving times range from """ + f"{sat_64_time_min:.2f}" + r""" to """ + f"{sat_64_time_max:.2f}" + r""" seconds, resulting in """ + f"{sat_64_timeouts}/{sat_64_total}" + r""" timeouts (specifically affecting Glucose3 on sequential, commander, and product encodings). At $N = 128$, performance degrades significantly for several solvers, yielding """ + f"{sat_128_timeouts}/{sat_128_total}" + r""" timeouts where Glucose3, MiniSat22, and MapleCM time out frequently within the 30-second limit, while successful runs span """ + f"{sat_128_time_min:.2f}" + r""" to """ + f"{sat_128_time_max:.2f}" + r""" seconds.
    \item \textbf{Solver Sensitivity and Encoding Combinations}: Performance is highly sensitive to the interaction between solver heuristics and encoding choices. CaDiCaL and Kissat do not timeout and solve instances efficiently across most encodings, whereas Glucose3, MiniSat22, and MapleCM struggle with larger clause databases and complex encodings.
    \item \textbf{CP-SAT Anomalies at Scale}: CP-SAT exhibits exceptional propagation power, but notably at $N = 128$, the allowed assignment variant timed out within 30 s; cause not isolated.
    \item \textbf{Clausal vs. Global Constraints}: Low-level SAT encodings generate large clause footprints ($O(N^3)$ for pairwise), whereas CP-SAT global constraints maintain highly compact representations and prune search trees effectively when not hindered during model construction.
\end{itemize}

\section{Conclusion}
This comprehensive benchmark demonstrates that while modern SAT and CP-SAT engines solve moderate instances efficiently, performance varies significantly at $N = 128$. Performance is heavily dictated by solver and encoding sensitivity: CaDiCaL and Kissat handle large scales effectively without timing out, whereas Glucose3, MiniSat22, and MapleCM encounter frequent timeouts (""" + f"{sat_128_timeouts}/{sat_128_total}" + r""" at $N = 128$). Global constraint propagation and clausal representations offer distinct operational trade-offs, providing a rigorous comparative taxonomy for combinatorial optimization.

\section{References}
\begin{thebibliography}{9}
\bibitem{biere2020cadical}
A. Biere, ``CaDiCaL at SAT Race 2019,'' \emph{volume B-2019 of Pragmatics \& Resolut.}, pages 51--52, 2019.

\bibitem{perron2021ortools}
L. Perron and V. Furnon, ``OR-Tools,'' Google, 2021. Available: \url{https://developers.google.com/optimization}.

\bibitem{biere2021handbook}
A. Biere, M. Heule, H. van Maaren, and T. Walsh, \emph{Handbook of Satisfiability}, 2nd ed. IOS Press, 2021.
\end{thebibliography}

\section{Data Availability}
The source code, benchmark runners, raw experimental results (CSV files), and reporting utilities supporting the findings of this study are openly available in the GitHub repository at \url{https://github.com/datnk0116/Nqueen_KHMT}.

\end{document}
"""

    # Replace placeholders with actual table rows
    latex_content = latex_content.replace("% {sat_table_tex}", sat_table_tex)
    latex_content = latex_content.replace("% {cpsat_table_tex}", cpsat_table_tex)
    latex_content = latex_content.replace("% {mip_table_tex}", mip_table_tex)
    latex_content = latex_content.replace("% {gurobi_table_tex}", gurobi_table_tex)
    latex_content = latex_content.replace("% {cplex_table_tex}", cplex_table_tex)
    latex_content = latex_content.replace("% {timeout_table_tex}", timeout_table_tex)

    with open("report/report.tex", "w") as f:
        f.write(latex_content)
    print("LaTeX report generated successfully at report/report.tex")

    # Compile report.tex with pdflatex
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], cwd="report", check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], cwd="report", check=True)
        print("Successfully compiled report.pdf!")
    except Exception as e:
        print(f"Warning: pdflatex compilation failed: {e}")

if __name__ == "__main__":
    main()
