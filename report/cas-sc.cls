\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cas-sc}[2026/01/01 Elsevier CAS Single Column Class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions*\relax
\LoadClass[11pt]{article}

\RequirePackage{geometry}
\geometry{a4paper, margin=1in}
\RequirePackage{fancyhdr}
\RequirePackage{etoolbox}

% Define Elsevier CAS specific commands and environments
\newcommand{\affiliation}[2][]{}
\newcommand{\correspondingauthor}[1]{}
\newcommand{\fnmark}[1]{}
\newcommand{\tnotemark}[1]{}
\newcommand{\credit}[1]{}
\newcommand{\printorcid}{}
\newcommand{\printcredits}{}

\newenvironment{frontmatter}{}{}
\newenvironment{highlights}{}{}
\newenvironment{keyhighlights}{}{}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Elsevier CAS -- Single Column Layout}
\fancyhead[R]{\small\thepage}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
