\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cas-sc}[2026/01/01 Elsevier CAS Single Column Class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions*\relax
\LoadClass[11pt,a4paper]{article}

\RequirePackage{geometry}
\geometry{a4paper, margin=1in}
\RequirePackage{fancyhdr}
\RequirePackage{etoolbox}

% Elsevier CAS compatible command definitions
\newcommand{\affiliation}[2][]{%
  \par\vspace{0.5em}\noindent\textbf{Affiliation #1:} #2\par
}
\newcommand{\correspondingauthor}[1]{#1}
\newcommand{\fnmark}[1]{$#1$}
\newcommand{\tnotemark}[1]{$#1$}
\newcommand{\credit}[1]{#1}
\newcommand{\printorcid}{}
\newcommand{\printcredits}{}
\newcommand{\ead}[1]{(#1)}

\newenvironment{frontmatter}{}{}
\newenvironment{highlights}{}{}
\newenvironment{keyhighlights}{}{}
\newenvironment{keywords}{%
  \vspace{1em}\noindent\textbf{Keywords: }%
}{%
  \par\vspace{1em}%
}
\newcommand{\sep}{, }

\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\small\thepage}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.0pt}
\renewcommand{\footrulewidth}{0.0pt}
