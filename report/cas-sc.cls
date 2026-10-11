\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cas-sc}[2023/01/01 Elsevier CAS Single Column Class Mock]

\LoadClass[11pt,a4paper]{article}

\RequirePackage{geometry}
\geometry{margin=1in}

% Define Elsevier CAS specific commands to prevent undefined control sequence errors
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
