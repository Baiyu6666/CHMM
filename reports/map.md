\documentclass[lettersize,journal]{IEEEtran}
% \usepackage[UTF8,fontset=windows]{ctex}
\usepackage{booktabs}
\usepackage{dsfont}
\usepackage{graphicx}
\usepackage[caption=false,font=footnotesize]{subfig}\usepackage{array}
\usepackage{textcomp}
\usepackage{verbatim}
\hyphenation{op-tical net-works semi-conduc-tor IEEE-Xplore}
\newenvironment{redsection}
  {\begingroup\color{red}
   \captionsetup{labelfont={color=red}, textfont={color=red}}}
  {\endgroup}
% \usepackage[utf8]{inputenc}
% \usepackage[T1]{fontenc}
% \usepackage{aecompl}
\usepackage{dsfont}
\usepackage{epstopdf}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts, amsthm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
\usepackage{algorithm}
\usepackage{xcolor}
\usepackage{float}
\usepackage[table]{xcolor}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\newcommand{\NA}{\cellcolor{gray!18}}
\newcommand{\NoAct}{--}
% \usepackage{picinpar}
% \usepackage{babel}
\usepackage{url}
% \usepackage[latin1]{inputenc}
\usepackage{multirow}
\usepackage{color}
\usepackage{enumerate}
\usepackage{siunitx}
% \usepackage{breakurl}
\usepackage{epstopdf}
\usepackage{pbox}
\usepackage{stfloats}
\usepackage{adjustbox}
\usepackage[colorlinks=true,
            linkcolor=black,  
            citecolor=black,    
            urlcolor=blue      
]{hyperref}\newtheorem{assumption}{Assumption}
\newtheorem{definition}{Definition}
\newtheorem{theorem}{Theorem}
\newtheorem{corollary}{Corollary}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{remark}{Remark}
\usepackage[most]{tcolorbox}
\tcbset{highlightstyle/.style={
  colback=yellow!10, 
  colframe=red!50,   
  boxrule=0.5pt, arc=3pt, left=3pt, right=3pt, top=3pt, bottom=3pt
}}
\algrenewcommand\algorithmicrequire{\textbf{Input:}}
\algrenewcommand\algorithmicensure{\textbf{Output:}}
\usepackage{stfloats}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\usepackage{balance}

\begin{document}

\title{Learning Stage-wise Constraints from Expert Demonstrations}


\author{Baiyu Peng$^{1}$, Aude Billard$^{1}$%
\thanks{
This work was supported by euROBIN Grant.  } %Use only for final RAL version
\thanks{$^{1}$ The authors are with the LASA, School of Engineering, EPFL (Swiss Federal Institute of Technology in Lausanne), Lausanne 1015 Vaud, Switzerland.
        ({\tt\footnotesize baiyu.peng@epfl.ch} ; {\tt\footnotesize aude.billard@epfl.ch}). Corresponding author: Baiyu Peng.}%

}
\markboth{Journal of \LaTeX\ Class Files,~Vol.~18, No.~9, September~2024}%
{How to Use the IEEEtran \LaTeX \ Templates}

\maketitle

\begin{abstract}
Learning task constraints from demonstrations provides an explicit and reusable representation for downstream planning. However, many existing constraint-learning methods assume that the same constraint set remains active throughout an entire trajectory, which is restrictive for multi-stage robotic tasks where different constraints govern different phases and the stage cutpoints are unobserved. This paper studies stage-wise constraint learning from feasible demonstrations. Given a task-dependent feature bank, the goal is to recover, for each stage and feature, whether the feature is inactive, equality-constrained, lower-bound constrained, or upper-bound constrained, together with the corresponding target or boundary value when active. We propose Stage-Wise Constraint Learning (SWCL) to jointly segments demonstrations and recovers stage-wise constraints. For each candidate segment, SWCL evaluates local feature evidence for equality, inequality, or inactive interpretations, subgoal-progress consistency, and cross-demonstration agreement with shared stage-wise prototypes. With fixed prototypes, each demonstration is segmented exactly by dynamic programming over ordered cutpoints; the prototypes are then updated by aggregating local modes and parameters across demonstrations. Experiments on planar obstacle avoidance, slide insertion, and spherical surface inspection show that SWCL recovers more accurate cutpoints and constraint parameters than HMM-, HSMM-, and clustering-based baselines, even though the baselines are given ground-truth active constraint pairs during parameter evaluation. Planning experiments further show that the learned constraints can be reused as feasibility conditions in modified and new task instances.
\end{abstract}

\begin{IEEEkeywords}
Constraint learning, Learning from demonstration
\end{IEEEkeywords}
\section{Method}

We formulate stage-wise constraint learning as an alternating procedure between latent stage segmentation and robust estimation of a shared stage-wise constraint model. The key modeling assumption is that the constraint mode and constraint parameter associated with each stage-feature pair are task-level quantities shared across all demonstrations. Demonstrations may nevertheless exhibit different feature distributions around the same shared constraint due to execution variability and, particularly for inequality constraints, different degrees of conservativeness relative to the constraint boundary.

\subsection{Stage-wise Constraint Model}

Let
\[
\mathcal{D}
=
\{\tau^{(i)}\}_{i=1}^{N},
\qquad
\tau^{(i)}
=
\{x_t^{(i)}\}_{t=1}^{T_i},
\]
denote a set of $N$ feasible demonstrations. Given a task-dependent feature bank
\[
\Phi
=
\{\phi_j:\mathcal{X}\rightarrow\mathbb{R}\}_{j=1}^{F},
\]
we define the feature observation
\[
u_{t,j}^{(i)}
=
\phi_j(x_t^{(i)}).
\]

The task consists of $K$ ordered stages. The latent stage schedule of demonstration $i$ is represented by
\[
B^{(i)}
=
\{b_0^{(i)},\ldots,b_K^{(i)}\},
\qquad
1=b_0^{(i)}<\cdots<b_K^{(i)}=T_i+1,
\]
which induces the stage intervals
\[
I_k^{(i)}
=
\{b_{k-1}^{(i)},\ldots,b_k^{(i)}-1\}.
\]

Each stage-feature pair $(k,j)$ is associated with a shared constraint prototype
\[
\theta_{k,j}
=
(\gamma_{k,j},\eta_{k,j}),
\]
where $\gamma_{k,j}$ denotes the constraint mode and $\eta_{k,j}$ denotes the corresponding equality target or inequality boundary when the feature is active. The admissible modes are
\[
\gamma_{k,j}\in
\begin{cases}
\{\emptyset,\mathrm{eq}\},
& j\in\mathcal{F}_{\mathrm{eq}},\\
\{\emptyset,\mathrm{lb},\mathrm{ub}\},
& j\in\mathcal{F}_{\mathrm{ineq}}.
\end{cases}
\]
The complete shared stage-wise constraint model is denoted by
\[
\Theta
=
\{\theta_{k,j}\}_{k=1,j=1}^{K,F}.
\]

Importantly, $\gamma_{k,j}$ and $\eta_{k,j}$ are properties of the task and are shared by all demonstrations. We do not introduce demonstration-specific constraint modes or constraint parameters. Demonstration-dependent variability is instead represented through the observation model described below.
\subsection{Constraint-aware Feature Models}

For an arbitrary interval $I$ of demonstration $i$, let
\[
U_j^{(i)}(I)
=
\{u_{t,j}^{(i)}:t\in I\}
\]
denote the samples of feature $j$ within the interval. Each stage-feature pair can take one of four modes,
\[
\gamma_{k,j}
\in
\mathcal{M}
=
\{\emptyset,\mathrm{eq},\mathrm{lb},\mathrm{ub}\},
\]
corresponding respectively to an inactive feature, an equality constraint, a lower-bound inequality, or an upper-bound inequality. Each mode is associated with a probabilistic model of the observed feature values. The equality target or inequality boundary $\eta_{k,j}$ is a task-level parameter shared across demonstrations, whereas distributional nuisance parameters used to describe ordinary feature variation or the degree of inequality-boundary expression may adapt locally to each candidate interval.

\subsubsection{Inactive Mode}

An inactive feature is not assumed to follow the same distribution across all stages or demonstrations. In particular, an unconstrained feature may have different local means and different amounts of variation due to the task dynamics, controller, or other features. We therefore model the inactive mode using a candidate-specific Student-$t$ distribution,
\[
p_{\emptyset,j}
\left(
u\mid \mu,\sigma
\right)
=
\frac{1}{\sigma}
t_{\nu_0}
\left(
\frac{u-\mu}{\sigma}
\right),
\]
where both the location $\mu$ and scale $\sigma$ are local nuisance parameters.

If $\sigma$ were completely unconstrained, however, the inactive model could collapse to an arbitrarily narrow distribution and explain strongly concentrated observations as well as an equality model. To preserve a distinction between ordinary unconstrained behavior and equality-constrained behavior, we impose the lower bound
\[
\sigma
\ge
\sigma_{\mathrm{bg},j}^{\min},
\qquad
\sigma_{\mathrm{bg},j}^{\min}
=
c_{\mathrm{bg}}\sigma_{\mathrm{eq},j},
\qquad
c_{\mathrm{bg}}>1,
\]
where $\sigma_{\mathrm{eq},j}$ is the characteristic observation scale of an equality constraint on feature $j$. Thus, the inactive model may adapt its local center and dispersion, but it cannot use an arbitrarily concentrated distribution to imitate an equality constraint.

For a candidate interval $I$, the inactive-mode cost is obtained by profiling out the local nuisance parameters,
\[
\ell_{\emptyset,j}
\left(
U_j^{(i)}(I)
\right)
=
\min_{\mu,\,
\sigma\ge\sigma_{\mathrm{bg},j}^{\min}}
\left[
-\sum_{u\in U_j^{(i)}(I)}
\log
p_{\emptyset,j}
(u\mid\mu,\sigma)
\right].
\]
The corresponding $\widehat{\mu}$ and $\widehat{\sigma}$ are used only to evaluate the candidate interval and are not part of the learned task constraint model.

% The lower bound on the background scale replaces the previous hard
% equality-dispersion threshold with a separation between the equality
% and inactive distribution families. A value such as c_bg = 2 can be
% used as an initial choice and subsequently examined in sensitivity analysis.

\subsubsection{Equality Mode}

For an equality constraint, the feature is expected to remain concentrated around a shared stage-specific target $\eta_{k,j}$. We use a Student-$t$ observation model,
\[
p_{\mathrm{eq},j}
\left(
u\mid\eta_{k,j}
\right)
=
\frac{1}{\sigma_{\mathrm{eq},j}}
t_{\nu_{\mathrm{eq}}}
\left(
\frac{u-\eta_{k,j}}
{\sigma_{\mathrm{eq},j}}
\right),
\]
where $\sigma_{\mathrm{eq},j}>0$ is a feature-specific observation scale and $\nu_{\mathrm{eq}}$ denotes the degrees of freedom. The scale $\sigma_{\mathrm{eq},j}$ represents the expected execution, measurement, and demonstration variability around a true equality target and is not independently refitted for every candidate interval.

For a candidate interval $I$ evaluated under a shared equality target $\eta_{k,j}$, the equality-mode cost is
\[
\ell_{\mathrm{eq},j}
\left(
U_j^{(i)}(I);
\eta_{k,j}
\right)
=
-\sum_{u\in U_j^{(i)}(I)}
\log
p_{\mathrm{eq},j}
\left(
u\mid\eta_{k,j}
\right).
\]

When estimating an equality constraint from an already recovered stage interval, the target can be fitted by
\[
\widetilde{\eta}_{k,j}^{(i)}
=
\arg\min_{\eta}
\left[
-\sum_{u\in U_j^{(i)}(I_k^{(i)})}
\log
p_{\mathrm{eq},j}
(u\mid\eta)
\right],
\]
or by an equivalent robust center estimator.

\subsubsection{Inequality Modes}

For an inequality constraint, the boundary $\eta_{k,j}$ is shared across demonstrations, but different demonstrations may satisfy the same boundary with different amounts of clearance. We therefore allow the feasible-side dispersion to adapt locally while keeping the constraint boundary itself shared.

For a candidate direction $d\in\{\mathrm{lb},\mathrm{ub}\}$, define the signed slack
\[
y_d(u;\eta)
=
\begin{cases}
u-\eta,
& d=\mathrm{lb},\\
\eta-u,
& d=\mathrm{ub}.
\end{cases}
\]
Thus, $y_d(u;\eta)\geq0$ indicates that the feature value lies on the feasible side of the candidate boundary, while $y_d(u;\eta)<0$ represents a small violation.

We model the observations using a soft half-Student-$t$ density,
\[
p_{\mathrm{sht}}
(u\mid d,\eta,\sigma)
=
\frac{1}{Z}
\begin{cases}
\dfrac{2}{\sigma}
t_{\nu_{\mathrm{ineq}}}
\left(
\dfrac{y_d(u;\eta)}{\sigma}
\right),
&
y_d(u;\eta)\geq0,
\\[2mm]
\dfrac{2}{\sigma}
t_{\nu_{\mathrm{ineq}}}(0)
\exp
\left(
\dfrac{y_d(u;\eta)}
{\kappa\sigma}
\right),
&
y_d(u;\eta)<0,
\end{cases}
\]
where $\sigma>0$ controls the feasible-side dispersion, $\kappa>0$ determines the softness of the infeasible-side tail, and
\[
Z
=
1+2\kappa t_{\nu_{\mathrm{ineq}}}(0)
\]
is the normalization constant.

Unlike the shared boundary $\eta_{k,j}$, the scale $\sigma$ is treated as a local nuisance quantity. This allows one candidate interval to be tightly concentrated near the boundary while another interval satisfying the same inequality may remain farther inside the feasible region.

For interval $I$, direction $d$, and shared boundary $\eta$, the inequality cost is defined by profiling out this nuisance scale,
\[
\ell_{\mathrm{ineq},j}
\left(
U_j^{(i)}(I);
d,\eta
\right)
=
\min_{\sigma>0}
\left[
-\sum_{u\in U_j^{(i)}(I)}
\log
p_{\mathrm{sht}}
(u\mid d,\eta,\sigma)
+
R_{\sigma,j}(\sigma)
\right],
\]
where $R_{\sigma,j}(\sigma)$ is an optional weak regularizer preventing degenerate scale estimates. The corresponding nuisance-scale estimate is
\[
\widehat{\sigma}_{j}^{(i)}
(I;d,\eta)
=
\arg\min_{\sigma>0}
\left[
-\sum_{u\in U_j^{(i)}(I)}
\log
p_{\mathrm{sht}}
(u\mid d,\eta,\sigma)
+
R_{\sigma,j}(\sigma)
\right].
\]
This scale is used only internally for evaluating the interval and is not included in the recovered task constraint.

% The existing implementation can retain the current quantile-based
% plug-in estimate of sigma instead of explicitly solving the optimization
% above. In that case, for each candidate boundary eta and direction d,
% sigma is deterministically computed from the corresponding feasible-side
% slacks and substituted into the soft half-t likelihood.

\subsubsection{Unified Mode-specific Interval Costs}

The four candidate interpretations of a feature trace are therefore evaluated using
\[
\ell_{k,j}^{(i)}
\left(
I;\gamma,\eta
\right)
=
\begin{cases}
\ell_{\emptyset,j}
\left(
U_j^{(i)}(I)
\right),
&
\gamma=\emptyset,
\\[2mm]
\ell_{\mathrm{eq},j}
\left(
U_j^{(i)}(I);
\eta
\right),
&
\gamma=\mathrm{eq},
\\[2mm]
\ell_{\mathrm{ineq},j}
\left(
U_j^{(i)}(I);
\mathrm{lb},\eta
\right),
&
\gamma=\mathrm{lb},
\\[2mm]
\ell_{\mathrm{ineq},j}
\left(
U_j^{(i)}(I);
\mathrm{ub},\eta
\right),
&
\gamma=\mathrm{ub}.
\end{cases}
\]

The inactive model therefore provides a proper candidate-specific background likelihood for every feature, while its minimum scale prevents it from collapsing into a narrow equality-like distribution. Equality constraints are represented by concentration around a shared target, whereas inequality constraints are represented by a shared one-sided boundary with locally adaptive feasible-side dispersion. This enables the four modes to be compared within a common likelihood-based framework without requiring the equality-versus-inequality family of each feature to be specified in advance.
\subsection{Block A: Segmentation under the Shared Constraint Model}

Given the current shared constraint model
\[
\widehat{\Theta}^{(q)}
=
\{
(\widehat{\gamma}_{k,j}^{(q)},
\widehat{\eta}_{k,j}^{(q)})
\}_{k,j},
\]
each demonstration is segmented independently.

The essential difference from a local-fit-first formulation is that a candidate interval does not independently infer its own constraint mode or constraint boundary during segmentation. Instead, the interval is evaluated directly under the current shared constraint mode and parameter of the corresponding stage. Only the local inequality dispersion is allowed to adapt to the candidate interval.

For feature $j$ and candidate interval $I$ assigned to stage $k$, define
\[
\ell_{k,j}^{(i)}
\left(
I;
\widehat{\theta}_{k,j}^{(q)}
\right)
=
\begin{cases}
-\displaystyle\sum_{u\in U_j^{(i)}(I)}
\log q_j(u),
&
\widehat{\gamma}_{k,j}^{(q)}
=
\emptyset,
\\[4mm]
-\displaystyle\sum_{u\in U_j^{(i)}(I)}
\log
p_{\mathrm{eq},j}
\left(
u\mid
\widehat{\eta}_{k,j}^{(q)}
\right),
&
\widehat{\gamma}_{k,j}^{(q)}
=
\mathrm{eq},
\\[4mm]
\ell_{\mathrm{ineq},j}
\left(
U_j^{(i)}(I);
\mathrm{lb},
\widehat{\eta}_{k,j}^{(q)}
\right),
&
\widehat{\gamma}_{k,j}^{(q)}
=
\mathrm{lb},
\\[3mm]
\ell_{\mathrm{ineq},j}
\left(
U_j^{(i)}(I);
\mathrm{ub},
\widehat{\eta}_{k,j}^{(q)}
\right),
&
\widehat{\gamma}_{k,j}^{(q)}
=
\mathrm{ub}.
\end{cases}
\]

The constraint-based cost of assigning interval $I$ to stage $k$ is
\[
C_{\mathrm{con},k}^{(i)}(I)
=
\sum_{j=1}^{F}
\ell_{k,j}^{(i)}
\left(
I;
\widehat{\theta}_{k,j}^{(q)}
\right).
\]

If desired, the motion-progress term from the original formulation can be retained as an auxiliary temporal regularizer,
\[
C_k^{(i)}(I)
=
C_{\mathrm{con},k}^{(i)}(I)
+
\lambda_{\mathrm{prog}}
C_{\mathrm{prog}}^{(i)}(I).
\]
The progress term is not interpreted as a learned task constraint; it only provides additional structural information when feature likelihoods alone do not sufficiently determine the stage boundaries.

% A duration prior or minimum-duration assumption can also be incorporated here if required:
%
% \[
% C_k^{(i)}(I)
% =
% C_{\mathrm{con},k}^{(i)}(I)
% -
% \log p_{\mathrm{dur}}(|I|)
% +
% \lambda_{\mathrm{prog}}
% C_{\mathrm{prog}}^{(i)}(I).
% \]
%
% This should only be included if experiments show that an explicit duration model is necessary.

Because the total cost is additive across ordered stages, the optimal segmentation conditioned on the current shared prototype can be obtained exactly by dynamic programming.

Let
\[
F_k^{(i)}(t)
\]
denote the minimum cost of segmenting the prefix
$\{x_1^{(i)},\ldots,x_t^{(i)}\}$ into the first $k$ stages. The recurrence is
\[
F_k^{(i)}(t)
=
\min_{s}
\left[
F_{k-1}^{(i)}(s-1)
+
C_k^{(i)}([s,t])
\right].
\]
The initialization is
\[
F_0^{(i)}(0)=0,
\qquad
F_0^{(i)}(t)=+\infty
\quad
\text{for }t>0.
\]
Backtracking from $F_K^{(i)}(T_i)$ yields the updated stage schedule
\[
B^{(i,q+1)}
=
\{b_k^{(i,q+1)}\}_{k=0}^{K}.
\]

Thus, Block A performs a hard latent segmentation conditioned on the current task-level shared constraint model. Candidate intervals are not permitted to alter the shared constraint mode or boundary during this step.

\subsection{Block B: Robust Shared Constraint Estimation}

Given the updated segmentations
\[
\{B^{(i,q+1)}\}_{i=1}^{N},
\]
we next update the shared stage-wise constraint model.

Rather than pooling all raw observations from all demonstrations and estimating a single parameter directly, we first obtain an independent constraint estimate from each demonstration. This gives each demonstration approximately equal influence on the shared prototype and reduces sensitivity to individual demonstrations with unusually long stages, noisy executions, or imperfect segmentation.

For demonstration $i$, stage $k$, and feature $j$, let
\[
\widetilde{\theta}_{k,j}^{(i)}
=
\left(
\widetilde{\gamma}_{k,j}^{(i)},
\widetilde{\eta}_{k,j}^{(i)}
\right)
=
\mathcal{E}_j
\left(
U_j^{(i)}(I_k^{(i)})
\right)
\]
denote the independently estimated constraint mode and parameter.

These quantities are local estimators of a common task-level constraint and are not interpreted as demonstration-specific true constraints.

For equality-type features, the local estimator compares the inactive and equality interpretations. When equality is selected, its local target is estimated using a robust center estimator or an equivalent Student-$t$ location fit.

For inequality-type features, the local estimator compares the inactive, lower-bound, and upper-bound interpretations. For each active inequality direction, the local boundary and nuisance dispersion are estimated from the corresponding recovered stage interval. The nuisance dispersion is discarded after constructing the local constraint estimate.

% Conceptually, the local inequality estimator can be written as
%
% \[
% \left(
% \widetilde{\eta}_{k,j}^{(i)}(d),
% \widetilde{\sigma}_{k,j}^{(i)}(d)
% \right)
% =
% \arg\min_{\eta,\sigma>0}
% \left[
% -
% \sum_{u\in U_j^{(i)}(I_k^{(i)})}
% \log
% p_{\mathrm{sht}}(u\mid d,\eta,\sigma)
% +
% R_{\sigma,j}(\sigma)
% \right],
% \]
%
% for d in {lb,ub}, followed by comparison with the inactive model.
%
% If the current quantile-based fitting rule is retained, eta and sigma can instead be obtained using the existing robust boundary and scale estimators.

The shared constraint mode is updated by majority voting,
\[
\widehat{\gamma}_{k,j}^{(q+1)}
=
\arg\max_{m\in\mathcal{M}_j}
\sum_{i=1}^{N}
\mathbf{1}
\left[
\widetilde{\gamma}_{k,j}^{(i)}
=
m
\right].
\]

For an active shared mode, the corresponding constraint parameter is updated by the median of the mode-consistent demonstration-level estimates,
\[
\widehat{\eta}_{k,j}^{(q+1)}
=
\operatorname{median}
\left\{
\widetilde{\eta}_{k,j}^{(i)}
:
\widetilde{\gamma}_{k,j}^{(i)}
=
\widehat{\gamma}_{k,j}^{(q+1)}
\right\},
\qquad
\widehat{\gamma}_{k,j}^{(q+1)}
\neq
\emptyset.
\]

This update deliberately performs aggregation at the demonstration level rather than at the sample level. Majority voting is robust to occasional mode-estimation errors, while the median parameter estimator limits the influence of demonstrations whose recovered stage or fitted constraint parameter is atypical.

The robust aggregation admits a simple estimation interpretation. Majority voting minimizes the number of demonstration-level mode disagreements,
\[
\widehat{\gamma}_{k,j}^{(q+1)}
=
\arg\min_{m\in\mathcal{M}_j}
\sum_{i=1}^{N}
\mathbf{1}
\left[
\widetilde{\gamma}_{k,j}^{(i)}
\neq
m
\right],
\]
while the median minimizes the sum of absolute deviations among mode-consistent parameter estimates,
\[
\widehat{\eta}_{k,j}^{(q+1)}
=
\arg\min_{\eta}
\sum_{i:
\widetilde{\gamma}_{k,j}^{(i)}
=
\widehat{\gamma}_{k,j}^{(q+1)}
}
\left|
\widetilde{\eta}_{k,j}^{(i)}
-
\eta
\right|.
\]

\subsection{Alternating Learning}

Blocks A and B are alternated until the recovered segmentations and shared constraint prototypes stop changing or a prescribed maximum number of iterations is reached. Starting from an initial shared prototype $\widehat{\Theta}^{(0)}$, iteration $q$ follows
\[
\widehat{\Theta}^{(q)}
\;\xrightarrow{\mathrm{Block~A}}\;
\{B^{(i,q+1)}\}_{i=1}^{N}
\;\xrightarrow{\mathrm{Block~B}}\;
\widehat{\Theta}^{(q+1)}.
\]

Block A evaluates each candidate interval directly under the current shared constraint semantics and solves the resulting ordered segmentation exactly by dynamic programming. Block B then independently estimates the constraint expressed by each recovered demonstration-stage pair and robustly aggregates these estimates into the updated shared task-level prototype.

Compared with a formulation in which candidate intervals first infer their own constraint modes and parameters and are subsequently penalized for disagreement with a shared prototype, the present formulation separates the two roles more clearly. During segmentation, the shared mode and boundary are fixed and directly determine the candidate-interval likelihood. Demonstration-level constraint fitting is performed only after segmentation, where it serves exclusively to robustly update the shared prototype.

% The method is closely related to hard-EM or Viterbi-style alternating latent-stage estimation, because Block A computes a hard segmentation conditioned on the current shared model. However, Block B deliberately uses robust demonstration-level majority and median aggregation rather than the pooled maximum-likelihood M-step of classical hard EM. Therefore, the complete algorithm should not be described as exact hard EM or exact coordinate ascent on a single raw-data likelihood. A safer description is likelihood-based alternating segmentation and robust shared-constraint estimation.

\begin{algorithm}[t]
\caption{Stage-Wise Constraint Learning with Robust Shared Prototypes}
\label{alg:swcl}
\begin{algorithmic}[1]
\REQUIRE Demonstrations $\mathcal{D}$, number of stages $K$, feature bank $\Phi$, admissible mode sets $\{\mathcal{M}_j\}$
\ENSURE Shared constraint model $\widehat{\Theta}$ and stage schedules $\{B^{(i)}\}$

\STATE Extract and standardize feature trajectories
\STATE Initialize the shared constraint model $\widehat{\Theta}^{(0)}$
\STATE $q\leftarrow0$

\REPEAT

\STATE \textbf{Block A: segmentation under the shared constraint model}

\FOR{$i=1,\ldots,N$}
    \FOR{each stage $k$ and candidate interval $I$}
        \FOR{$j=1,\ldots,F$}
            \STATE Evaluate $U_j^{(i)}(I)$ under the current shared mode $\widehat{\gamma}_{k,j}^{(q)}$ and parameter $\widehat{\eta}_{k,j}^{(q)}$
            \IF{$\widehat{\gamma}_{k,j}^{(q)}\in\{\mathrm{lb},\mathrm{ub}\}$}
                \STATE Adapt the local inequality nuisance scale $\sigma$ for this candidate interval
            \ENDIF
        \ENDFOR
        \STATE Compute the candidate stage cost $C_k^{(i)}(I)$
    \ENDFOR
    \STATE Solve the ordered segmentation by dynamic programming
    \STATE Backtrack to recover $B^{(i,q+1)}$
\ENDFOR

\STATE \textbf{Block B: robust shared-prototype update}

\FOR{$k=1,\ldots,K$}
    \FOR{$j=1,\ldots,F$}
        \FOR{$i=1,\ldots,N$}
            \STATE Independently estimate
            $\left(
            \widetilde{\gamma}_{k,j}^{(i)},
            \widetilde{\eta}_{k,j}^{(i)}
            \right)$
            from the recovered interval $I_k^{(i)}$
        \ENDFOR
        \STATE Update $\widehat{\gamma}_{k,j}^{(q+1)}$ by majority vote
        \IF{$\widehat{\gamma}_{k,j}^{(q+1)}\neq\emptyset$}
            \STATE Update $\widehat{\eta}_{k,j}^{(q+1)}$ by the median of the mode-consistent estimates
        \ENDIF
    \ENDFOR
\ENDFOR

\STATE $q\leftarrow q+1$

\UNTIL{the segmentations and shared prototypes converge, or the maximum number of iterations is reached}

\RETURN $\widehat{\Theta}$ and $\{B^{(i)}\}$
\end{algorithmic}
\end{algorithm}

\bibliographystyle{IEEEtran}
\bibliography{ref}
% \begin{thebibliography}{1}
% \bibliographystyle{IEEEtran}


\end{document}
