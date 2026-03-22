from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.backend.utils.io import abs_path, ensure_dir, write_json, write_text


def _load_primary_terms(ols_params: pd.DataFrame, bootstrap_params: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    merged = ols_params.merge(bootstrap_params, on="term", how="left")
    return {row["term"]: row for row in merged.to_dict(orient="records")}


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "&": "\\&",
        "%": "\\%",
        "#": "\\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _render_bib() -> str:
    return r"""
@article{rosseel2012lavaan,
  title = {lavaan: An {R} Package for Structural Equation Modeling},
  author = {Rosseel, Yves},
  journal = {Journal of Statistical Software},
  volume = {48},
  number = {2},
  pages = {1--36},
  year = {2012}
}

@book{efron1993bootstrap,
  title = {An Introduction to the Bootstrap},
  author = {Efron, Bradley and Tibshirani, Robert J.},
  year = {1993},
  publisher = {Chapman \& Hall/CRC}
}

@book{costa1992neo,
  title = {Neo {PI-R} Professional Manual},
  author = {Costa, Paul T. and McCrae, Robert R.},
  year = {1992},
  publisher = {Psychological Assessment Resources}
}

@article{vosoughi2018false,
  title = {The Spread of True and False News Online},
  author = {Vosoughi, Soroush and Roy, Deb and Aral, Sinan},
  journal = {Science},
  volume = {359},
  number = {6380},
  pages = {1146--1151},
  year = {2018}
}

@article{lewandowsky2017beyond,
  title = {Beyond Misinformation: Understanding and Coping with the Post-Truth Era},
  author = {Lewandowsky, Stephan and Ecker, Ullrich K. H. and Cook, John},
  journal = {Journal of Applied Research in Memory and Cognition},
  volume = {6},
  number = {4},
  pages = {353--369},
  year = {2017}
}
""".strip() + "\n"


def _render_tex(
    paper_title: str,
    run_id: str,
    config: Dict[str, Any],
    sem_df: pd.DataFrame,
    sem_result: Dict[str, Any],
    ols_lookup: Dict[str, Dict[str, Any]],
    exploratory_df: pd.DataFrame,
    report_assets_root: Path,
) -> str:
    primary_moderator = config.get("primary_moderator", "profile_cont_susceptibility_index")
    primary_moderator_tex = _latex_escape(primary_moderator)
    run_id_tex = _latex_escape(run_id)
    attack_leaf_tex = _latex_escape(config.get("attack_leaf"))
    n_rows = len(sem_df)
    attack_mean = float(sem_df.loc[sem_df["attack_present"] == 1, "delta_score"].mean())
    control_mean = float(sem_df.loc[sem_df["attack_present"] == 0, "delta_score"].mean())
    realism_mean = float(sem_df.loc[sem_df["attack_present"] == 1, "attack_realism_score"].dropna().mean())
    primary_interaction = ols_lookup.get("attack_x_primary_moderator", {})
    attack_effect = ols_lookup.get("attack_present", {})
    baseline_effect = ols_lookup.get("baseline_score", {})
    fit_indices = sem_result.get("fit_indices", {})
    cfi = fit_indices.get("CFI")
    rmsea = fit_indices.get("RMSEA")
    top_exploratory = exploratory_df[exploratory_df["role"] == "exploratory"].head(3)

    exploratory_text = "Exploratory moderator models were uniformly uncertain."
    if not top_exploratory.empty:
        clauses = []
        for row in top_exploratory.to_dict(orient="records"):
            clauses.append(
                f"{row['moderator_label']} (interaction est. {_fmt(row['interaction_estimate'], 2)}, p = {_fmt(row['interaction_p_value'], 3)})"
            )
        exploratory_text = "The most informative exploratory moderators were " + "; ".join(clauses) + "."

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{threeparttable}}
\usepackage{{longtable}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{fancyhdr}}
\usepackage{{hyperref}}
\usepackage{{natbib}}
\usepackage{{setspace}}
\usepackage{{float}}
\usepackage{{array}}
\usepackage{{xcolor}}

\graphicspath{{{{../assets/figures/}}}}
\setstretch{{1.12}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{Multi-agent Simulation of Susceptibility to Cyber-manipulation}}
\fancyhead[R]{{Pilot Report}}
\fancyfoot[C]{{\thepage}}
\setlength{{\headheight}}{{14pt}}

\title{{{paper_title}}}
\author{{Stijn Van Severen$^{{1,*}}$ \and Thomas De Schryver$^1$ \\
\normalsize $^1$Ghent University \\
\normalsize $^*$Corresponding author}}
\date{{March 22, 2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This pilot study examines whether inter-individual differences moderate susceptibility to cyber-manipulation in a high-dimensional political opinion state space. We implemented an ontology-driven multi-agent simulation pipeline linking PROFILE, ATTACK, and OPINION ontologies, generated baseline and post-exposure opinions with structured large-language-model agents, and estimated moderation using a parsimonious structural equation model supplemented by robust regression and bootstrap intervals. In run {run_id_tex}, the pilot included {n_rows} scenarios, balanced attack and control conditions, and one common misinformation vector. Attack-present scenarios showed a mean delta of {_fmt(attack_mean, 1)} versus {_fmt(control_mean, 1)} for controls, while the primary susceptibility interaction was estimated at {_fmt(primary_interaction.get("estimate"), 2)} (p = {_fmt(primary_interaction.get("p_value"), 3)}). The pipeline produced auditable realism and coherence checks, interactive exploration outputs, and print-ready publication assets. Findings are exploratory and intended to validate the methodology rather than support causal claims.
\end{{abstract}}

\section{{Introduction}}
Political persuasion in digitally mediated environments increasingly operates through repeated exposure, manipulative framing, and socially embedded misinformation rather than through overtly technical attacks \citep{{vosoughi2018false,lewandowsky2017beyond}}. The present project frames this problem in terms of \textit{{cognitive sovereignty}}: the degree to which an individual's political opinion state remains self-governed under adversarial informational pressure. Rather than relying on inaccessible or poorly ontology-mapped empirical datasets, this pilot uses a multi-agent simulation architecture to test whether profile-level differences systematically moderate attack effectivity.

The methodological contribution is twofold. First, the pipeline represents persons, attack vectors, and political opinion items using explicit hierarchical ontologies. Second, the simulation formalizes attack effectivity as a within-scenario shift between baseline and post-exposure opinions, enabling moderation analysis at the level of individual profile configuration. This pilot therefore focuses on methodological credibility, internal coherence, and reproducibility rather than on publication-grade statistical power.

\section{{Materials and Methods}}
\subsection{{Study Design}}
Run {run_id_tex} used the test ontology triplet and generated {n_rows} scenarios with a 50/50 control-treatment split. The treatment arm used a single common misinformation vector, \texttt{{{attack_leaf_tex}}}, to reduce design variance while the pipeline architecture was still being validated. Scenario creation was stratified over a derived susceptibility index so that low- and high-susceptibility profiles were represented across both conditions.

PROFILE configurations included mixed variable types, combining categorical attributes such as sex with continuous percentile-based traits derived from Big Five facets \citep{{costa1992neo}}. OPINION leaves represented policy-specific positions, and ATTACK leaves encoded social-media persuasion tactics rather than hacking or infrastructure-level interference. Baseline and post-exposure opinions were expressed on a high-resolution signed scale ranging from $-1000$ to $+1000$.

\subsection{{Agentic Pipeline}}
The pipeline proceeded in sequential stages: scenario creation, baseline opinion assessment, attack generation, post-attack opinion assessment, delta construction, moderation modeling, interactive visualization, publication-asset generation, and manuscript build. Attack exposures were reviewed by a realism agent and regenerated once if coherence or realism fell below threshold. Opinion assessments underwent a second self-supervision pass to penalize coarse scoring and implausible reversals.

\subsection{{Modeling Strategy}}
The primary moderator was fixed in advance as \texttt{{{primary_moderator_tex}}}. The main structural equation model followed a parsimonious pilot specification inspired by standard SEM practice \citep{{rosseel2012lavaan}}:
\begin{{quote}}
\texttt{{delta\_score \textasciitilde{{}} attack\_present + baseline\_score + susceptibility + attack\_present:susceptibility}} \\
\texttt{{post\_score \textasciitilde{{}} baseline\_score + attack\_present + susceptibility + attack\_present:susceptibility}}
\end{{quote}}
Because pilot samples are unstable for inference, robust OLS estimates and non-parametric bootstrap intervals \citep{{efron1993bootstrap}} were reported alongside the SEM.

\section{{Results}}
Figure~\ref{{fig:study_design}} summarizes the workflow, Figure~\ref{{fig:delta_distribution}} shows the treated-control delta distribution, Figure~\ref{{fig:interaction}} visualizes the primary moderation trend, and Figure~\ref{{fig:forest}} summarizes primary and exploratory interaction coefficients.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{figure_1_study_design.pdf}}
\caption{{Figure 1. Pilot study design and pipeline schematic.}}
\label{{fig:study_design}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{figure_2_attack_control_delta_distribution.pdf}}
\caption{{Figure 2. Attack versus control delta distribution with raw scenario points.}}
\label{{fig:delta_distribution}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{figure_3_primary_moderation_interaction.pdf}}
\caption{{Figure 3. Primary moderation interaction over susceptibility.}}
\label{{fig:interaction}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{figure_4_moderator_coefficient_forest.pdf}}
\caption{{Figure 4. Primary and exploratory moderator interaction coefficients.}}
\label{{fig:forest}}
\end{{figure}}

\input{{../assets/tables/table_1_pilot_design_and_configuration.tex}}
\input{{../assets/tables/table_2_condition_descriptive_statistics.tex}}
\input{{../assets/tables/table_3_primary_moderation_model.tex}}

The attack condition shifted opinions more strongly than the control condition in directional terms (mean treated delta = {_fmt(attack_mean, 1)}; mean control delta = {_fmt(control_mean, 1)}). The robust attack-present coefficient was {_fmt(attack_effect.get("estimate"), 2)} with p = {_fmt(attack_effect.get("p_value"), 3)}, while the primary moderation coefficient was {_fmt(primary_interaction.get("estimate"), 2)} with p = {_fmt(primary_interaction.get("p_value"), 3)}. Baseline opinion was comparatively stable across models (robust estimate = {_fmt(baseline_effect.get("estimate"), 3)}, p = {_fmt(baseline_effect.get("p_value"), 3)}).

The SEM converged, but fit remained pilot-limited (CFI = {_fmt(cfi, 3)}, RMSEA = {_fmt(rmsea, 3)}). Treated exposures achieved a mean realism score of {_fmt(realism_mean, 2)}, which indicates that the self-supervised reviewer generally accepted the generated persuasion content as platform-plausible. {exploratory_text}

\section{{Discussion}}
The present pilot demonstrates that the pipeline works end to end: ontology-guided scenario generation, structured LLM assessments, realism/coherence review loops, moderation estimation, static figure export, interactive SEM inspection, and automated manuscript generation all completed within a single reproducible workflow. This is the main result of the pilot.

At the same time, the inferential limitations are substantial. With only {n_rows} scenarios, the moderation coefficients are far too unstable for substantive claims about real-world susceptibility. The objective of this pilot is therefore methodological validation, not empirical estimation. The directional separation between attack and control conditions is encouraging because it suggests that the attack-generation and opinion-assessment components are not degenerate. However, the uncertainty around the interaction term means that profile moderation remains a hypothesis to be tested at scale rather than a result established here.

Three methodological design decisions improved the credibility of the pilot. First, the primary moderator was fixed in advance rather than selected opportunistically from whichever variable produced the strongest coefficient. Second, attack realism and opinion coherence were explicitly audited with repair loops. Third, the report package preserves provenance, stage manifests, and static/interactice outputs together, making the pipeline easier to scrutinize or reproduce.

\section{{Conclusion}}
This pilot supports the feasibility of an ontology-driven multi-agent simulation framework for studying how inter-individual differences may moderate susceptibility to cyber-manipulation in political opinion spaces. The pipeline is operational, auditable, and sufficiently modular to scale beyond the present pilot. The next step is not interpretive escalation but design expansion: more scenarios, multiple attack leaves, and cross-model sensitivity checks.

\clearpage
\appendix
\section{{Supplementary Materials}}
The repository also includes an interactive SEM dashboard generated from run {run_id_tex}. The static publication assets below are the paper-ready subset of that richer exploratory output.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\textwidth]{{supplementary_figure_s1_baseline_post_scatter.pdf}}
\caption{{Supplementary Figure S1. Baseline versus post-attack opinion scores.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{supplementary_figure_s2_attack_quality.pdf}}
\caption{{Supplementary Figure S2. Attack realism and coherence distributions.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{supplementary_figure_s3_scenario_composition.pdf}}
\caption{{Supplementary Figure S3. Scenario composition across opinion leaves.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{supplementary_figure_s4_sem_overview.pdf}}
\caption{{Supplementary Figure S4. Primary moderation model overview.}}
\end{{figure}}

\input{{../assets/tables/supplementary_table_s1_ontology_leaves_used.tex}}
\input{{../assets/tables/supplementary_table_s2_exploratory_moderator_comparison.tex}}
\input{{../assets/tables/supplementary_table_s3_assumption_and_risk_register.tex}}
\input{{../assets/tables/supplementary_table_s4_reproducibility_manifest.tex}}

\bibliographystyle{{apalike}}
\bibliography{{references}}

\end{{document}}
"""
    return tex.strip() + "\n"


def build_research_report(
    sem_long_csv_path: str | Path,
    sem_result_json_path: str | Path,
    ols_params_csv_path: str | Path,
    bootstrap_params_csv_path: str | Path,
    exploratory_comparison_csv_path: str | Path,
    config_json_path: str | Path,
    report_root: str | Path,
    report_assets_root: str | Path,
    paper_title: str,
    run_id: str,
) -> Dict[str, str]:
    report_root = ensure_dir(report_root)
    report_assets_root = ensure_dir(report_assets_root)

    sem_df = pd.read_csv(sem_long_csv_path)
    sem_result = json.loads(Path(sem_result_json_path).read_text(encoding="utf-8"))
    ols_params = pd.read_csv(ols_params_csv_path)
    bootstrap_params = pd.read_csv(bootstrap_params_csv_path)
    exploratory_df = pd.read_csv(exploratory_comparison_csv_path)
    config = json.loads(Path(config_json_path).read_text(encoding="utf-8"))
    ols_lookup = _load_primary_terms(ols_params, bootstrap_params)

    tex_path = Path(report_root) / "main.tex"
    bib_path = Path(report_root) / "references.bib"
    pdf_path = Path(report_root) / "main.pdf"
    summary_path = Path(report_root) / "report_summary.json"

    write_text(
        tex_path,
        _render_tex(
            paper_title=paper_title,
            run_id=run_id,
            config=config,
            sem_df=sem_df,
            sem_result=sem_result,
            ols_lookup=ols_lookup,
            exploratory_df=exploratory_df,
            report_assets_root=Path(report_assets_root),
        ),
    )
    write_text(bib_path, _render_bib())

    env = dict(os.environ)
    env.setdefault("XDG_CACHE_HOME", "/tmp/tectonic-cache")
    compile_result = subprocess.run(
        ["tectonic", "main.tex"],
        cwd=report_root,
        capture_output=True,
        text=True,
        env=env,
    )
    if compile_result.returncode != 0:
        raise RuntimeError(f"Tectonic compile failed:\nSTDOUT:\n{compile_result.stdout}\nSTDERR:\n{compile_result.stderr}")

    write_json(
        summary_path,
        {
            "run_id": run_id,
            "paper_title": paper_title,
            "main_tex": abs_path(tex_path),
            "main_pdf": abs_path(pdf_path),
            "references_bib": abs_path(bib_path),
        },
    )
    return {
        "tex_path": abs_path(tex_path),
        "pdf_path": abs_path(pdf_path),
        "bib_path": abs_path(bib_path),
        "summary_path": abs_path(summary_path),
    }
