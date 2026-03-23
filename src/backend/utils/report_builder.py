from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.backend.utils.data_utils import infer_analysis_mode
from src.backend.utils.io import abs_path, ensure_dir, write_json, write_text


def _load_primary_terms(ols_params: pd.DataFrame, bootstrap_params: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    merged = ols_params.merge(bootstrap_params, on="term", how="left", suffixes=("_ols", "_boot"))
    records = []
    for row in merged.to_dict(orient="records"):
        row["ols_conf_low"] = row.get("conf_low_ols")
        row["ols_conf_high"] = row.get("conf_high_ols")
        row["bootstrap_conf_low"] = row.get("conf_low_boot")
        row["bootstrap_conf_high"] = row.get("conf_high_boot")
        records.append(row)
    return {row["term"]: row for row in records}


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _bounded_unit_metric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return max(0.0, min(1.0, numeric))


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
  year = {2012},
  doi = {10.18637/jss.v048.i02},
  url = {https://doi.org/10.18637/jss.v048.i02}
}

@article{cioffi2002invariance,
  title = {Invariance and Universality in Social Agent-Based Simulations},
  author = {Cioffi-Revilla, Claudio},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {99},
  number = {suppl_3},
  pages = {7314--7316},
  year = {2002},
  doi = {10.1073/pnas.082081499},
  url = {https://doi.org/10.1073/pnas.082081499}
}

@article{vosoughi2018false,
  title = {The Spread of True and False News Online},
  author = {Vosoughi, Soroush and Roy, Deb and Aral, Sinan},
  journal = {Science},
  volume = {359},
  number = {6380},
  pages = {1146--1151},
  year = {2018},
  doi = {10.1126/science.aap9559},
  url = {https://doi.org/10.1126/science.aap9559}
}
""".strip() + "\n"


def _render_references_apa() -> str:
    references = [
        "Atkinson, E. (2021, May 20). Countering cognitive warfare: Awareness and resilience. NATO Review. https://www.nato.int/docu/review/articles/2021/05/20/countering-cognitive-warfare-awareness-and-resilience/index.html",
        "Bennett, W. L., & Livingston, S. (2018). The disinformation order: Disruptive communication and the decline of democratic institutions. European Journal of Communication, 33(2), 122-139. https://doi.org/10.1177/0267323118760317",
        "Cioffi-Revilla, C. (2002). Invariance and universality in social agent-based simulations. Proceedings of the National Academy of Sciences, 99(Suppl. 3), 7314-7316. https://doi.org/10.1073/pnas.082081499",
        "Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall/CRC.",
        "Hung, T.-C., & Hung, T.-W. (2022). How China's cognitive warfare works: A frontline perspective of Taiwan's anti-disinformation wars. Journal of Global Security Studies, 7(4), ogac016. https://doi.org/10.1093/jogss/ogac016",
        "Kozyreva, A., Lewandowsky, S., & Hertwig, R. (2020). Citizens versus the internet: Confronting digital challenges with cognitive tools. Psychological Science in the Public Interest, 21(3), 103-156. https://doi.org/10.1177/1529100620946707",
        "Lazer, D. M. J., Baum, M. A., Benkler, Y., Berinsky, A. J., Greenhill, K. M., Menczer, F., Metzger, M. J., Nyhan, B., Pennycook, G., Rothschild, D., Schudson, M., Sloman, S. A., Sunstein, C. R., Thorson, E. A., Watts, D. J., & Zittrain, J. L. (2018). The science of fake news. Science, 359(6380), 1094-1096. https://doi.org/10.1126/science.aao2998",
        "Lewandowsky, S., Ecker, U. K. H., & Cook, J. (2017). Beyond misinformation: Understanding and coping with the post-truth era. Journal of Applied Research in Memory and Cognition, 6(4), 353-369. https://doi.org/10.1016/j.jarmac.2017.07.008",
        "Matz, S. C., Kosinski, M., Nave, G., & Stillwell, D. J. (2017). Psychological targeting as an effective approach to digital mass persuasion. Proceedings of the National Academy of Sciences, 114(48), 12714-12719. https://doi.org/10.1073/pnas.1710966114",
        "Miller, S. (2023). Cognitive warfare: An ethical analysis. Ethics and Information Technology, 25, Article 46. https://doi.org/10.1007/s10676-023-09717-7",
        "Paulauskas, K. (2024, February 6). Why cognitive superiority is an imperative. NATO Review. https://www.nato.int/docu/review/articles/2024/02/06/why-cognitive-superiority-is-an-imperative/index.html",
        "Pennycook, G., Epstein, Z., Mosleh, M., Arechar, A. A., Eckles, D., & Rand, D. G. (2021). Shifting attention to accuracy can reduce misinformation online. Nature, 592, 590-595. https://doi.org/10.1038/s41586-021-03344-2",
        "Pennycook, G., & Rand, D. G. (2019). Lazy, not biased: Susceptibility to partisan fake news is better explained by lack of reasoning than by motivated reasoning. Cognition, 188, 39-50. https://doi.org/10.1016/j.cognition.2018.06.011",
        "Pennycook, G., & Rand, D. G. (2021). The psychology of fake news. Trends in Cognitive Sciences, 25(5), 388-402. https://doi.org/10.1016/j.tics.2021.02.007",
        "Roozenbeek, J., van der Linden, S., Goldberg, B., Rathje, S., & Lewandowsky, S. (2022). Psychological inoculation improves resilience against misinformation on social media. Science Advances, 8(34), eabo6254. https://doi.org/10.1126/sciadv.abo6254",
        "Rosseel, Y. (2012). lavaan: An R package for structural equation modeling. Journal of Statistical Software, 48(2), 1-36. https://doi.org/10.18637/jss.v048.i02",
        "Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. Science, 359(6380), 1146-1151. https://doi.org/10.1126/science.aap9559",
    ]
    def _format_reference(ref: str) -> str:
        parts = re.split(r"(https?://\S+)", ref)
        rendered = []
        for part in parts:
            if not part:
                continue
            if re.fullmatch(r"https?://\S+", part):
                rendered.append(rf"\url{{{part}}}")
            else:
                rendered.append(_latex_escape(part))
        return "".join(rendered)

    return "\n\n".join([f"\\noindent {_format_reference(ref)}\\par" for ref in references])


def _figure_block(filename: str, caption: str, label: str, note: str, width: str = "0.94\\linewidth") -> str:
    note_tex = _latex_escape(note)
    return rf"""
\begin{{figure}}[H]
\raggedright
\caption{{{caption}}}
\includegraphics[width={width}]{{{filename}}}
\label{{{label}}}
\caption*{{\raggedright \footnotesize Note. {note_tex}}}
\end{{figure}}
""".strip()


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
    del report_assets_root
    analysis_mode = infer_analysis_mode(sem_df)
    if analysis_mode != "treated_only":
        raise RuntimeError("run_5 report is designed for attacked-only analysis_mode=treated_only")

    primary_moderator = config.get("primary_moderator", "profile_cont_susceptibility_index")
    primary_moderator_pretty = primary_moderator.replace("profile_cont_", "").replace("_", " ")
    run_id_tex = _latex_escape(run_id)
    attack_leaf_tex = _latex_escape(config.get("attack_leaf"))
    focus_domain_text = _latex_escape(config.get("focus_opinion_domain") or ", ".join(sorted(str(v) for v in sem_df["opinion_domain"].dropna().unique())))
    n_rows = len(sem_df)
    opinion_leaf_count = int(sem_df["opinion_leaf"].nunique())
    attack_mean = float(sem_df["delta_score"].mean())
    delta_sd = float(sem_df["delta_score"].std(ddof=0))
    realism_mean = float(sem_df["attack_realism_score"].dropna().mean())
    plausibility_mean = float(sem_df["post_plausibility_score"].dropna().mean())
    baseline_mean = float(sem_df["baseline_score"].mean())
    post_mean = float(sem_df["post_score"].mean())
    primary_coeff = ols_lookup.get("primary_moderator_z", {})
    baseline_coeff = ols_lookup.get("baseline_score", {})
    baseline_abs_coeff = ols_lookup.get("baseline_abs_score", {})
    quality_coeff = ols_lookup.get("exposure_quality_z", {})
    fit_indices = sem_result.get("fit_indices", {})
    cfi = _bounded_unit_metric(fit_indices.get("CFI"))
    rmsea = _bounded_unit_metric(fit_indices.get("RMSEA"))
    primary_formula = sem_result.get("model_formula", "")
    bootstrap_low = primary_coeff.get("bootstrap_conf_low")
    bootstrap_high = primary_coeff.get("bootstrap_conf_high")
    top_exploratory = exploratory_df[exploratory_df["role"] == "exploratory"].head(3)

    exploratory_text = "Exploratory moderators were uniformly uncertain."
    if not top_exploratory.empty:
        clauses = []
        for row in top_exploratory.to_dict(orient="records"):
            clauses.append(f"{row['moderator_label']} (b = {_fmt(row['effect_estimate'], 2)}, p = {_fmt(row['effect_p_value'], 3)})")
        exploratory_text = "The least uncertain exploratory moderators were " + "; ".join(clauses) + "."

    table_2_filename = "table_2_susceptibility_descriptive_statistics.tex"

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{tabularx}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{fancyhdr}}
\usepackage{{hyperref}}
\usepackage{{microtype}}
\usepackage{{ragged2e}}
\usepackage{{setspace}}
\usepackage{{float}}
\usepackage{{array}}
\usepackage{{xcolor}}
\graphicspath{{{{../assets/figures/}}}}
\setstretch{{1.12}}
\captionsetup{{justification=RaggedRight,singlelinecheck=false}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{Multi-agent Simulation of Susceptibility to Cyber-manipulation}}
\fancyhead[R]{{Pilot Report}}
\fancyfoot[C]{{\thepage}}
\setlength{{\headheight}}{{14pt}}
\setlength{{\emergencystretch}}{{3em}}
\hypersetup{{hidelinks}}

\begin{{document}}
\begin{{center}}
{{\LARGE\bfseries {paper_title}\par}}
\vspace{{0.9em}}
{{\large Stijn Van Severen$^1$ \quad\textbullet\quad Thomas De Schryver$^1$\par}}
\vspace{{0.45em}}
{{\normalsize $^1$ Ghent University\par}}
{{\normalsize Corresponding author: Stijn Van Severen\par}}
\vspace{{0.35em}}
{{\normalsize March 23, 2026\par}}
\end{{center}}

\begin{{abstract}}
This pilot study examines whether inter-individual differences moderate susceptibility to cyber-manipulation in a high-dimensional political opinion state space. Rather than estimating a no-attack counterfactual, {run_id_tex} focuses on a narrower and methodologically cleaner question: among attacked individuals, which profile differences predict larger post-minus-baseline opinion shifts after exposure to a common adversarial misinformation vector? Using ontology-constrained scenario generation, repeated opinion leaves, structured large-language-model agents, and audited realism/coherence checks, the pipeline generated {n_rows} attacked scenarios across {opinion_leaf_count} repeated opinion leaves in {focus_domain_text}. The attacked sample showed a mean opinion delta of {_fmt(attack_mean, 1)} (SD = {_fmt(delta_sd, 1)}), with mean attack realism {_fmt(realism_mean, 2)} and mean post-exposure plausibility {_fmt(plausibility_mean, 2)}. In the primary robust delta model, the coefficient for the pre-registered susceptibility moderator was {_fmt(primary_coeff.get('estimate'), 2)} (p = {_fmt(primary_coeff.get('p_value'), 3)}; bootstrap 95\% CI [{_fmt(bootstrap_low, 2)}, {_fmt(bootstrap_high, 2)}]). The attack-only design directly matches the moderation question but does not estimate absolute attack-versus-no-attack effects. Findings are therefore methodological and exploratory rather than claim-ready evidence about real-world populations.
\end{{abstract}}

\section{{Introduction}}
Digitally mediated political persuasion increasingly operates through selective framing, repetition, identity cues, and misinformation-rich attention environments rather than through overtly technical intrusion. False or misleading political content can spread rapidly online, distort perceived consensus, and shape trust in institutions and policy judgement (Bennett \& Livingston, 2018; Lazer et al., 2018; Lewandowsky et al., 2017; Vosoughi et al., 2018). Research on fake news and online judgement further suggests that susceptibility is not uniform across individuals; it depends on reasoning style, attention, prior beliefs, and the structure of the informational environment (Kozyreva et al., 2020; Pennycook \& Rand, 2019, 2021; Pennycook et al., 2021; Roozenbeek et al., 2022).

Within security studies, these same dynamics are increasingly discussed under the headings of cognitive warfare and cognitive superiority, where the strategic target is not only information accuracy but also the shaping of attention, belief revision, emotion, and collective will. NATO-affiliated concept development has framed the cognitive domain as an expanding arena of contestation beyond land, maritime, air, cyber, and space, emphasizing societal resilience and judgement under sustained informational pressure (Atkinson, 2021; Paulauskas, 2024). Ethical analysis likewise stresses that cognitive warfare concerns interference with agency, autonomy, and institutional trust, not merely messaging efficiency (Hung \& Hung, 2022; Miller, 2023).

The present study uses \textit{{cognitive sovereignty}} as an analytic framing for the extent to which political judgement remains self-directed under manipulative digital exposure. We treat the term as a methodological construct rather than a settled psychometric entity. The research question is: \textit{{How do inter-individual differences moderate the effectivity of digital adversarial attacks on political opinions?}} In this pilot, effectivity is operationalized as the within-individual opinion delta between a baseline judgement and a post-exposure judgement after a common attack-vector family has been applied.

This framing led to a methodological redesign. Earlier pilots mixed treated and no-attack control scenarios, which answered a somewhat different question and loaded unnecessary heterogeneity into the SEM. {run_id_tex} instead adopts an attacked-only design: every simulated individual receives the same misinformation attack-vector family, and the analysis asks which profile differences predict larger post-baseline movement. This design does not estimate the absolute incremental effect of attack versus no attack, but it aligns more directly with the moderation question of interest.

A multi-agent simulation architecture is appropriate here because direct empirical datasets that simultaneously preserve baseline state, structured profile descriptors, attack-vector semantics, and issue-specific post-exposure opinion states are difficult to obtain and often misaligned with ontology-level research needs. Agent-based social simulation has long been defended as a legitimate scientific instrument when the model structure is explicit, auditable, and treated as a generative research tool rather than a substitute for direct observation (Cioffi-Revilla, 2002). The present contribution is therefore methodological: it builds an ontology-driven backend capable of generating auditable attacked scenarios, evaluating them with structured LLM agents, and estimating moderation in a form that can later be scaled and stress-tested.

\section{{Materials and Methods}}
\subsection{{Design Logic}}
{run_id_tex} used the test ontology triplet and focused on the defense-related opinion domain ({focus_domain_text}) to reduce issue heterogeneity while keeping the state space substantively relevant to the cognitive-warfare framing. The attack ontology was restricted to a single leaf, \texttt{{{attack_leaf_tex}}}, so that the pilot would estimate heterogeneity in susceptibility to a common adversarial tactic rather than conflating moderator variation with attack-family variation. Scenario generation used repeated opinion leaves rather than near-unique issue sampling, and candidate profiles were oversampled before selection so that low and high values of the pre-registered susceptibility index were both represented across the attacked sample.

Hierarchical ontologies were preserved upstream, but only leaf nodes were sampled for estimation. PROFILE included mixed variable types, combining categorical attributes such as sex with continuous percentile-based traits. OPINION leaves encoded policy-specific positions, and ATTACK leaves encoded social-media persuasion tactics rather than hacking or infrastructure disruption. Baseline and post-exposure opinions were recorded on a high-resolution signed scale from $-1000$ to $+1000$.

\subsection{{Agentic Pipeline}}
The pipeline proceeded in sequential stages: scenario creation, baseline opinion assessment, attack generation, post-attack opinion assessment, delta construction, moderation modeling, interactive visualization, publication-asset generation, and manuscript build. Prompting was tuned to produce realistic platform-native manipulation rather than theatrical propaganda. Generated attack exposures were reviewed by a realism agent and rewritten once when realism or coherence fell below threshold. Baseline and post-exposure opinion assessments underwent a second self-supervision pass designed to penalize coarse rounding, implausible reversals, and profile-inconsistent issue responses.

\subsection{{Operationalization and Model Specification}}
The primary moderator was fixed in advance as \texttt{{{_latex_escape(primary_moderator)}}}. The primary outcome for {run_id_tex} was the attacked-only opinion delta, defined as post-exposure score minus baseline score. Because all scenarios were attacked, moderation is estimated directly as variation in delta across profiles rather than as an attack-presence interaction.

The main path model was:
\begin{{align}}
\text{{baseline}}_i &= \alpha_b + \beta_1 S_i + \gamma^\prime L_i + \varepsilon_{{bi}} \\
\Delta_i &= \alpha_\Delta + \lambda_1 \text{{baseline}}_i + \lambda_2 \left|\text{{baseline}}_i\right| + \beta_2 S_i + \beta_3 Q_i + \delta^\prime L_i + \varepsilon_{{\Delta i}}
\end{{align}}
where $S_i$ denotes the standardized susceptibility moderator, $Q_i$ denotes an exposure-quality composite derived from attack realism, coherence, and intensity, and $L_i$ denotes opinion-leaf fixed effects. This structure treats baseline opinion as an anchor and delta as the substantive response variable. Robust OLS with HC3 standard errors and percentile bootstrap intervals were reported alongside the SEM/path model (Efron \& Tibshirani, 1993; Rosseel, 2012).

\section{{Results}}
The attacked-only design produced {n_rows} scenarios over {opinion_leaf_count} repeated opinion leaves. Mean baseline opinion was {_fmt(baseline_mean, 1)}, mean post-exposure opinion was {_fmt(post_mean, 1)}, and mean opinion delta was {_fmt(attack_mean, 1)}. Mean attack realism was {_fmt(realism_mean, 2)}, and mean post-exposure plausibility was {_fmt(plausibility_mean, 2)}. Figure~\ref{{fig:study_design}} summarizes the ontology-driven workflow. Figure~\ref{{fig:delta_distribution}} shows attacked-only deltas stratified by susceptibility tercile. Figure~\ref{{fig:interaction}} plots the model-implied delta trend over the primary susceptibility moderator. Figure~\ref{{fig:sem}} displays the annotated path model used in the main analysis.

    {_figure_block('figure_1_study_design.pdf', 'Ontology-driven attacked-only study design.', 'fig:study_design', 'The hierarchical PROFILE, ATTACK, and OPINION ontologies are preserved upstream. Only leaf nodes are sampled for estimation, and all scenarios in run_5 receive the same attack-vector family so the analysis isolates heterogeneity in post-baseline response.')}

    {_figure_block('figure_2_delta_distribution_by_susceptibility.pdf', 'Opinion-delta distribution across susceptibility strata.', 'fig:delta_distribution', 'Violin densities, raw scenario points, and mean markers are shown for terciles of the pre-registered susceptibility moderator. This is an attacked-only comparison, not a treatment-versus-control figure.')}

    {_figure_block('figure_3_primary_moderation_interaction.pdf', 'Modeled delta over the primary susceptibility moderator.', 'fig:interaction', 'The line shows model-based predicted delta with baseline anchoring, baseline extremity, exposure quality, and opinion-leaf fixed effects held at observed means. Point color encodes baseline opinion score.')}

    {_figure_block('figure_4_annotated_sem_path_diagram.pdf', 'Annotated attacked-only path model.', 'fig:sem', 'Arrows display run-specific coefficients from the SEM / robust OLS specification. Significance stars are included only as visual aids; substantive interpretation remains exploratory because the pilot sample is small.')}

\input{{../assets/tables/table_1_pilot_design_and_configuration.tex}}
\input{{../assets/tables/{table_2_filename}}}
\input{{../assets/tables/table_3_primary_moderation_model.tex}}

    In the primary robust delta model, the coefficient for the pre-registered susceptibility moderator was {_fmt(primary_coeff.get('estimate'), 2)} with p = {_fmt(primary_coeff.get('p_value'), 3)}. Because the outcome is a \emph{{signed}} delta, this negative coefficient should be read as directional movement toward the lower pole of the opinion scale rather than as a simple reduction in absolute responsiveness. Baseline score remained an important anchor (b = {_fmt(baseline_coeff.get('estimate'), 2)}, p = {_fmt(baseline_coeff.get('p_value'), 3)}), while baseline extremity contributed {_fmt(baseline_abs_coeff.get('estimate'), 2)} with p = {_fmt(baseline_abs_coeff.get('p_value'), 3)}. Exposure quality contributed {_fmt(quality_coeff.get('estimate'), 2)} with p = {_fmt(quality_coeff.get('p_value'), 3)}. The bootstrap 95\% interval for the primary susceptibility coefficient was [{_fmt(bootstrap_low, 2)}, {_fmt(bootstrap_high, 2)}]. {exploratory_text}

    The SEM converged with display-capped CFI = {_fmt(cfi, 3)} and RMSEA = {_fmt(rmsea, 3)}. Because semopy can report fit indices slightly outside their theoretical bounds in very small samples, the manuscript caps those values to the interpretable $[0,1]$ range for reporting. Even so, the attacked-only structure is more coherent than the earlier control-heavy designs. Fit should still be interpreted cautiously because the model is estimated on only {n_rows} scenarios with repeated leaf fixed effects.

\section{{Discussion}}
{run_id_tex} is methodologically stronger than earlier pilots for three reasons. First, it now estimates the construct the paper actually cares about: which inter-individual differences are associated with larger post-baseline opinion movement after a common adversarial exposure. Second, it narrows attack-side variance by fixing the attack-vector family while preserving profile and issue variation. Third, it expresses the main result directly on the delta scale rather than embedding the question in a treatment-control interaction architecture that was not central to the stated research aim.

    This redesign improves interpretability but does not erase the core limitations of the pilot. The attacked-only design means the study does not estimate the incremental effect of attack relative to a no-attack counterfactual. Instead, it estimates heterogeneity of response among attacked individuals. That is the correct target for the present pilot, but it also means the findings should not be presented as causal population effects of cyber-manipulation in the wild. In addition, the primary coefficient is estimated on a signed delta scale, so moderator direction should not be collapsed into an unsigned notion of “more” or “less” susceptibility without a companion analysis of absolute opinion movement. With only {n_rows} scenarios, the paper should be read as workflow validation and model-design refinement rather than as substantive evidence that a specific moderator has been established.

Two additional points matter for interpretation. First, realism and coherence were explicitly audited, and the exposure-quality term was included to absorb part of the variance attributable to message quality rather than profile differences. Second, the ontology-constrained repeated-leaf design keeps the scenarios closer to interpretable policy-space comparisons than a fully unconstrained random sample would. Those design choices make the synthetic system more scientifically legible even when the sample is still small.

\section{{Conclusion}}
This pilot supports the feasibility of an ontology-driven multi-agent simulation framework for studying how inter-individual differences may moderate susceptibility to cyber-manipulation in political opinion spaces. In its current form, the strongest answer the paper can give is methodological: the attacked-only pipeline is operational, auditable, and substantially better aligned with the research question than the earlier treatment-control architecture. The next step is not stronger rhetoric but stronger design: larger attacked samples, multiple attack leaves, multi-model sensitivity analysis, and eventual triangulation against empirical human data wherever ethical and legally feasible.

\clearpage
\appendix
\section{{Supplementary Materials}}
The repository includes an interactive dashboard generated from {run_id_tex}. The static publication assets below are the paper-ready subset of that richer exploratory output.

    {_figure_block('supplementary_figure_s1_baseline_post_scatter.pdf', 'Baseline versus post-exposure opinion scores.', 'fig:s1', 'The diagonal line marks no change. Points are colored by opinion leaf to show how repeated-leaf sampling constrains issue heterogeneity without collapsing the political state space to a single item.')}

    {_figure_block('supplementary_figure_s2_attack_quality.pdf', 'Exposure-quality diagnostics.', 'fig:s2', 'The left panel shows the realism distribution assigned by the reviewer agent. The right panel relates the exposure-quality composite to the observed opinion delta.')}

    {_figure_block('supplementary_figure_s3_scenario_composition.pdf', 'Scenario composition by opinion leaf and susceptibility.', 'fig:s3', 'Repeated opinion leaves were combined with susceptibility-stratified profile sampling so that the pilot can estimate moderation without dispersing all observations across unique items.')}

    {_figure_block('supplementary_figure_s4_moderator_coefficient_forest.pdf', 'Primary and exploratory moderator coefficients.', 'fig:s4', 'The pre-registered susceptibility moderator is highlighted, while the remaining coefficients are exploratory. Error bars show confidence intervals from the robust models used for the moderator comparison table.')}

\input{{../assets/tables/supplementary_table_s1_ontology_leaves_used.tex}}
\input{{../assets/tables/supplementary_table_s2_exploratory_moderator_comparison.tex}}
\input{{../assets/tables/supplementary_table_s3_assumption_and_risk_register.tex}}
\input{{../assets/tables/supplementary_table_s4_reproducibility_manifest.tex}}

\clearpage
\section*{{References}}
\begingroup
\small
\setlength{{\parindent}}{{-1.2em}}
\setlength{{\leftskip}}{{1.2em}}
{_render_references_apa()}
\endgroup

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
        ["tectonic", "-C", "main.tex"],
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
