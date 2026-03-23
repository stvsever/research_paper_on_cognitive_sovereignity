from __future__ import annotations

import json
import os
import shutil
import textwrap
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.backend.utils.data_utils import infer_analysis_mode
from src.backend.utils.io import abs_path, ensure_dir, write_json, write_text
from src.backend.utils.scenario_realism import pretty_label


PALETTE = {
    "navy": "#14213d",
    "blue": "#1d4e89",
    "teal": "#1f7a8c",
    "coral": "#d95d39",
    "sand": "#f4e3b2",
    "ink": "#222222",
    "grid": "#d9dde6",
    "gold": "#c89b3c",
}


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _pretty_column(column_name: str) -> str:
    label = column_name
    for prefix in ["profile_cont_", "profile_cat__", "profile_cat_"]:
        if label.startswith(prefix):
            label = label[len(prefix) :]
    label = label.replace("__", " ")
    return pretty_label(label).title()


def _wrap(value: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def _fixed_effect_columns(df: pd.DataFrame) -> List[str]:
    return sorted(
        column
        for column in df.columns
        if column.startswith("opinion_leaf_fe_") or column.startswith("opinion_domain_fe_")
    )


def _treated_primary_formula(df: pd.DataFrame) -> str:
    terms = ["baseline_score", "baseline_abs_score", "primary_moderator_z"]
    if "exposure_quality_z" in df.columns:
        terms.append("exposure_quality_z")
    terms.extend(_fixed_effect_columns(df))
    return "delta_score ~ " + " + ".join(terms)


def _mixed_primary_formula(df: pd.DataFrame) -> str:
    terms = ["baseline_score", "attack_present", "primary_moderator_z", "attack_x_primary_moderator"]
    if "exposure_quality_z" in df.columns:
        terms.append("exposure_quality_z")
    terms.extend(_fixed_effect_columns(df))
    return "post_score ~ " + " + ".join(terms)


def _setup_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.edgecolor": PALETTE["navy"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 10,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _latex_escape_text(value: str) -> str:
    escaped = str(value)
    for old, new in {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }.items():
        escaped = escaped.replace(old, new)
    return escaped


def _p_to_stars(p_value: float | None) -> str:
    if p_value is None or pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _fmt_number(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _sem_lookup(sem_result: Dict[str, Any], lhs: str, rhs: str) -> Dict[str, Any] | None:
    for row in sem_result.get("coefficients", []):
        if row.get("lhs") == lhs and row.get("rhs") == rhs and row.get("op") == "~":
            return row
    return None


def _ols_lookup(ols_params: pd.DataFrame, term: str) -> Dict[str, Any] | None:
    match = ols_params.loc[ols_params["term"] == term]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _save_figure(fig: plt.Figure, base_path: Path) -> List[str]:
    ensure_dir(base_path.parent)
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [abs_path(png_path), abs_path(pdf_path)]


def _write_table_bundle(
    df: pd.DataFrame,
    base_path: Path,
    caption: str,
    note: str,
    label: str,
) -> List[str]:
    ensure_dir(base_path.parent)
    csv_path = base_path.with_suffix(".csv")
    tex_path = base_path.with_suffix(".tex")
    df.to_csv(csv_path, index=False)

    table_key = base_path.stem
    column_format = None
    body_font = r"\footnotesize"
    resize_wide = len(df.columns) > 6
    if len(df.columns) == 2:
        column_format = r"p{0.22\linewidth}p{0.72\linewidth}"
    if table_key == "supplementary_table_s1_ontology_leaves_used":
        column_format = r"p{0.11\linewidth}p{0.18\linewidth}p{0.55\linewidth}r"
        body_font = r"\scriptsize"
        resize_wide = False
    elif table_key == "supplementary_table_s3_assumption_and_risk_register":
        column_format = r"p{0.16\linewidth}p{0.26\linewidth}p{0.10\linewidth}p{0.40\linewidth}"
        body_font = r"\scriptsize"
        resize_wide = False
    elif table_key == "supplementary_table_s4_reproducibility_manifest":
        column_format = r"p{0.22\linewidth}p{0.72\linewidth}"
        body_font = r"\footnotesize"
        resize_wide = False
    elif table_key == "table_1_pilot_design_and_configuration":
        column_format = r"p{0.22\linewidth}p{0.72\linewidth}"
        body_font = r"\footnotesize"
        resize_wide = False
    elif table_key in {"table_2_susceptibility_descriptive_statistics", "table_2_condition_descriptive_statistics"}:
        body_font = r"\footnotesize"
        resize_wide = False
    elif table_key == "table_3_primary_moderation_model":
        column_format = r"p{0.30\linewidth}" + "r" * (len(df.columns) - 1)
        body_font = r"\scriptsize"
        resize_wide = True
    elif table_key == "supplementary_table_s2_exploratory_moderator_comparison":
        body_font = r"\scriptsize"
        resize_wide = True

    table_latex = df.to_latex(
        index=False,
        escape=True,
        na_rep="",
        float_format=lambda x: f"{x:.3f}",
        column_format=column_format,
    )
    body_lines: List[str] = []
    body_lines.append("{")
    body_lines.append(body_font)
    body_lines.append(r"\setlength{\tabcolsep}{4pt}")
    body_lines.append(r"\renewcommand{\arraystretch}{1.1}")
    if resize_wide:
        body_lines.extend([
            "\\resizebox{\\linewidth}{!}{%",
            table_latex.rstrip(),
            "}",
        ])
    else:
        body_lines.append(table_latex.rstrip())
    body_lines.append("}")

    tex_content = "\n".join(
        [
            "\\begin{table}[htbp]",
            "\\raggedright",
            f"\\caption{{{_latex_escape_text(caption)}}}",
            f"\\label{{{label}}}",
            *body_lines,
            "\\par\\smallskip",
            f"{{\\normalsize Note. {_latex_escape_text(note)}}}",
            "\\end{table}",
        ]
    )
    write_text(tex_path, tex_content)
    return [abs_path(csv_path), abs_path(tex_path)]


def _copy_tree_contents(source_dir: Path, target_dir: Path) -> List[str]:
    ensure_dir(target_dir)
    copied: List[str] = []
    for item in source_dir.iterdir():
        target = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        copied.append(abs_path(target))
    return copied


def _primary_moderator_stats(ols_params: pd.DataFrame) -> tuple[float, float]:
    row = _ols_lookup(ols_params, "primary_moderator_z")
    if row is None:
        row = _ols_lookup(ols_params, "attack_x_primary_moderator")
    if row is None:
        return np.nan, np.nan
    return float(row.get("estimate", np.nan)), float(row.get("p_value", np.nan))


def _susceptibility_terciles(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if work["primary_moderator_value"].nunique() < 3:
        work["susceptibility_group"] = "All"
        return work
    work["susceptibility_group"] = pd.qcut(
        work["primary_moderator_value"],
        q=3,
        labels=["Low susceptibility", "Mid susceptibility", "High susceptibility"],
        duplicates="drop",
    )
    return work


def _draw_pipeline_schematic(base_path: Path, paper_title: str, analysis_mode: str) -> List[str]:
    fig, ax = plt.subplots(figsize=(13.6, 6.4))
    ax.set_axis_off()

    boxes = [
        (0.03, 0.62, 0.18, 0.18, "PROFILE ontology\nhierarchy -> stratified\nprofile leaves"),
        (0.03, 0.37, 0.18, 0.18, "OPINION ontology\nhierarchy -> repeated\nleaf selection"),
        (0.03, 0.12, 0.18, 0.18, "ATTACK ontology\nhierarchy -> fixed\nadversarial leaf"),
        (0.28, 0.50, 0.18, 0.22, "Scenario manifest\nattacked-only design\n+ susceptibility coverage" if analysis_mode == "treated_only" else "Scenario manifest\ncondition assignment\n+ susceptibility coverage"),
        (0.50, 0.62, 0.18, 0.18, "Baseline opinion\nassessment"),
        (0.50, 0.32, 0.18, 0.18, "Exposure generation\n+ realism review"),
        (0.50, 0.02, 0.18, 0.18, "Post-exposure opinion\n+ coherence review"),
        (0.75, 0.50, 0.2, 0.22, "Delta dataset\nleaf fixed effects\n+ exposure quality"),
        (0.75, 0.14, 0.2, 0.22, "Path model / robust OLS\ninteractive dashboard\npublication assets"),
    ]

    for x, y, w, h, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.025",
                linewidth=1.4,
                edgecolor=PALETTE["navy"],
                facecolor="#f7fbff",
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=PALETTE["navy"], fontweight="bold")

    arrows = [
        ((0.21, 0.71), (0.28, 0.61)),
        ((0.21, 0.46), (0.28, 0.61)),
        ((0.21, 0.21), (0.28, 0.61)),
        ((0.46, 0.61), (0.50, 0.71)),
        ((0.46, 0.61), (0.50, 0.41)),
        ((0.68, 0.71), (0.75, 0.61)),
        ((0.68, 0.41), (0.75, 0.61)),
        ((0.68, 0.11), (0.75, 0.25)),
        ((0.59, 0.62), (0.59, 0.50)),
        ((0.59, 0.32), (0.59, 0.20)),
        ((0.85, 0.50), (0.85, 0.36)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.25, color=PALETTE["blue"]))

    ax.text(0.5, 0.94, "Ontology-driven attacked-only moderation workflow", ha="center", va="center", fontsize=16, fontweight="bold", color=PALETTE["navy"])
    ax.text(0.5, 0.885, paper_title, ha="center", va="center", fontsize=9.5, color=PALETTE["ink"])
    ax.text(
        0.28,
        0.84,
        "Hierarchical ontologies are preserved upstream, but only leaf nodes are sampled for estimation so the pilot remains interpretable and estimable.",
        ha="left",
        va="center",
        fontsize=9.3,
        color=PALETTE["ink"],
    )
    return _save_figure(fig, base_path)


def _draw_treated_delta_distribution(df: pd.DataFrame, base_path: Path, moderator_est: float, moderator_p: float) -> List[str]:
    work = _susceptibility_terciles(df)
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    order = [group for group in ["Low susceptibility", "Mid susceptibility", "High susceptibility"] if group in set(work["susceptibility_group"].astype(str))]
    sns.violinplot(
        data=work,
        x="susceptibility_group",
        y="delta_score",
        hue="susceptibility_group",
        order=order,
        palette=[PALETTE["teal"], PALETTE["gold"], PALETTE["coral"]][: len(order)],
        legend=False,
        inner=None,
        cut=0,
        linewidth=1.1,
        ax=ax,
    )
    sns.stripplot(
        data=work,
        x="susceptibility_group",
        y="delta_score",
        order=order,
        color=PALETTE["navy"],
        alpha=0.75,
        size=5.5,
        jitter=0.12,
        ax=ax,
    )
    sns.pointplot(
        data=work,
        x="susceptibility_group",
        y="delta_score",
        order=order,
        estimator=np.mean,
        errorbar=("ci", 95),
        color=PALETTE["sand"],
        linestyle="none",
        markers="D",
        markersize=8,
        ax=ax,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=PALETTE["navy"], alpha=0.7)
    ax.set_title("Attacked-only opinion deltas across susceptibility strata")
    ax.set_xlabel("")
    ax.set_ylabel("Post - baseline opinion score")
    ax.text(
        0.02,
        0.97,
        f"Primary moderator b = {moderator_est:.2f}, p = {moderator_p:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "boxstyle": "round,pad=0.3"},
    )
    return _save_figure(fig, base_path)


def _draw_treated_moderation_scatter(df: pd.DataFrame, base_path: Path, primary_moderator: str) -> List[str]:
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    formula = _treated_primary_formula(df)
    model = smf.ols(formula, data=df).fit()

    raw_values = df["primary_moderator_value"].astype(float)
    raw_grid = np.linspace(float(raw_values.min()), float(raw_values.max()), 80)
    raw_mean = float(raw_values.mean())
    raw_std = float(raw_values.std(ddof=0)) or 1.0
    z_grid = (raw_grid - raw_mean) / raw_std

    grid_df = pd.DataFrame(
        {
            "baseline_score": float(df["baseline_score"].mean()),
            "baseline_abs_score": float(df["baseline_abs_score"].mean()),
            "primary_moderator_z": z_grid,
        }
    )
    if "exposure_quality_z" in df.columns:
        grid_df["exposure_quality_z"] = float(df["exposure_quality_z"].mean())
    for column in _fixed_effect_columns(df):
        grid_df[column] = float(df[column].mean())

    prediction = model.get_prediction(grid_df).summary_frame(alpha=0.05)
    scatter = ax.scatter(
        df["primary_moderator_value"],
        df["delta_score"],
        c=df["baseline_score"],
        cmap="coolwarm",
        edgecolor="white",
        linewidth=0.7,
        s=80,
        alpha=0.92,
    )
    ax.plot(raw_grid, prediction["mean"], color=PALETTE["navy"], linewidth=2.7)
    ax.fill_between(raw_grid, prediction["mean_ci_lower"], prediction["mean_ci_upper"], color=PALETTE["blue"], alpha=0.16)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=PALETTE["navy"], alpha=0.6)
    ax.set_title("Predicted delta over the primary susceptibility moderator")
    ax.set_xlabel(_pretty_column(primary_moderator))
    ax.set_ylabel("Predicted opinion delta")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Baseline opinion score")
    return _save_figure(fig, base_path)


def _draw_mixed_delta_distribution(df: pd.DataFrame, base_path: Path, attack_effect_p: float) -> List[str]:
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    plot_df = df.copy()
    plot_df["Condition"] = plot_df["attack_present"].map({0: "Control", 1: "Attack"})
    sns.violinplot(
        data=plot_df,
        x="Condition",
        y="delta_score",
        hue="Condition",
        palette={"Control": PALETTE["teal"], "Attack": PALETTE["coral"]},
        inner=None,
        linewidth=1.1,
        cut=0,
        legend=False,
        ax=ax,
    )
    sns.stripplot(data=plot_df, x="Condition", y="delta_score", color=PALETTE["navy"], alpha=0.75, size=5.5, jitter=0.12, ax=ax)
    sns.pointplot(
        data=plot_df,
        x="Condition",
        y="delta_score",
        estimator=np.mean,
        errorbar=("ci", 95),
        color=PALETTE["sand"],
        linestyle="none",
        markers="D",
        markersize=8,
        ax=ax,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=PALETTE["navy"], alpha=0.7)
    ax.set_title("Opinion deltas by condition")
    ax.set_xlabel("")
    ax.set_ylabel("Post - baseline opinion score")
    ax.text(
        0.02,
        0.97,
        f"Adjusted attack effect p = {attack_effect_p:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "boxstyle": "round,pad=0.3"},
    )
    return _save_figure(fig, base_path)


def _draw_baseline_post_scatter(df: pd.DataFrame, base_path: Path, analysis_mode: str) -> List[str]:
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    if analysis_mode == "treated_only":
        hue = "opinion_leaf_label"
        palette = sns.color_palette("blend:#1d4e89,#d95d39", n_colors=max(3, df[hue].nunique()))
    else:
        plot_df = df.copy()
        plot_df["Condition"] = plot_df["attack_present"].map({0: "Control", 1: "Attack"})
        df = plot_df
        hue = "Condition"
        palette = {"Control": PALETTE["teal"], "Attack": PALETTE["coral"]}
    sns.scatterplot(data=df, x="baseline_score", y="post_score", hue=hue, palette=palette, s=70, ax=ax)
    min_axis = float(min(df["baseline_score"].min(), df["post_score"].min()))
    max_axis = float(max(df["baseline_score"].max(), df["post_score"].max()))
    ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color=PALETTE["navy"], linewidth=1.0)
    ax.set_title("Baseline vs post-exposure scores")
    ax.set_xlabel("Baseline score")
    ax.set_ylabel("Post-exposure score")
    ax.legend(loc="best", fontsize=8)
    return _save_figure(fig, base_path)


def _draw_attack_quality(df: pd.DataFrame, base_path: Path) -> List[str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    sns.histplot(df["attack_realism_score"].dropna(), kde=True, color=PALETTE["coral"], ax=axes[0])
    axes[0].set_title("Attack realism")
    axes[0].set_xlabel("Realism score")
    sns.scatterplot(data=df, x="exposure_quality_score", y="delta_score", color=PALETTE["blue"], s=70, ax=axes[1])
    axes[1].set_title("Exposure quality vs opinion delta")
    axes[1].set_xlabel("Exposure quality composite")
    axes[1].set_ylabel("Delta score")
    fig.suptitle("Exposure quality diagnostics", y=1.02, fontsize=13, fontweight="bold")
    return _save_figure(fig, base_path)


def _draw_scenario_composition(df: pd.DataFrame, base_path: Path) -> List[str]:
    work = _susceptibility_terciles(df)
    comp_df = (
        work.groupby(["opinion_leaf_label", "susceptibility_group"], as_index=False, observed=False)
        .size()
        .rename(columns={"size": "n_scenarios"})
    )
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    sns.barplot(
        data=comp_df,
        x="n_scenarios",
        y="opinion_leaf_label",
        hue="susceptibility_group",
        palette=[PALETTE["teal"], PALETTE["gold"], PALETTE["coral"]],
        ax=ax,
    )
    ax.set_title("Scenario composition by opinion leaf and susceptibility")
    ax.set_xlabel("Number of scenarios")
    ax.set_ylabel("Opinion leaf")
    ax.legend(title="Susceptibility group", loc="best", fontsize=8)
    return _save_figure(fig, base_path)


def _draw_coefficient_forest(exploratory_df: pd.DataFrame, base_path: Path, analysis_mode: str) -> List[str]:
    if exploratory_df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        ax.set_axis_off()
        ax.text(0.5, 0.6, "No moderator coefficients available", ha="center", va="center", fontsize=14, fontweight="bold")
        ax.text(0.5, 0.4, "The pilot did not yield enough variability for exploratory comparison.", ha="center", va="center", fontsize=10)
        return _save_figure(fig, base_path)

    forest_df = exploratory_df.copy().sort_values("effect_estimate")
    forest_df["display_label"] = forest_df["moderator_label"].map(lambda value: _wrap(value, 22))
    fig, ax = plt.subplots(figsize=(10.8, max(5.4, 0.8 * len(forest_df))))
    y_pos = np.arange(len(forest_df))
    colors = [PALETTE["coral"] if role == "primary" else PALETTE["blue"] for role in forest_df["role"]]
    ax.errorbar(
        forest_df["effect_estimate"],
        y_pos,
        xerr=[
            forest_df["effect_estimate"] - forest_df["effect_conf_low"],
            forest_df["effect_conf_high"] - forest_df["effect_estimate"],
        ],
        fmt="none",
        ecolor=PALETTE["navy"],
        elinewidth=1.25,
        capsize=3,
    )
    ax.scatter(forest_df["effect_estimate"], y_pos, c=colors, s=72, zorder=3)
    ax.axvline(0.0, linestyle="--", linewidth=1.0, color=PALETTE["navy"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(forest_df["display_label"])
    ax.set_xlabel("Moderator coefficient on opinion delta")
    ax.set_title("Primary and exploratory moderator coefficients")
    ax.margins(x=0.15)
    for y_idx, row in enumerate(forest_df.to_dict(orient="records")):
        ax.text(
            float(row["effect_conf_high"]) + 0.02 * (forest_df["effect_conf_high"].max() - forest_df["effect_conf_low"].min() + 1),
            y_idx,
            _p_to_stars(row["effect_p_value"]),
            va="center",
            ha="left",
            color=PALETTE["ink"],
            fontsize=11,
        )
    fig.subplots_adjust(left=0.38, right=0.95)
    return _save_figure(fig, base_path)


def _draw_sem_overview(
    base_path: Path,
    primary_moderator: str,
    sem_result: Dict[str, Any],
    ols_params: pd.DataFrame,
    analysis_mode: str,
) -> List[str]:
    fig, ax = plt.subplots(figsize=(11.6, 5.3))
    ax.set_axis_off()

    primary_label = _pretty_column(primary_moderator)
    sem_profile_to_baseline = _sem_lookup(sem_result, "baseline_score", f"{primary_label} (z)")
    ols_baseline = _ols_lookup(ols_params, "baseline_score")
    ols_extremity = _ols_lookup(ols_params, "baseline_abs_score")
    ols_primary = _ols_lookup(ols_params, "primary_moderator_z")
    ols_quality = _ols_lookup(ols_params, "exposure_quality_z")

    boxes = [
        (0.05, 0.42, 0.18, 0.22, f"PROFILE\n{primary_label}"),
        (0.30, 0.68, 0.18, 0.18, "Baseline\nopinion"),
        (0.30, 0.16, 0.18, 0.18, "Exposure\nquality"),
        (0.56, 0.68, 0.18, 0.18, "Opinion leaf\nfixed effects"),
        (0.79, 0.42, 0.16, 0.22, "Delta score\n(post - baseline)"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.025",
                linewidth=1.5,
                edgecolor=PALETTE["navy"],
                facecolor="#fbfcfe",
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=PALETTE["navy"], fontweight="bold")

    arrow_specs = [
        ((0.23, 0.53), (0.30, 0.77), sem_profile_to_baseline),
        ((0.48, 0.77), (0.79, 0.53), ols_baseline),
        ((0.48, 0.25), (0.79, 0.47), ols_quality),
        ((0.23, 0.53), (0.79, 0.60), ols_primary),
    ]
    for start, end, row in arrow_specs:
        color = PALETTE["coral"] if end[1] >= 0.55 else PALETTE["blue"]
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color=color))
        if row is not None:
            text = f"b = {_fmt_number(row.get('estimate'), 2)}\np = {_fmt_number(row.get('p_value'), 3)} {_p_to_stars(row.get('p_value'))}"
            ax.text(
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 + 0.05,
                text,
                ha="center",
                va="center",
                fontsize=8.8,
                bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "boxstyle": "round,pad=0.22"},
            )

    ax.add_patch(FancyArrowPatch((0.74, 0.77), (0.79, 0.57), arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color=PALETTE["gold"]))
    ax.text(0.71, 0.88, "leaf fixed effects", ha="center", va="center", fontsize=9.0, bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "boxstyle": "round,pad=0.22"})
    if ols_extremity is not None:
        ax.add_patch(FancyArrowPatch((0.40, 0.68), (0.79, 0.42), connectionstyle="arc3,rad=-0.22", arrowstyle="-|>", mutation_scale=16, linewidth=1.2, linestyle="--", color=PALETTE["navy"]))
        ax.text(
            0.60,
            0.29,
            f"Baseline extremity\nb = {_fmt_number(ols_extremity.get('estimate'), 2)}, p = {_fmt_number(ols_extremity.get('p_value'), 3)} {_p_to_stars(ols_extremity.get('p_value'))}",
            ha="center",
            va="center",
            fontsize=8.6,
            bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "boxstyle": "round,pad=0.22"},
        )

    ax.text(0.5, 0.95, "Annotated attacked-only path model", ha="center", va="center", fontsize=14, fontweight="bold", color=PALETTE["navy"])
    ax.text(0.5, 0.07, "Coefficients shown from the run-specific SEM / robust OLS outputs. Stars denote p-value thresholds used for visual annotation only.", ha="center", va="center", fontsize=9.0, color=PALETTE["ink"])
    return _save_figure(fig, base_path)


def generate_publication_assets(
    sem_long_csv_path: str | Path,
    sem_result_json_path: str | Path,
    ols_params_csv_path: str | Path,
    bootstrap_params_csv_path: str | Path,
    exploratory_comparison_csv_path: str | Path,
    config_json_path: str | Path,
    ontology_catalog_path: str | Path,
    assumptions_json_path: str | Path,
    critiques_json_path: str | Path,
    output_dir: str | Path,
    report_assets_root: str | Path,
    run_id: str,
    paper_title: str,
) -> Dict[str, Any]:
    _setup_theme()

    output_root = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_root / "figures")
    tables_dir = ensure_dir(output_root / "tables")
    snapshots_dir = ensure_dir(output_root / "data_snapshots")
    report_assets_root = ensure_dir(report_assets_root)
    report_figures_dir = ensure_dir(report_assets_root / "figures")
    report_tables_dir = ensure_dir(report_assets_root / "tables")

    df = pd.read_csv(sem_long_csv_path)
    sem_result = json.loads(Path(sem_result_json_path).read_text(encoding="utf-8"))
    ols_params = pd.read_csv(ols_params_csv_path)
    bootstrap_params = pd.read_csv(bootstrap_params_csv_path)
    exploratory = pd.read_csv(exploratory_comparison_csv_path)
    config = json.loads(Path(config_json_path).read_text(encoding="utf-8"))
    ontology_catalog = json.loads(Path(ontology_catalog_path).read_text(encoding="utf-8"))
    assumptions = json.loads(Path(assumptions_json_path).read_text(encoding="utf-8"))
    critiques = json.loads(Path(critiques_json_path).read_text(encoding="utf-8"))

    analysis_mode = infer_analysis_mode(df)
    primary_moderator = config.get("primary_moderator", "profile_cont_susceptibility_index")
    moderator_est, moderator_p = _primary_moderator_stats(ols_params)
    attack_row = _ols_lookup(ols_params, "attack_present")
    attack_p = float(attack_row["p_value"]) if attack_row else np.nan

    visual_files: List[str] = []
    table_files: List[str] = []

    visual_files.extend(_draw_pipeline_schematic(figures_dir / "figure_1_study_design", paper_title, analysis_mode))
    if analysis_mode == "treated_only":
        visual_files.extend(_draw_treated_delta_distribution(df, figures_dir / "figure_2_delta_distribution_by_susceptibility", moderator_est, moderator_p))
        visual_files.extend(_draw_treated_moderation_scatter(df, figures_dir / "figure_3_primary_moderation_interaction", primary_moderator))
        visual_files.extend(_draw_sem_overview(figures_dir / "figure_4_annotated_sem_path_diagram", primary_moderator, sem_result, ols_params, analysis_mode))
    else:
        visual_files.extend(_draw_mixed_delta_distribution(df, figures_dir / "figure_2_attack_control_delta_distribution", attack_p))
        visual_files.extend(_draw_treated_moderation_scatter(df, figures_dir / "figure_3_primary_moderation_interaction", primary_moderator))
        visual_files.extend(_draw_sem_overview(figures_dir / "figure_4_annotated_sem_path_diagram", primary_moderator, sem_result, ols_params, analysis_mode))

    visual_files.extend(_draw_baseline_post_scatter(df, figures_dir / "supplementary_figure_s1_baseline_post_scatter", analysis_mode))
    visual_files.extend(_draw_attack_quality(df, figures_dir / "supplementary_figure_s2_attack_quality"))
    visual_files.extend(_draw_scenario_composition(df, figures_dir / "supplementary_figure_s3_scenario_composition"))
    visual_files.extend(_draw_coefficient_forest(exploratory, figures_dir / "supplementary_figure_s4_moderator_coefficient_forest", analysis_mode))

    selected_opinion_leaves = config.get("max_opinion_leaves")
    table_1 = pd.DataFrame(
        {
            "Parameter": [
                "Run identifier",
                "Paper title",
                "Analysis mode",
                "Scenarios",
                "Attack ratio",
                "Attack leaf",
                "Focus opinion domain",
                "Opinion leaves sampled",
                "OpenRouter model",
                "Primary moderator",
                "Bootstrap samples",
            ],
            "Value": [
                run_id,
                paper_title,
                analysis_mode,
                len(df),
                config.get("attack_ratio"),
                config.get("attack_leaf"),
                config.get("focus_opinion_domain"),
                selected_opinion_leaves,
                config.get("openrouter_model"),
                primary_moderator,
                config.get("bootstrap_samples"),
            ],
        }
    )
    table_files.extend(
        _write_table_bundle(
            table_1,
            tables_dir / "table_1_pilot_design_and_configuration",
            "Pilot design and run configuration",
            f"Run-level configuration for the pilot simulation reported in {run_id}.",
            "tab:pilot_design",
        )
    )

    if analysis_mode == "treated_only":
        descriptives = (
            _susceptibility_terciles(df)
            .groupby("susceptibility_group", as_index=False, observed=False)
            .agg(
                n=("scenario_id", "count"),
                baseline_mean=("baseline_score", "mean"),
                post_mean=("post_score", "mean"),
                delta_mean=("delta_score", "mean"),
                delta_sd=("delta_score", "std"),
                exposure_quality_mean=("exposure_quality_score", "mean"),
            )
        )
        table_2_caption = "Descriptive statistics by susceptibility stratum"
        table_2_note = "All scenarios in run_5 received the same attack-vector family; strata are based on terciles of the primary susceptibility moderator."
        table_2_name = "table_2_susceptibility_descriptive_statistics"
    else:
        descriptives = (
            df.assign(Condition=df["attack_present"].map({0: "Control", 1: "Attack"}))
            .groupby("Condition", as_index=False)
            .agg(
                n=("scenario_id", "count"),
                baseline_mean=("baseline_score", "mean"),
                post_mean=("post_score", "mean"),
                delta_mean=("delta_score", "mean"),
                delta_sd=("delta_score", "std"),
                exposure_quality_mean=("exposure_quality_score", "mean"),
            )
        )
        table_2_caption = "Descriptive statistics by condition"
        table_2_note = "Scores are reported on the high-resolution signed opinion scale from -1000 to +1000."
        table_2_name = "table_2_condition_descriptive_statistics"
    table_files.extend(
        _write_table_bundle(
            descriptives,
            tables_dir / table_2_name,
            table_2_caption,
            table_2_note,
            "tab:descriptives",
        )
    )

    table_3 = ols_params.merge(bootstrap_params, on="term", how="left")
    table_files.extend(
        _write_table_bundle(
            table_3,
            tables_dir / "table_3_primary_moderation_model",
            "Primary delta model with robust and bootstrapped estimates",
            "The primary outcome is post-minus-baseline opinion delta. Robust standard errors are HC3. Bootstrap intervals are percentile intervals from resampled pilot fits and remain exploratory given the small sample.",
            "tab:primary_model",
        )
    )

    opinion_counts = (
        df.groupby(["opinion_leaf", "opinion_leaf_label"], as_index=False)
        .size()
        .rename(columns={"size": "n_scenarios"})
    )
    attack_leaf_counts = pd.DataFrame(
        {
            "attack_leaf": [config.get("attack_leaf")],
            "n_scenarios": [len(df)],
        }
    )
    supp_ontology = pd.concat(
        [
            opinion_counts.assign(ontology="OPINION").rename(columns={"opinion_leaf": "leaf_path", "opinion_leaf_label": "leaf_label"}),
            attack_leaf_counts.assign(ontology="ATTACK", leaf_label="Fixed attack leaf").rename(columns={"attack_leaf": "leaf_path"}),
        ],
        ignore_index=True,
        sort=False,
    )[["ontology", "leaf_label", "leaf_path", "n_scenarios"]]
    table_files.extend(
        _write_table_bundle(
            supp_ontology,
            tables_dir / "supplementary_table_s1_ontology_leaves_used",
            "Ontology leaves used in the pilot",
            f"Opinion leaves reflect the scenario allocation in {run_id}; the attack ontology was restricted to one common misinformation vector for this attacked-only pilot.",
            "tab:supp_ontology",
        )
    )

    table_files.extend(
        _write_table_bundle(
            exploratory,
            tables_dir / "supplementary_table_s2_exploratory_moderator_comparison",
            "Exploratory moderator comparison",
            "Each moderator was estimated in a separate delta model with baseline anchoring, exposure-quality control, and opinion fixed effects included as controls.",
            "tab:supp_exploratory",
        )
    )

    risk_rows: List[Dict[str, Any]] = []
    for item in assumptions:
        risk_rows.append(
            {
                "section": "Assumption register",
                "item": item.get("assumption"),
                "status": item.get("status"),
                "detail": item.get("mitigation"),
            }
        )
    for item in critiques:
        risk_rows.append(
            {
                "section": "Peer-review critique",
                "item": item.get("critique"),
                "status": "addressed",
                "detail": item.get("implemented_change"),
            }
        )
    risk_table = pd.DataFrame(risk_rows)
    table_files.extend(
        _write_table_bundle(
            risk_table,
            tables_dir / "supplementary_table_s3_assumption_and_risk_register",
            "Assumption and peer-review risk register",
            "This table summarizes the principal methodological assumptions and the explicit mitigations implemented in the pilot pipeline.",
            "tab:supp_risks",
        )
    )

    repro_table = pd.DataFrame(
        {
            "Field": [
                "run_id",
                "output_root",
                "ontology_root",
                "attack_leaf",
                "openrouter_model",
                "primary_moderator",
                "opinion_leaf_count",
                "attack_leaf_count",
            ],
            "Value": [
                config.get("run_id"),
                config.get("output_root"),
                config.get("ontology_root"),
                config.get("attack_leaf"),
                config.get("openrouter_model"),
                primary_moderator,
                ontology_catalog.get("opinion_leaf_count"),
                ontology_catalog.get("attack_leaf_count"),
            ],
        }
    )
    table_files.extend(
        _write_table_bundle(
            repro_table,
            tables_dir / "supplementary_table_s4_reproducibility_manifest",
            "Reproducibility manifest",
            "Selected run metadata required to reproduce the pilot configuration and ontology selection.",
            "tab:supp_repro",
        )
    )

    df.to_csv(snapshots_dir / "sem_long_encoded_snapshot.csv", index=False)
    ols_params.to_csv(snapshots_dir / "ols_params_snapshot.csv", index=False)
    exploratory.to_csv(snapshots_dir / "exploratory_moderators_snapshot.csv", index=False)

    copied_figures = _copy_tree_contents(figures_dir, report_figures_dir)
    copied_tables = _copy_tree_contents(tables_dir, report_tables_dir)

    manifest_path = output_root / "publication_assets_manifest.json"
    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "paper_title": paper_title,
            "analysis_mode": analysis_mode,
            "primary_moderator": primary_moderator,
            "visual_files": visual_files,
            "table_files": table_files,
            "report_asset_copies": {"figures": copied_figures, "tables": copied_tables},
        },
    )

    return {
        "visual_files": visual_files,
        "table_files": table_files,
        "snapshot_files": [
            abs_path(snapshots_dir / "sem_long_encoded_snapshot.csv"),
            abs_path(snapshots_dir / "ols_params_snapshot.csv"),
            abs_path(snapshots_dir / "exploratory_moderators_snapshot.csv"),
        ],
        "manifest_path": abs_path(manifest_path),
        "copied_figures": copied_figures,
        "copied_tables": copied_tables,
    }
