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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
    "mint": "#2a9d8f",
}


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _pretty(value: str) -> str:
    text = value
    for prefix in ["profile_cont_", "profile_cat__profile_cat_", "profile_cat__", "profile_cat_", "abs_delta_indicator__"]:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.replace("_z", "")
    text = text.replace("__", " ")
    return pretty_label(text).title()


def _wrap(value: str, width: int = 26) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


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


def _save_figure(fig: plt.Figure, base_path: Path) -> List[str]:
    ensure_dir(base_path.parent)
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [abs_path(png_path), abs_path(pdf_path)]


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


def _write_table_bundle(df: pd.DataFrame, base_path: Path, caption: str, note: str, label: str) -> List[str]:
    ensure_dir(base_path.parent)
    csv_path = base_path.with_suffix(".csv")
    tex_path = base_path.with_suffix(".tex")
    df.to_csv(csv_path, index=False)

    resize_wide = len(df.columns) > 6
    column_format = None
    if len(df.columns) == 2:
        column_format = r"p{0.24\linewidth}p{0.70\linewidth}"
    elif base_path.stem == "table_3_multivariate_profile_moderator_model":
        column_format = r"p{0.34\linewidth}" + "r" * (len(df.columns) - 1)
    elif base_path.stem == "supplementary_table_s1_ontology_leaves_used":
        column_format = r"p{0.14\linewidth}p{0.22\linewidth}p{0.54\linewidth}"
        resize_wide = False

    table_latex = df.to_latex(index=False, escape=True, na_rep="", float_format=lambda x: f"{x:.3f}", column_format=column_format)
    body_lines = ["{", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.08}"]
    if resize_wide:
        body_lines.extend([r"\resizebox{\linewidth}{!}{%", table_latex.rstrip(), "}"])
    else:
        body_lines.append(table_latex.rstrip())
    body_lines.append("}")

    tex_content = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\raggedright",
            f"\\caption{{{_latex_escape_text(caption)}}}",
            f"\\label{{{label}}}",
            *body_lines,
            r"\par\smallskip",
            f"{{\\normalsize Note. {_latex_escape_text(note)}}}",
            r"\end{table}",
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


def _coefficient_lookup(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    match = df.loc[df["term"] == column]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def _draw_study_design(base_path: Path, config: Dict[str, Any]) -> List[str]:
    fig, ax = plt.subplots(figsize=(14.4, 6.5))
    ax.set_axis_off()
    n_profiles = int(config.get("n_profiles") or config.get("n_scenarios") or 0)

    boxes = [
        (0.03, 0.66, 0.18, 0.18, f"PROFILE ontology\\nleaf sampling -> {n_profiles}\\ndiverse pseudoprofiles"),
        (0.03, 0.38, 0.18, 0.18, "ATTACK ontology\nfixed leaf:\nmisleading narrative framing"),
        (0.03, 0.10, 0.18, 0.18, "OPINION ontology\nrepeated leaves within\nfocused policy domain"),
        (0.29, 0.46, 0.18, 0.24, "Profile-panel manifest\n50 profiles x repeated\nattacked opinion leaves"),
        (0.54, 0.63, 0.18, 0.18, "Baseline opinion\nassessment"),
        (0.54, 0.35, 0.18, 0.18, "Attack exposure\ngeneration + realism\naudit"),
        (0.54, 0.07, 0.18, 0.18, "Post-exposure opinion\nassessment + coherence\naudit"),
        (0.79, 0.46, 0.18, 0.24, "Profile-level effectivity\nabsolute attacked deltas\n+ latent SEM / robust OLS"),
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
                facecolor="#f7fbff",
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=PALETTE["navy"], fontweight="bold")

    arrows = [
        ((0.21, 0.75), (0.29, 0.58)),
        ((0.21, 0.47), (0.29, 0.58)),
        ((0.21, 0.19), (0.29, 0.58)),
        ((0.47, 0.58), (0.54, 0.72)),
        ((0.47, 0.58), (0.54, 0.44)),
        ((0.47, 0.58), (0.54, 0.16)),
        ((0.72, 0.72), (0.79, 0.58)),
        ((0.72, 0.44), (0.79, 0.58)),
        ((0.72, 0.16), (0.79, 0.58)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.3, color=PALETTE["blue"]))

    ax.text(0.5, 0.95, "Run 6 attacked-only profile-panel design", ha="center", va="center", fontsize=16, fontweight="bold", color=PALETTE["navy"])
    ax.text(
        0.29,
        0.88,
        "Hierarchical ontologies are preserved upstream. Estimation uses repeated leaf nodes so the fixed attack leaf connects to multiple attacked opinion-shift indicators per profile.",
        ha="left",
        va="center",
        fontsize=9.4,
        color=PALETTE["ink"],
    )
    return _save_figure(fig, base_path)


def _draw_abs_delta_distribution(long_df: pd.DataFrame, base_path: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(10.4, 5.9))
    sns.boxplot(data=long_df, x="opinion_leaf_label", y="abs_delta_score", color="#dce8f7", fliersize=0, ax=ax)
    sns.stripplot(data=long_df, x="opinion_leaf_label", y="abs_delta_score", color=PALETTE["navy"], alpha=0.65, jitter=0.18, size=4.5, ax=ax)
    ax.set_title("Absolute attacked opinion shift by repeated opinion leaf")
    ax.set_xlabel("Opinion leaf")
    ax.set_ylabel("Absolute post-baseline shift")
    ax.tick_params(axis="x", rotation=24)
    return _save_figure(fig, base_path)


def _draw_moderator_forest(weight_df: pd.DataFrame, base_path: Path) -> List[str]:
    work = weight_df.copy()
    for column in ["normalized_weight_pct", "estimate"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.loc[
        work["normalized_weight_pct"].notna()
        & np.isfinite(work["normalized_weight_pct"])
    ].sort_values("normalized_weight_pct", ascending=True)
    if work.empty:
        return []
    fig, ax = plt.subplots(figsize=(10.8, 6.1))
    palette = sns.color_palette("blend:#1f4b99,#d96c06", n_colors=max(3, work["ontology_group"].nunique()))
    group_palette = {group: palette[idx] for idx, group in enumerate(work["ontology_group"].dropna().unique())}
    bar_colors = [group_palette.get(group, PALETTE["blue"]) for group in work["ontology_group"]]
    ax.barh(work["moderator_label"], work["normalized_weight_pct"], color=bar_colors, alpha=0.9)
    for y, (_, row) in enumerate(work.iterrows()):
        ax.text(
            row["normalized_weight_pct"] + 0.35,
            y,
            f"{row['normalized_weight_pct']:.1f}% | b={row['estimate']:.2f}",
            va="center",
            fontsize=8.8,
        )
    ax.set_title("Descriptive susceptibility weights across profile moderators")
    ax.set_xlabel("Normalized weight share (%)")
    ax.set_ylabel("Moderator")
    return _save_figure(fig, base_path)


def _draw_sem_diagram(
    sem_result: Dict[str, Any],
    exploratory_df: pd.DataFrame,
    config: Dict[str, Any],
    indicator_columns: List[str],
    base_path: Path,
) -> List[str]:
    sem_lookup = pd.DataFrame(sem_result.get("coefficients", []))
    if sem_lookup.empty:
        return []
    sem_lookup = sem_lookup.loc[sem_lookup["op"].astype(str) == "~"].copy()
    sem_lookup["estimate"] = pd.to_numeric(sem_lookup["estimate"], errors="coerce")
    sem_lookup["p_value"] = pd.to_numeric(sem_lookup["p_value"], errors="coerce")
    sem_lookup = sem_lookup.loc[
        sem_lookup["lhs"].isin(indicator_columns)
        & sem_lookup["estimate"].notna()
        & np.isfinite(sem_lookup["estimate"])
    ]
    if sem_lookup.empty:
        return []

    heatmap_df = sem_lookup.pivot_table(index="rhs", columns="lhs", values="estimate", aggfunc="mean")
    ordered_columns = (
        exploratory_df.sort_values("normalized_weight_pct", ascending=False)["moderator_label"].tolist()
        if "normalized_weight_pct" in exploratory_df.columns
        else list(heatmap_df.index)
    )
    ordered_columns = [column for column in ordered_columns if column in heatmap_df.index]
    heatmap_df = heatmap_df.loc[ordered_columns]
    heatmap_df = heatmap_df[[column for column in indicator_columns if column in heatmap_df.columns]]

    annot = heatmap_df.copy().astype(str)
    for rhs in heatmap_df.index:
        for lhs in heatmap_df.columns:
            row = sem_lookup.loc[(sem_lookup["lhs"] == lhs) & (sem_lookup["rhs"] == rhs)]
            if row.empty:
                annot.loc[rhs, lhs] = ""
                continue
            first = row.iloc[0]
            p_value = first["p_value"]
            stars = "***" if pd.notna(p_value) and p_value < 0.001 else "**" if pd.notna(p_value) and p_value < 0.01 else "*" if pd.notna(p_value) and p_value < 0.05 else "†" if pd.notna(p_value) and p_value < 0.10 else ""
            annot.loc[rhs, lhs] = f"{first['estimate']:.2f}{stars}"

    fig, ax = plt.subplots(figsize=(12.6, 7.4))
    sns.heatmap(
        heatmap_df,
        cmap="RdBu_r",
        center=0.0,
        linewidths=0.6,
        linecolor="white",
        annot=annot,
        fmt="",
        cbar_kws={"label": "SEM path coefficient"},
        ax=ax,
    )
    ax.set_title("Path-SEM coefficients from profile moderators to attacked opinion shifts")
    ax.set_xlabel("Repeated attacked opinion outcome")
    ax.set_ylabel("Profile moderator")
    ax.set_yticklabels([_wrap(_pretty(index), 24) for index in heatmap_df.index], rotation=0)
    ax.set_xticklabels([_wrap(_pretty(column), 18) for column in heatmap_df.columns], rotation=15, ha="right")
    fig.text(
        0.01,
        0.01,
        "Note. Cells show path coefficients from the repeated-outcome SEM. Stars mark p < .05; dagger marks p < .10. The fixed ATTACK leaf is a design constant applied to all repeated opinion outcomes.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=PALETTE["ink"],
    )
    return _save_figure(fig, base_path)


def _draw_baseline_post_scatter(long_df: pd.DataFrame, base_path: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    sns.scatterplot(data=long_df, x="baseline_score", y="post_score", hue="opinion_leaf_label", palette="Set2", s=46, ax=ax)
    min_axis = float(min(long_df["baseline_score"].min(), long_df["post_score"].min()))
    max_axis = float(max(long_df["baseline_score"].max(), long_df["post_score"].max()))
    ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color="#666")
    ax.set_title("Baseline versus post-attack opinion scores")
    ax.set_xlabel("Baseline score")
    ax.set_ylabel("Post-attack score")
    return _save_figure(fig, base_path)


def _draw_profile_heatmap(long_df: pd.DataFrame, profile_index_df: pd.DataFrame, base_path: Path) -> List[str]:
    matrix = long_df.pivot_table(index="profile_id", columns="opinion_leaf_label", values="abs_delta_score", aggfunc="mean")
    if not profile_index_df.empty:
        ordered_ids = [profile_id for profile_id in profile_index_df["profile_id"].tolist() if profile_id in matrix.index]
        matrix = matrix.loc[ordered_ids]
    fig, ax = plt.subplots(figsize=(11.2, max(6.2, 0.22 * len(matrix.index) + 2.2)))
    sns.heatmap(matrix, cmap="YlOrRd", linewidths=0.3, linecolor="white", cbar_kws={"label": "|Delta|"}, ax=ax)
    ax.set_title("Per-profile attack effectivity heatmap")
    ax.set_xlabel("Opinion leaf")
    ax.set_ylabel("Profile")
    return _save_figure(fig, base_path)


def _draw_susceptibility_distribution(profile_index_df: pd.DataFrame, base_path: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    sns.histplot(profile_index_df["susceptibility_index_pct"], bins=12, color=PALETTE["coral"], edgecolor="white", ax=ax)
    ax.set_title("Distribution of post hoc empirical susceptibility index")
    ax.set_xlabel("Susceptibility index percentile")
    ax.set_ylabel("Profiles")
    return _save_figure(fig, base_path)


def _ontology_table(ontology_catalog: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    selected_attack = ontology_catalog.get("selected_attack_leaf")
    if selected_attack:
        rows.append({"Ontology": "ATTACK", "Role": "Fixed pilot leaf", "Leaf": selected_attack})
    for leaf in ontology_catalog.get("selected_opinion_leaves", []):
        rows.append({"Ontology": "OPINION", "Role": "Repeated indicator leaf", "Leaf": leaf})
    return pd.DataFrame(rows)


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
    figures_dir = ensure_dir(Path(output_root) / "figures")
    tables_dir = ensure_dir(Path(output_root) / "tables")
    snapshots_dir = ensure_dir(Path(output_root) / "snapshots")
    report_assets_root = ensure_dir(report_assets_root)
    report_figures_dir = ensure_dir(Path(report_assets_root) / "figures")
    report_tables_dir = ensure_dir(Path(report_assets_root) / "tables")

    long_df = pd.read_csv(sem_long_csv_path)
    sem_result = json.loads(Path(sem_result_json_path).read_text(encoding="utf-8"))
    ols_params = pd.read_csv(ols_params_csv_path)
    bootstrap_params = pd.read_csv(bootstrap_params_csv_path)
    exploratory_df = pd.read_csv(exploratory_comparison_csv_path)
    config = json.loads(Path(config_json_path).read_text(encoding="utf-8"))
    ontology_catalog = json.loads(Path(ontology_catalog_path).read_text(encoding="utf-8"))
    assumptions = json.loads(Path(assumptions_json_path).read_text(encoding="utf-8"))
    critiques = json.loads(Path(critiques_json_path).read_text(encoding="utf-8"))

    stage05_dir = Path(sem_long_csv_path).resolve().parent
    stage06_dir = Path(sem_result_json_path).resolve().parent
    profile_summary_df = pd.read_csv(stage05_dir / "profile_level_effectivity.csv")
    profile_wide_df = pd.read_csv(stage05_dir / "profile_sem_wide.csv")
    profile_index_df = pd.read_csv(stage06_dir / "profile_susceptibility_index.csv")
    weight_table_path = stage06_dir / "moderator_weight_table.csv"
    weight_df = pd.read_csv(weight_table_path) if weight_table_path.exists() else pd.DataFrame()

    figure_files: List[str] = []
    table_files: List[str] = []
    snapshot_files: List[str] = []

    indicator_columns = [column for column in profile_wide_df.columns if column.startswith("abs_delta_indicator__") and not column.endswith("_z")]

    figure_files.extend(_draw_study_design(figures_dir / "figure_1_study_design", config))
    figure_files.extend(_draw_abs_delta_distribution(long_df, figures_dir / "figure_2_absolute_delta_distribution"))
    if not weight_df.empty:
        figure_files.extend(_draw_moderator_forest(weight_df, figures_dir / "figure_3_profile_moderator_coefficient_forest"))
    figure_files.extend(_draw_sem_diagram(sem_result, exploratory_df, config, indicator_columns, figures_dir / "figure_4_annotated_sem_path_diagram"))
    figure_files.extend(_draw_baseline_post_scatter(long_df, figures_dir / "supplementary_figure_s1_baseline_post_scatter"))
    figure_files.extend(_draw_profile_heatmap(long_df, profile_index_df, figures_dir / "supplementary_figure_s2_profile_effectivity_heatmap"))
    figure_files.extend(_draw_susceptibility_distribution(profile_index_df, figures_dir / "supplementary_figure_s3_susceptibility_distribution"))

    sem_coeff_df = pd.DataFrame(sem_result.get("coefficients", []))
    if not sem_coeff_df.empty:
        sem_coeff_df.to_csv(snapshots_dir / "sem_coefficients_snapshot.csv", index=False)
        snapshot_files.append(abs_path(snapshots_dir / "sem_coefficients_snapshot.csv"))

    config_table = pd.DataFrame(
        [
            {"Parameter": "Run ID", "Value": run_id},
            {"Parameter": "Paper title", "Value": paper_title},
            {"Parameter": "Profiles", "Value": config.get("n_profiles") or profile_summary_df["profile_id"].nunique()},
            {"Parameter": "Attacked rows", "Value": len(long_df)},
            {"Parameter": "Attack leaf", "Value": config.get("attack_leaf")},
            {"Parameter": "Opinion domain", "Value": config.get("focus_opinion_domain") or "Multiple"},
            {"Parameter": "Repeated opinion leaves", "Value": long_df["opinion_leaf_label"].nunique()},
            {"Parameter": "Model", "Value": config.get("openrouter_model")},
        ]
    )
    table_files.extend(
        _write_table_bundle(
            config_table,
            tables_dir / "table_1_pilot_design_and_configuration",
            "Pilot run design and configuration for the attacked-only profile-panel study.",
            "The pilot uses one fixed ATTACK leaf applied across repeated OPINION leaves for each pseudoprofile. All rows are attacked; there is no no-attack control condition in run_6.",
            "tab:design",
        )
    )

    descriptive_table = (
        long_df.groupby("opinion_leaf_label", as_index=False)
        .agg(
            n_rows=("scenario_id", "count"),
            mean_baseline=("baseline_score", "mean"),
            mean_post=("post_score", "mean"),
            mean_signed_delta=("delta_score", "mean"),
            mean_abs_delta=("abs_delta_score", "mean"),
            sd_abs_delta=("abs_delta_score", lambda s: float(s.std(ddof=0))),
        )
        .sort_values("mean_abs_delta", ascending=False)
    )
    table_files.extend(
        _write_table_bundle(
            descriptive_table,
            tables_dir / "table_2_attacked_effectivity_descriptive_statistics",
            "Attacked effectivity descriptive statistics by repeated opinion leaf.",
            "Absolute shift is the primary effectivity metric for run_6 because the same fixed attack leaf is linked to multiple opinion deltas that can move in different signed directions.",
            "tab:descriptives",
        )
    )

    sem_path_df = pd.DataFrame(sem_result.get("coefficients", []))
    sem_path_df = sem_path_df.loc[sem_path_df["op"].astype(str) == "~"].copy() if not sem_path_df.empty else pd.DataFrame()
    if not sem_path_df.empty:
        sem_path_df["estimate"] = pd.to_numeric(sem_path_df["estimate"], errors="coerce")
        sem_path_df["p_value"] = pd.to_numeric(sem_path_df["p_value"], errors="coerce")
        sem_summary = (
            sem_path_df.groupby("rhs", as_index=False)
            .agg(
                mean_sem_b=("estimate", "mean"),
                mean_abs_sem_b=("estimate", lambda s: float(np.mean(np.abs(s)))),
                min_sem_p=("p_value", "min"),
                n_sem_paths=("lhs", "count"),
            )
            .rename(columns={"rhs": "Moderator"})
        )
    else:
        sem_summary = pd.DataFrame(columns=["Moderator", "mean_sem_b", "mean_abs_sem_b", "min_sem_p", "n_sem_paths"])

    model_table = exploratory_df.rename(
        columns={
            "moderator_label": "Moderator",
            "univariate_estimate": "Controlled mean b",
            "univariate_p_value": "Controlled p",
            "ridge_mean_estimate": "Ridge mean b",
            "normalized_weight_pct": "Weight %",
            "role": "Role",
        }
    )[["Moderator", "Role", "Controlled mean b", "Controlled p", "Ridge mean b", "Weight %"]]
    model_table = model_table.merge(sem_summary, on="Moderator", how="left")
    model_table = model_table.rename(
        columns={
            "mean_sem_b": "SEM mean b",
            "mean_abs_sem_b": "SEM mean |b|",
            "min_sem_p": "SEM min p",
            "n_sem_paths": "SEM paths",
        }
    )
    table_files.extend(
        _write_table_bundle(
            model_table,
            tables_dir / "table_3_multivariate_profile_moderator_model",
            "Profile moderators of attacked effectivity: controlled contrasts, SEM paths, and descriptive susceptibility weights.",
            "Controlled coefficients summarize moderator associations with mean attacked effectivity. SEM columns summarize the repeated-outcome path model across the attacked opinion leaves. Weight percentages come from the target-conditional regularized aggregation used to compute the post hoc susceptibility index.",
            "tab:multivariate",
        )
    )

    table_files.extend(
        _write_table_bundle(
            _ontology_table(ontology_catalog),
            tables_dir / "supplementary_table_s1_ontology_leaves_used",
            "Ontology leaves used in the run_6 pilot.",
            "Only ontology leaves are sampled for estimation. The ATTACK ontology contributes one fixed pilot leaf; the OPINION ontology contributes repeated indicator leaves.",
            "tab:ontologies",
        )
    )

    moderator_table = exploratory_df.rename(
        columns={
            "moderator_label": "Moderator",
            "multivariate_estimate": "Multivariate b",
            "multivariate_p_value": "Multivariate p",
            "univariate_estimate": "Univariate b",
            "univariate_p_value": "Univariate p",
            "role": "Role",
        }
    )[["Moderator", "Role", "Multivariate b", "Multivariate p", "Univariate b", "Univariate p"]]
    if not weight_df.empty:
        moderator_table = moderator_table.merge(
            weight_df.rename(
                columns={
                    "moderator_label": "Moderator",
                    "ontology_group": "Ontology group",
                    "normalized_weight_pct": "Normalized weight %",
                }
            )[["Moderator", "Ontology group", "Normalized weight %"]],
            on="Moderator",
            how="left",
        )
    table_files.extend(
        _write_table_bundle(
            moderator_table,
            tables_dir / "supplementary_table_s2_moderator_comparison",
            "Core and exploratory profile moderator comparison.",
            "Core terms are entered into the latent SEM or primary multivariate profile model. Normalized weight percentages summarize each moderator's share of the total fitted importance after accounting for the moderator's observed variability.",
            "tab:moderators",
        )
    )

    if not sem_path_df.empty:
        sem_table = sem_path_df.rename(
            columns={
                "lhs": "Outcome leaf",
                "rhs": "Moderator",
                "estimate": "SEM b",
                "std_error": "SE",
                "p_value": "p",
            }
        )[["Outcome leaf", "Moderator", "SEM b", "SE", "p"]]
        sem_table["Outcome leaf"] = sem_table["Outcome leaf"].map(_pretty)
        table_files.extend(
            _write_table_bundle(
                sem_table,
                tables_dir / "supplementary_table_s5_sem_path_coefficients",
                "Leaf-specific path-SEM coefficients.",
                "Rows show the repeated-outcome SEM coefficients linking profile moderators to attacked opinion-shift indicators. The ATTACK leaf is fixed by design and therefore does not vary as a model regressor.",
                "tab:sempaths",
            )
        )

    risk_rows = []
    for item in assumptions:
        risk_rows.append({"Type": "Assumption", "Item": item["assumption"], "Status": item["status"], "Mitigation": item["mitigation"]})
    for item in critiques:
        risk_rows.append({"Type": "Critique", "Item": item["critique"], "Status": "addressed", "Mitigation": item["implemented_change"]})
    table_files.extend(
        _write_table_bundle(
            pd.DataFrame(risk_rows),
            tables_dir / "supplementary_table_s3_assumption_and_risk_register",
            "Assumption register and peer-review risk register.",
            "The pilot is methodological and exploratory. Risks are surfaced explicitly so later scale-up runs can target the current design bottlenecks.",
            "tab:risks",
        )
    )

    reproducibility_rows = pd.DataFrame(
        [{"Field": key, "Value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in config.items()]
    )
    table_files.extend(
        _write_table_bundle(
            reproducibility_rows,
            tables_dir / "supplementary_table_s4_reproducibility_manifest",
            "Reproducibility manifest for run_6.",
            "The manifest captures the full pipeline configuration used to generate the pilot outputs and manuscript assets.",
            "tab:repro",
        )
    )

    long_df.to_csv(snapshots_dir / "sem_long_encoded_snapshot.csv", index=False)
    profile_summary_df.to_csv(snapshots_dir / "profile_level_effectivity_snapshot.csv", index=False)
    profile_index_df.to_csv(snapshots_dir / "profile_susceptibility_index_snapshot.csv", index=False)
    if not weight_df.empty:
        weight_df.to_csv(snapshots_dir / "moderator_weight_table_snapshot.csv", index=False)
    snapshot_files.extend(
        [
            abs_path(snapshots_dir / "sem_long_encoded_snapshot.csv"),
            abs_path(snapshots_dir / "profile_level_effectivity_snapshot.csv"),
            abs_path(snapshots_dir / "profile_susceptibility_index_snapshot.csv"),
        ]
    )
    if not weight_df.empty:
        snapshot_files.append(abs_path(snapshots_dir / "moderator_weight_table_snapshot.csv"))

    copied_figures = _copy_tree_contents(figures_dir, report_figures_dir)
    copied_tables = _copy_tree_contents(tables_dir, report_tables_dir)

    manifest_payload = {
        "run_id": run_id,
        "paper_title": paper_title,
        "figure_count": len(figure_files),
        "table_count": len(table_files),
        "copied_figures": copied_figures,
        "copied_tables": copied_tables,
    }
    manifest_path = Path(output_root) / "publication_assets_manifest.json"
    write_json(manifest_path, manifest_payload)

    return {
        "manifest_path": abs_path(manifest_path),
        "visual_files": figure_files,
        "table_files": table_files,
        "snapshot_files": snapshot_files,
        "copied_figures": copied_figures,
        "copied_tables": copied_tables,
    }
