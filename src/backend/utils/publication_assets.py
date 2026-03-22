from __future__ import annotations

import json
import os
import shutil
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

    table_latex = df.to_latex(index=False, escape=True, na_rep="", float_format=lambda x: f"{x:.3f}")
    tex_content = "\n".join(
        [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\begin{threeparttable}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\small",
            table_latex,
            f"\\begin{{tablenotes}}[flushleft]\\footnotesize\\item Note. {note}\\end{{tablenotes}}",
            "\\end{threeparttable}",
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


def _draw_pipeline_schematic(base_path: Path, paper_title: str) -> List[str]:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_axis_off()

    boxes = [
        (0.04, 0.55, 0.2, 0.22, "PROFILE\nSampling"),
        (0.29, 0.55, 0.2, 0.22, "Baseline\nOpinion"),
        (0.54, 0.55, 0.2, 0.22, "ATTACK / Control\nExposure"),
        (0.79, 0.55, 0.17, 0.22, "Post-attack\nOpinion"),
        (0.29, 0.15, 0.2, 0.18, "Delta\nConstruction"),
        (0.54, 0.15, 0.2, 0.18, "Moderation\nSEM + Robust OLS"),
    ]

    for x, y, w, h, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                linewidth=1.4,
                edgecolor=PALETTE["navy"],
                facecolor="#f7fbff",
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=PALETTE["navy"], fontweight="bold")

    arrows = [
        ((0.24, 0.66), (0.29, 0.66)),
        ((0.49, 0.66), (0.54, 0.66)),
        ((0.74, 0.66), (0.79, 0.66)),
        ((0.39, 0.55), (0.39, 0.33)),
        ((0.64, 0.55), (0.64, 0.33)),
        ((0.49, 0.24), (0.54, 0.24)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.3, color=PALETTE["blue"]))

    ax.text(
        0.5,
        0.94,
        "Pilot multi-agent simulation workflow",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=PALETTE["navy"],
    )
    ax.text(
        0.5,
        0.88,
        paper_title,
        ha="center",
        va="center",
        fontsize=9.5,
        color=PALETTE["ink"],
    )
    return _save_figure(fig, base_path)


def _draw_delta_distribution(df: pd.DataFrame, base_path: Path, attack_effect_p: float) -> List[str]:
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    plot_df = df.copy()
    plot_df["Condition"] = plot_df["attack_present"].map({0: "Control", 1: "Attack"})
    sns.violinplot(data=plot_df, x="Condition", y="delta_score", palette=[PALETTE["teal"], PALETTE["coral"]], inner=None, linewidth=1.1, cut=0, ax=ax)
    sns.stripplot(data=plot_df, x="Condition", y="delta_score", color=PALETTE["navy"], alpha=0.75, size=5.5, jitter=0.12, ax=ax)
    sns.pointplot(data=plot_df, x="Condition", y="delta_score", estimator=np.mean, errorbar=("ci", 95), color=PALETTE["sand"], join=False, scale=0.8, ax=ax)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=PALETTE["navy"], alpha=0.7)
    ax.set_title("Figure 2. Attack vs. control effectivity deltas")
    ax.set_xlabel("")
    ax.set_ylabel("Post - baseline opinion score")
    ax.text(
        0.02,
        0.97,
        f"Primary attack effect p = {attack_effect_p:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "boxstyle": "round,pad=0.3"},
    )
    return _save_figure(fig, base_path)


def _draw_interaction_plot(df: pd.DataFrame, base_path: Path, primary_moderator: str) -> List[str]:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    model = smf.ols(
        "delta_score ~ attack_present + baseline_score + primary_moderator_z + attack_x_primary_moderator",
        data=df,
    ).fit()

    moderator_raw = df["primary_moderator_value"].astype(float)
    raw_grid = np.linspace(float(moderator_raw.min()), float(moderator_raw.max()), 60)
    raw_mean = float(moderator_raw.mean())
    raw_std = float(moderator_raw.std(ddof=0))
    if raw_std == 0.0:
        raw_std = 1.0
    z_grid = (raw_grid - raw_mean) / raw_std
    baseline_mean = float(df["baseline_score"].mean())

    for attack_present, color, label in [
        (0, PALETTE["teal"], "Control"),
        (1, PALETTE["coral"], "Attack"),
    ]:
        grid_df = pd.DataFrame(
            {
                "attack_present": attack_present,
                "baseline_score": baseline_mean,
                "primary_moderator_z": z_grid,
                "attack_x_primary_moderator": attack_present * z_grid,
            }
        )
        prediction = model.get_prediction(grid_df).summary_frame(alpha=0.05)
        ax.plot(raw_grid, prediction["mean"], color=color, linewidth=2.5, label=label)
        ax.fill_between(raw_grid, prediction["mean_ci_lower"], prediction["mean_ci_upper"], color=color, alpha=0.16)

    ax.set_title("Figure 3. Primary moderation over susceptibility")
    ax.set_xlabel(_pretty_column(primary_moderator))
    ax.set_ylabel("Predicted delta score")
    ax.legend(loc="best")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color=PALETTE["navy"], alpha=0.7)
    return _save_figure(fig, base_path)


def _draw_coefficient_forest(exploratory_df: pd.DataFrame, base_path: Path) -> List[str]:
    if exploratory_df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        ax.set_axis_off()
        ax.text(0.5, 0.6, "No moderator coefficients available", ha="center", va="center", fontsize=14, fontweight="bold")
        ax.text(0.5, 0.4, "The pilot did not yield enough variability for exploratory comparison.", ha="center", va="center", fontsize=10)
        return _save_figure(fig, base_path)

    fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.6 * max(4, len(exploratory_df)))))
    forest_df = exploratory_df.copy()
    forest_df = forest_df.sort_values("interaction_estimate")
    y_pos = np.arange(len(forest_df))
    colors = [PALETTE["coral"] if role == "primary" else PALETTE["blue"] for role in forest_df["role"]]
    ax.errorbar(
        forest_df["interaction_estimate"],
        y_pos,
        xerr=[
            forest_df["interaction_estimate"] - forest_df["interaction_conf_low"],
            forest_df["interaction_conf_high"] - forest_df["interaction_estimate"],
        ],
        fmt="none",
        ecolor=PALETTE["navy"],
        elinewidth=1.3,
        capsize=3,
    )
    ax.scatter(forest_df["interaction_estimate"], y_pos, c=colors, s=68, zorder=3)
    ax.axvline(0.0, linestyle="--", linewidth=1.0, color=PALETTE["navy"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(forest_df["moderator_label"])
    ax.set_xlabel("Interaction coefficient (attack x moderator)")
    ax.set_title("Figure 4. Primary and exploratory moderation effects")
    for y_idx, row in enumerate(forest_df.to_dict(orient="records")):
        if row["interaction_p_value"] < 0.05:
            ax.text(row["interaction_conf_high"], y_idx, " *", va="center", ha="left", color=PALETTE["ink"], fontsize=11)
    return _save_figure(fig, base_path)


def _draw_baseline_post_scatter(df: pd.DataFrame, base_path: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    plot_df = df.copy()
    plot_df["Condition"] = plot_df["attack_present"].map({0: "Control", 1: "Attack"})
    sns.scatterplot(
        data=plot_df,
        x="baseline_score",
        y="post_score",
        hue="Condition",
        palette={"Control": PALETTE["teal"], "Attack": PALETTE["coral"]},
        s=70,
        ax=ax,
    )
    min_axis = float(min(plot_df["baseline_score"].min(), plot_df["post_score"].min()))
    max_axis = float(max(plot_df["baseline_score"].max(), plot_df["post_score"].max()))
    ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color=PALETTE["navy"], linewidth=1.0)
    ax.set_title("Supplementary Figure S1. Baseline vs. post-attack scores")
    ax.set_xlabel("Baseline score")
    ax.set_ylabel("Post score")
    return _save_figure(fig, base_path)


def _draw_attack_quality(df: pd.DataFrame, base_path: Path) -> List[str]:
    attack_df = df[df["attack_present"] == 1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    sns.histplot(attack_df["attack_realism_score"].dropna(), kde=True, color=PALETTE["coral"], ax=axes[0])
    axes[0].set_title("Attack realism scores")
    axes[0].set_xlabel("Realism score")
    sns.histplot(attack_df["attack_coherence_score"].dropna(), kde=True, color=PALETTE["blue"], ax=axes[1])
    axes[1].set_title("Attack coherence scores")
    axes[1].set_xlabel("Coherence score")
    fig.suptitle("Supplementary Figure S2. Exposure quality diagnostics", y=1.02, fontsize=13, fontweight="bold")
    return _save_figure(fig, base_path)


def _draw_scenario_composition(df: pd.DataFrame, base_path: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    comp_df = (
        df.assign(Condition=df["attack_present"].map({0: "Control", 1: "Attack"}))
        .groupby(["opinion_leaf_label", "Condition"], as_index=False)
        .size()
    )
    sns.barplot(data=comp_df, x="size", y="opinion_leaf_label", hue="Condition", palette=[PALETTE["teal"], PALETTE["coral"]], ax=ax)
    ax.set_title("Supplementary Figure S3. Scenario composition by opinion leaf")
    ax.set_xlabel("Number of scenarios")
    ax.set_ylabel("Opinion leaf")
    return _save_figure(fig, base_path)


def _draw_sem_overview(base_path: Path, primary_moderator: str) -> List[str]:
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_axis_off()
    boxes = [
        (0.06, 0.35, 0.2, 0.25, "PROFILE\n(primary moderator)"),
        (0.39, 0.62, 0.2, 0.2, "ATTACK\npresence"),
        (0.39, 0.15, 0.2, 0.2, "Baseline\nopinion"),
        (0.73, 0.38, 0.2, 0.25, "Delta / Post\nopinion"),
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

    for start, end, text in [
        ((0.26, 0.48), (0.39, 0.72), "moderates"),
        ((0.26, 0.48), (0.39, 0.25), "conditions"),
        ((0.59, 0.72), (0.73, 0.50), "effect"),
        ((0.59, 0.25), (0.73, 0.50), "controls"),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color=PALETTE["blue"]))
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.05, text, ha="center", va="center", fontsize=9.5)
    ax.text(0.5, 0.92, "Supplementary Figure S4. Primary moderation model overview", ha="center", va="center", fontsize=14, fontweight="bold", color=PALETTE["navy"])
    ax.text(0.5, 0.08, f"Primary moderator operationalized as {_pretty_column(primary_moderator)}.", ha="center", va="center", fontsize=9.5, color=PALETTE["ink"])
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

    primary_moderator = config.get("primary_moderator", "profile_cont_susceptibility_index")
    attack_row = ols_params.loc[ols_params["term"] == "attack_present"]
    attack_p = float(attack_row["p_value"].iloc[0]) if len(attack_row) else np.nan

    visual_files: List[str] = []
    table_files: List[str] = []

    visual_files.extend(_draw_pipeline_schematic(figures_dir / "figure_1_study_design", paper_title))
    visual_files.extend(_draw_delta_distribution(df, figures_dir / "figure_2_attack_control_delta_distribution", attack_p))
    visual_files.extend(_draw_interaction_plot(df, figures_dir / "figure_3_primary_moderation_interaction", primary_moderator))
    visual_files.extend(_draw_coefficient_forest(exploratory, figures_dir / "figure_4_moderator_coefficient_forest"))
    visual_files.extend(_draw_baseline_post_scatter(df, figures_dir / "supplementary_figure_s1_baseline_post_scatter"))
    visual_files.extend(_draw_attack_quality(df, figures_dir / "supplementary_figure_s2_attack_quality"))
    visual_files.extend(_draw_scenario_composition(df, figures_dir / "supplementary_figure_s3_scenario_composition"))
    visual_files.extend(_draw_sem_overview(figures_dir / "supplementary_figure_s4_sem_overview", primary_moderator))

    table_1 = pd.DataFrame(
        {
            "Parameter": [
                "Run identifier",
                "Paper title",
                "Scenarios",
                "Attack ratio",
                "Attack leaf",
                "OpenRouter model",
                "Primary moderator",
                "Bootstrap samples",
                "Self-supervised attack realism",
                "Test ontology",
            ],
            "Value": [
                run_id,
                paper_title,
                config.get("n_scenarios"),
                config.get("attack_ratio"),
                config.get("attack_leaf"),
                config.get("openrouter_model"),
                primary_moderator,
                config.get("bootstrap_samples"),
                config.get("self_supervise_attack_realism"),
                config.get("use_test_ontology"),
            ],
        }
    )
    table_files.extend(
        _write_table_bundle(
            table_1,
            tables_dir / "table_1_pilot_design_and_configuration",
            "Table 1. Pilot design and run configuration",
            "Run-level configuration for the pilot simulation reported in run 3.",
            "tab:pilot_design",
        )
    )

    table_2 = (
        df.assign(Condition=df["attack_present"].map({0: "Control", 1: "Attack"}))
        .groupby("Condition", as_index=False)
        .agg(
            n=("scenario_id", "count"),
            baseline_mean=("baseline_score", "mean"),
            post_mean=("post_score", "mean"),
            delta_mean=("delta_score", "mean"),
            delta_sd=("delta_score", "std"),
            susceptibility_mean=("primary_moderator_value", "mean"),
            realism_mean=("attack_realism_score", "mean"),
        )
    )
    table_files.extend(
        _write_table_bundle(
            table_2,
            tables_dir / "table_2_condition_descriptive_statistics",
            "Table 2. Descriptive statistics by condition",
            "Scores are reported on the high-resolution signed opinion scale from -1000 to +1000. Realism scores apply only to treated scenarios.",
            "tab:descriptives",
        )
    )

    table_3 = ols_params.merge(bootstrap_params, on="term", how="left")
    table_files.extend(
        _write_table_bundle(
            table_3,
            tables_dir / "table_3_primary_moderation_model",
            "Table 3. Primary moderation model with robust and bootstrapped estimates",
            "Robust standard errors are HC3. Bootstrap intervals are percentile intervals from resampled pilot fits and are exploratory given the small sample.",
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
            "n_scenarios": [int(df["attack_present"].sum())],
        }
    )
    supp_ontology = pd.concat(
        [
            opinion_counts.assign(ontology="OPINION").rename(columns={"opinion_leaf": "leaf_path", "opinion_leaf_label": "leaf_label"}),
            attack_leaf_counts.assign(ontology="ATTACK", leaf_label="Misleading narrative framing").rename(columns={"attack_leaf": "leaf_path"}),
        ],
        ignore_index=True,
        sort=False,
    )[["ontology", "leaf_label", "leaf_path", "n_scenarios"]]
    table_files.extend(
        _write_table_bundle(
            supp_ontology,
            tables_dir / "supplementary_table_s1_ontology_leaves_used",
            "Supplementary Table S1. Ontology leaves used in the pilot",
            "Opinion leaves reflect the scenario allocation in run 3; the attack ontology was restricted to one common misinformation vector for this pilot.",
            "tab:supp_ontology",
        )
    )

    table_files.extend(
        _write_table_bundle(
            exploratory,
            tables_dir / "supplementary_table_s2_exploratory_moderator_comparison",
            "Supplementary Table S2. Exploratory moderator comparison",
            "Each moderator was estimated in a separate attack-by-moderator interaction model with baseline opinion included as a covariate.",
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
            "Supplementary Table S3. Assumption and peer-review risk register",
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
            "Supplementary Table S4. Reproducibility manifest",
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
            "primary_moderator": primary_moderator,
            "visual_files": visual_files,
            "table_files": table_files,
            "report_asset_copies": {
                "figures": copied_figures,
                "tables": copied_tables,
            },
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
