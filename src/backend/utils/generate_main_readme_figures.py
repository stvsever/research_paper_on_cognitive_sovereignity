"""
Generate the two main README / paper matrix figures with top and right dendrograms.

The implementation is deliberately data-driven:
- attack / opinion labels are resolved from the actual run artefacts
- configured ontology order is respected when available
- canonical run_10 Big Five + Sex moderators are used when present
- figure 2 degrades gracefully if those moderators are absent in a custom run

These figures are used by the root README and the manually curated paper
(`research_report/report/main.tex`) and can also be generated from stage 08.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE05 = ROOT / "evaluation" / "run_10" / "stage_outputs" / "05_compute_effectivity_deltas"
DEFAULT_STAGE06 = ROOT / "evaluation" / "run_10" / "stage_outputs" / "06_construct_structural_equation_model"
DEFAULT_CONFIG = ROOT / "evaluation" / "run_10" / "config" / "pipeline_config.json"
DEFAULT_ONTOLOGY_CATALOG = ROOT / "evaluation" / "run_10" / "stage_outputs" / "01_create_scenarios" / "ontology_leaf_catalog.json"

DEFAULT_OUTPUT_DIRS = [
    ROOT / "research_report" / "assets" / "figures",
    ROOT / "evaluation" / "run_10" / "publication_assets" / "figures",
    ROOT / "evaluation" / "run_10" / "paper" / "publication_assets" / "figures",
    ROOT / "evaluation" / "run_10" / "stage_outputs" / "08_generate_publication_assets" / "figures",
]

WHITE = "#ffffff"
NAVY = "#14213d"
INK = "#222222"

CANONICAL_PREDICTORS: list[tuple[str, str]] = [
    ("profile_cont_big_five_conscientiousness_mean_pct", "Conscientiousness"),
    ("profile_cont_big_five_neuroticism_mean_pct", "Neuroticism"),
    ("profile_cont_big_five_openness_to_experience_mean_pct", "Openness to Exp."),
    ("profile_cont_big_five_agreeableness_mean_pct", "Agreeableness"),
    ("profile_cont_big_five_extraversion_mean_pct", "Extraversion"),
    ("profile_cat__profile_cat_sex_Female", "Sex: Female"),
    ("profile_cat__profile_cat_sex_Other", "Sex: Other"),
]


def _setup() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "axes.edgecolor": NAVY,
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def _normalize_output_dirs(output_dirs: Path | str | Sequence[Path | str]) -> list[Path]:
    if isinstance(output_dirs, (str, Path)):
        return [Path(output_dirs)]
    return [Path(path) for path in output_dirs]


def _save(fig: plt.Figure, stem: str, output_dirs: Path | str | Sequence[Path | str]) -> list[str]:
    saved: list[str] = []
    for out_dir in _normalize_output_dirs(output_dirs):
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in ("png", "pdf"):
            path = out_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE)
            saved.append(str(path.resolve()))
            print(f"  saved {path}")
    plt.close(fig)
    return saved


def _last_leaf(value: str) -> str:
    return str(value).split(" > ")[-1].strip()


def _display_text(value: str) -> str:
    words = _last_leaf(value).replace("_", " ").split()
    if not words:
        return str(value)
    text = " ".join(word if word.isupper() else word.capitalize() for word in words)
    replacements = {
        "Llm": "LLM",
        "Nato": "NATO",
        "Eu": "EU",
        "Us": "US",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _wrap_label(value: str, width: int = 14) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def _cluster_label_from_value(value: str, width: int) -> str:
    return _wrap_label(_display_text(value), width=width)


def _load_json_if_exists(path: str | Path | None) -> dict | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _ordered_labels(
    discovered_values: Iterable[str],
    preferred_values: Iterable[str] | None = None,
) -> list[str]:
    discovered = []
    seen = set()
    for value in discovered_values:
        if pd.isna(value):
            continue
        text = str(value)
        if text not in seen:
            discovered.append(text)
            seen.add(text)

    ordered: list[str] = []
    if preferred_values is not None:
        for value in preferred_values:
            last = _last_leaf(str(value))
            if last in seen and last not in ordered:
                ordered.append(last)

    for value in discovered:
        if value not in ordered:
            ordered.append(value)
    return ordered


def _make_cluster(mat: np.ndarray, axis: int) -> tuple[np.ndarray | None, list[int]]:
    work = np.nan_to_num(mat.T if axis == 1 else mat, nan=0.0)
    n = work.shape[0]
    if n <= 1:
        return None, list(range(n))
    dist = pdist(work, metric="euclidean")
    if len(dist) == 0 or np.allclose(dist, 0):
        return None, list(range(n))
    Z = optimal_leaf_ordering(linkage(dist, method="ward"), dist)
    return Z, list(range(n))


def _draw_dend(ax: plt.Axes, Z: np.ndarray | None, orientation: str, n_leaves: int, color: str = "#8d8d8d") -> list[int]:
    if Z is None:
        ax.set_axis_off()
        return list(range(n_leaves))
    dend = dendrogram(
        Z,
        ax=ax,
        orientation=orientation,
        color_threshold=0,
        above_threshold_color=color,
        no_labels=True,
    )
    ax.set_axis_off()
    return dend["leaves"]


def _imshow_heat(
    ax: plt.Axes,
    data: np.ndarray,
    cmap: str,
    norm_or_vmax,
    extent: tuple[float, float, float, float],
    annot: np.ndarray | None = None,
):
    n_rows, n_cols = data.shape
    kwargs = dict(aspect="auto", extent=extent, interpolation="nearest")
    if isinstance(norm_or_vmax, (int, float)):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=norm_or_vmax, **kwargs)
        vmax_color = float(norm_or_vmax) if norm_or_vmax else 1.0
    else:
        im = ax.imshow(data, cmap=cmap, norm=norm_or_vmax, **kwargs)
        vmax_color = float(getattr(norm_or_vmax, "vmax", 1.0) or 1.0)

    for i in range(1, n_cols):
        ax.axvline(x=i * 10, color="white", linewidth=0.55)
    for i in range(1, n_rows):
        ax.axhline(y=i * 10, color="white", linewidth=0.55)

    if annot is not None:
        for row in range(n_rows):
            for col in range(n_cols):
                text = annot[row, col]
                if not text:
                    continue
                cell_value = data[row, col]
                brightness = abs(float(cell_value)) / max(vmax_color, 1e-9)
                color = "white" if brightness > 0.55 else INK
                ax.text(
                    col * 10 + 5,
                    row * 10 + 5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color=color,
                )
    return im


def _align_dendrogram_axes(ax_heat: plt.Axes, ax_dend_top: plt.Axes, ax_dend_right: plt.Axes) -> None:
    fig = ax_heat.figure
    fig.canvas.draw()
    heat_pos = ax_heat.get_position()
    top_pos = ax_dend_top.get_position()
    right_pos = ax_dend_right.get_position()
    ax_dend_top.set_position([heat_pos.x0, top_pos.y0, heat_pos.width, top_pos.height])
    ax_dend_right.set_position([right_pos.x0, heat_pos.y0, right_pos.width, heat_pos.height])


def _resolve_attack_order(long_df: pd.DataFrame, config: dict | None, ontology_catalog: dict | None) -> list[str]:
    discovered = long_df["attack_leaf_label"].dropna().tolist()
    preferred: list[str] = []

    if ontology_catalog and ontology_catalog.get("selected_attack_leaves"):
        preferred.extend(ontology_catalog["selected_attack_leaves"])
    elif config and config.get("attack_leaves"):
        preferred.extend([part.strip() for part in str(config["attack_leaves"]).split(",") if part.strip()])
    elif config and config.get("attack_leaf"):
        preferred.append(config["attack_leaf"])

    return _ordered_labels(discovered, preferred)


def _resolve_opinion_order(long_df: pd.DataFrame, ontology_catalog: dict | None) -> list[str]:
    discovered = long_df["opinion_leaf_label"].dropna().tolist()
    preferred = ontology_catalog.get("selected_opinion_leaves") if ontology_catalog else None
    return _ordered_labels(discovered, preferred)


def _available_predictors(profile_wide_df: pd.DataFrame) -> list[tuple[str, str]]:
    available = [(column, label) for column, label in CANONICAL_PREDICTORS if column in profile_wide_df.columns]
    return available


def _make_figure_1(
    long_df: pd.DataFrame,
    output_dirs: Path | str | Sequence[Path | str],
    config: dict | None,
    ontology_catalog: dict | None,
) -> list[str]:
    attack_order = _resolve_attack_order(long_df, config, ontology_catalog)
    opinion_order = _resolve_opinion_order(long_df, ontology_catalog)

    mean_piv = (
        long_df.groupby(["attack_leaf_label", "opinion_leaf_label"])["adversarial_effectivity"]
        .mean()
        .reset_index()
        .pivot(index="attack_leaf_label", columns="opinion_leaf_label", values="adversarial_effectivity")
        .reindex(index=attack_order, columns=opinion_order)
    )
    std_piv = (
        long_df.groupby(["attack_leaf_label", "opinion_leaf_label"])["adversarial_effectivity"]
        .std()
        .reset_index()
        .pivot(index="attack_leaf_label", columns="opinion_leaf_label", values="adversarial_effectivity")
        .reindex(index=attack_order, columns=opinion_order)
    )

    mean_vals = mean_piv.values.astype(float)
    if mean_vals.size == 0:
        return []

    n_rows, n_cols = mean_piv.shape
    Z_col, _ = _make_cluster(mean_vals, axis=1)
    Z_row, _ = _make_cluster(mean_vals, axis=0)

    fig = plt.figure(figsize=(22, 6.8))
    fig.patch.set_facecolor(WHITE)
    gs = gridspec.GridSpec(
        3,
        5,
        figure=fig,
        height_ratios=[0.18, 1.0, 0.10],
        width_ratios=[1.0, 0.05, 1.0, 0.04, 0.14],
        hspace=0.0,
        wspace=0.0,
    )

    ax_dend_top = fig.add_subplot(gs[0, 0])
    ax_heat_mean = fig.add_subplot(gs[1, 0])
    ax_heat_std = fig.add_subplot(gs[1, 2])
    ax_dend_right = fig.add_subplot(gs[1, 4])

    col_order = _draw_dend(ax_dend_top, Z_col, orientation="top", n_leaves=n_cols)
    row_order = _draw_dend(ax_dend_right, Z_row, orientation="right", n_leaves=n_rows)

    mean_ordered = mean_piv.iloc[row_order, col_order].values.astype(float)
    std_ordered = std_piv.iloc[row_order, col_order].values.astype(float)
    col_labels = [_cluster_label_from_value(opinion_order[idx], width=14) for idx in col_order]
    row_labels = [_cluster_label_from_value(attack_order[idx], width=12) for idx in row_order]

    extent = (0, n_cols * 10, n_rows * 10, 0)
    x_ticks = np.arange(n_cols) * 10 + 5
    y_ticks = np.arange(n_rows) * 10 + 5
    run_id = str((config or {}).get("run_id", "run"))
    n_profiles = int(long_df["profile_id"].nunique()) if "profile_id" in long_df.columns else 0
    n_scenarios = int(len(long_df))

    vmax_mean = max(float(np.nanmax(np.abs(mean_ordered))) * 1.05, 1.0)
    norm_mean = TwoSlopeNorm(vmin=-vmax_mean, vcenter=0.0, vmax=vmax_mean)
    annot_mean = np.array(
        [
            [f"{mean_ordered[row, col]:.1f}" if not np.isnan(mean_ordered[row, col]) else "" for col in range(n_cols)]
            for row in range(n_rows)
        ]
    )
    im_mean = _imshow_heat(ax_heat_mean, mean_ordered, "RdBu_r", norm_mean, extent, annot_mean)

    ax_heat_mean.set_xticks(x_ticks)
    ax_heat_mean.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8.5)
    ax_heat_mean.set_yticks(y_ticks)
    ax_heat_mean.set_yticklabels(row_labels, fontsize=9.5)
    ax_heat_mean.set_xlim(0, n_cols * 10)
    ax_heat_mean.set_ylim(n_rows * 10, 0)
    ax_heat_mean.tick_params(axis="both", which="both", length=0)
    ax_heat_mean.set_title("Mean Adversarial Effectivity (AE)\nby Attack × Opinion cell", fontsize=12, fontweight="bold", pad=8)
    ax_heat_mean.set_xlabel("Political opinion leaf", fontsize=10.5, labelpad=6)
    ax_heat_mean.set_ylabel("Attack vector", fontsize=10.5, labelpad=6)

    vmax_std = max(float(np.nanmax(std_ordered)) * 1.05, 1.0)
    annot_std = np.array(
        [
            [f"{std_ordered[row, col]:.1f}" if not np.isnan(std_ordered[row, col]) else "" for col in range(n_cols)]
            for row in range(n_rows)
        ]
    )
    im_std = _imshow_heat(ax_heat_std, std_ordered, "YlOrRd", vmax_std, extent, annot_std)

    ax_heat_std.set_xticks(x_ticks)
    ax_heat_std.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8.5)
    ax_heat_std.set_yticks(y_ticks)
    ax_heat_std.set_yticklabels([])
    ax_heat_std.set_xlim(0, n_cols * 10)
    ax_heat_std.set_ylim(n_rows * 10, 0)
    ax_heat_std.tick_params(axis="both", which="both", length=0)
    ax_heat_std.set_title("Inter-individual SD of AE\n(profile variability per cell)", fontsize=12, fontweight="bold", pad=8)
    ax_heat_std.set_xlabel("Political opinion leaf", fontsize=10.5, labelpad=6)

    ax_dend_top.set_xlim(0, n_cols * 10)
    ax_dend_top.set_ylim(bottom=0)
    ax_dend_right.set_ylim(n_rows * 10, 0)
    ax_dend_right.set_xlim(left=0)

    plt.tight_layout(rect=[0, 0.13, 1, 1])
    _align_dendrogram_axes(ax_heat_mean, ax_dend_top, ax_dend_right)

    pos_mean = ax_heat_mean.get_position()
    pos_std = ax_heat_std.get_position()
    cbar_ax1 = fig.add_axes([pos_mean.x0, 0.04, pos_mean.width, 0.03])
    cbar_ax2 = fig.add_axes([pos_std.x0, 0.04, pos_std.width, 0.03])

    cb1 = fig.colorbar(im_mean, cax=cbar_ax1, orientation="horizontal")
    cb1.set_label("Mean AE  (positive = attack succeeded)", fontsize=8.5)
    cb1.ax.tick_params(labelsize=8)
    cb2 = fig.colorbar(im_std, cax=cbar_ax2, orientation="horizontal")
    cb2.set_label("SD(AE) across profiles  (inter-individual spread)", fontsize=8.5)
    cb2.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Adversarial effectivity across the {n_rows}-attack × {n_cols}-opinion factorial"
        f"  ({run_id}  |  {n_profiles} profiles  |  {n_scenarios:,} scenarios)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "Top and right dendrograms show hierarchical clustering on the mean-AE matrix."
        "  AE = (post − baseline) × direction_k  |  direction_k in {+1, −1} from OPINION ontology",
        ha="center",
        fontsize=8.5,
        color="#666666",
    )

    return _save(fig, "figure_readme_1_ae_factorial", output_dirs)


def _make_figure_2(
    profile_wide_df: pd.DataFrame,
    long_df: pd.DataFrame,
    output_dirs: Path | str | Sequence[Path | str],
    config: dict | None,
    ontology_catalog: dict | None,
) -> list[str]:
    predictor_pairs = _available_predictors(profile_wide_df)
    if not predictor_pairs:
        print("  skipped figure_readme_2_moderation_heatmap: canonical Big Five/Sex predictors not available")
        return []

    opinion_order = _resolve_opinion_order(long_df, ontology_catalog)
    label_by_slug = {_last_leaf(label).lower(): _cluster_label_from_value(label, width=12) for label in opinion_order}

    outcome_cols = []
    outcome_labels = []
    for label in opinion_order:
        slug = _last_leaf(label).lower()
        column = f"adversarial_delta_indicator__{slug}"
        if column in profile_wide_df.columns:
            outcome_cols.append(column)
            outcome_labels.append(label_by_slug[slug])

    if not outcome_cols:
        print("  skipped figure_readme_2_moderation_heatmap: adversarial delta indicators not available")
        return []
    run_id = str((config or {}).get("run_id", "run"))
    n_profiles = int(profile_wide_df["profile_id"].nunique()) if "profile_id" in profile_wide_df.columns else int(len(profile_wide_df))

    X = profile_wide_df[[column for column, _ in predictor_pairs]].copy()
    for column, _label in predictor_pairs:
        if column.startswith("profile_cont_"):
            mu = X[column].mean()
            sd = X[column].std(ddof=1)
            if pd.notna(sd) and sd > 0:
                X[column] = (X[column] - mu) / sd

    coef_df = pd.DataFrame(index=[label for _, label in predictor_pairs], columns=outcome_labels, dtype=float)
    pval_df = pd.DataFrame(index=[label for _, label in predictor_pairs], columns=outcome_labels, dtype=float)

    Xmat = X.values.astype(float)
    n_obs = len(Xmat)
    X_design = np.column_stack([np.ones(n_obs), Xmat])
    n_params = X_design.shape[1]

    for outcome_col, outcome_label in zip(outcome_cols, outcome_labels):
        y = profile_wide_df[outcome_col].fillna(0).values.astype(float)
        coefs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        resid = y - X_design @ coefs
        sse = float((resid**2).sum())
        df_error = n_obs - n_params
        if df_error <= 0 or sse <= 0:
            continue
        mse = sse / df_error
        xtx_inv = np.linalg.pinv(X_design.T @ X_design)
        se = np.sqrt(mse * np.diag(xtx_inv))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_vals = np.divide(coefs, se, out=np.zeros_like(coefs), where=se > 0)
        p_vals = 2 * stats.t.sf(np.abs(t_vals), df=df_error)
        for idx, (_column, predictor_label) in enumerate(predictor_pairs):
            coef_df.loc[predictor_label, outcome_label] = coefs[idx + 1]
            pval_df.loc[predictor_label, outcome_label] = p_vals[idx + 1]

    heat_vals = coef_df.values.astype(float)
    n_rows, n_cols = heat_vals.shape
    if n_rows == 0 or n_cols == 0:
        return []

    Z_col, _ = _make_cluster(heat_vals, axis=1)
    Z_row, _ = _make_cluster(heat_vals, axis=0)

    fig = plt.figure(figsize=(16, 6.8))
    fig.patch.set_facecolor(WHITE)
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[0.20, 1.0, 0.10],
        width_ratios=[1.0, 0.16],
        hspace=0.0,
        wspace=0.0,
    )

    ax_dend_top = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_dend_right = fig.add_subplot(gs[1, 1])

    col_order = _draw_dend(ax_dend_top, Z_col, orientation="top", n_leaves=n_cols)
    row_order = _draw_dend(ax_dend_right, Z_row, orientation="right", n_leaves=n_rows)

    ordered_coef = coef_df.iloc[row_order, col_order]
    ordered_pval = pval_df.iloc[row_order, col_order]
    col_labels = [coef_df.columns[idx] for idx in col_order]
    row_labels = [coef_df.index[idx] for idx in row_order]

    annot = np.array(
        [
            [
                (
                    f"{ordered_coef.iloc[row, col]:.1f}"
                    + (
                        "***"
                        if ordered_pval.iloc[row, col] < 0.001
                        else "**"
                        if ordered_pval.iloc[row, col] < 0.01
                        else "*"
                        if ordered_pval.iloc[row, col] < 0.05
                        else "†"
                        if ordered_pval.iloc[row, col] < 0.10
                        else ""
                    )
                )
                if not pd.isna(ordered_coef.iloc[row, col])
                else ""
                for col in range(n_cols)
            ]
            for row in range(n_rows)
        ]
    )

    vmax = max(float(np.nanmax(np.abs(ordered_coef.values))) * 1.05, 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    extent = (0, n_cols * 10, n_rows * 10, 0)
    im = _imshow_heat(ax_heat, ordered_coef.values.astype(float), "RdBu_r", norm, extent, annot)

    x_ticks = np.arange(n_cols) * 10 + 5
    y_ticks = np.arange(n_rows) * 10 + 5
    ax_heat.set_xticks(x_ticks)
    ax_heat.set_xticklabels(col_labels, fontsize=9)
    ax_heat.set_yticks(y_ticks)
    ax_heat.set_yticklabels(row_labels, fontsize=10.5)
    ax_heat.set_xlim(0, n_cols * 10)
    ax_heat.set_ylim(n_rows * 10, 0)
    ax_heat.tick_params(axis="both", which="both", length=0)
    ax_heat.set_xlabel("Political opinion domain  (10 attacked leaves)", fontsize=11, labelpad=10)
    ax_heat.set_ylabel("Profile moderator", fontsize=11, labelpad=8)
    ax_heat.set_title(
        "Profile moderators → adversarial opinion shifts"
        f"  (OLS per opinion leaf, z-scored Big Five, {run_id}, N = {n_profiles} profiles)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )

    ax_dend_top.set_xlim(0, n_cols * 10)
    ax_dend_top.set_ylim(bottom=0)
    ax_dend_right.set_ylim(n_rows * 10, 0)
    ax_dend_right.set_xlim(left=0)

    plt.tight_layout(rect=[0, 0.14, 1, 1])
    _align_dendrogram_axes(ax_heat, ax_dend_top, ax_dend_right)

    pos = ax_heat.get_position()
    cbar_ax = fig.add_axes([pos.x0, 0.04, pos.width * 0.85, 0.03])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cb.set_label("OLS coefficient  (z-scored Big Five  |  unstd. Sex dummies)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.text(
        pos.x0,
        0.005,
        "Top and right dendrograms show hierarchical clustering of the moderator-coefficient matrix."
        "  Blue = moderator reduces susceptibility  |  Red = moderator increases susceptibility"
        "  |  †p < .10   *p < .05   **p < .01   ***p < .001",
        fontsize=8.5,
        color="#555555",
    )

    return _save(fig, "figure_readme_2_moderation_heatmap", output_dirs)


def generate_main_readme_figures(
    *,
    stage05_dir: str | Path,
    stage06_dir: str | Path,
    output_dirs: Path | str | Sequence[Path | str],
    config: dict | None = None,
    ontology_catalog: dict | None = None,
) -> list[str]:
    _setup()
    stage05_dir = Path(stage05_dir)
    stage06_dir = Path(stage06_dir)
    long_df = pd.read_csv(stage05_dir / "sem_long_raw.csv")
    profile_wide_df = pd.read_csv(stage06_dir / "profile_sem_wide.csv")

    saved: list[str] = []
    saved.extend(_make_figure_1(long_df, output_dirs, config, ontology_catalog))
    saved.extend(_make_figure_2(profile_wide_df, long_df, output_dirs, config, ontology_catalog))
    return saved


if __name__ == "__main__":
    config = _load_json_if_exists(DEFAULT_CONFIG)
    ontology_catalog = _load_json_if_exists(DEFAULT_ONTOLOGY_CATALOG)
    files = generate_main_readme_figures(
        stage05_dir=DEFAULT_STAGE05,
        stage06_dir=DEFAULT_STAGE06,
        output_dirs=DEFAULT_OUTPUT_DIRS,
        config=config,
        ontology_catalog=ontology_catalog,
    )
    print(f"Done. Wrote {len(files)} files.")
