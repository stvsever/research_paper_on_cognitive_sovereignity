from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from src.backend.utils.data_utils import infer_analysis_mode


def _significance_stars(p_value: float | None) -> str:
    if p_value is None or pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _pretty_term(term: str) -> str:
    label = term
    for prefix in ["profile_cont_", "profile_cat__profile_cat_", "profile_cat__", "profile_cat_"]:
        if label.startswith(prefix):
            label = label[len(prefix) :]
    label = label.replace("_z", "")
    label = label.replace("__", " ")
    label = label.replace("_", " ")
    return label.title()


def _save_figure_html(fig: go.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    return str(path)


def _render_dashboard_html(
    run_id: str,
    summary_cards: Dict[str, Any],
    figure_divs: List[Tuple[str, str]],
    notes: List[str],
) -> str:
    plotly_js = get_plotlyjs()
    cards_html = "\n".join(
        [
            (
                f"<div class='card'><div class='label'>{key}</div>"
                f"<div class='value'>{value}</div></div>"
            )
            for key, value in summary_cards.items()
        ]
    )

    nav_items = "\n".join(
        [f"<button class='tab-btn' data-tab='tab-{idx}'>{title}</button>" for idx, (title, _) in enumerate(figure_divs)]
    )

    tab_html_blocks: List[str] = []
    for idx, (title, div_html) in enumerate(figure_divs):
        active = "active" if idx == 0 else ""
        tab_html_blocks.append(
            (
                f"<section id='tab-{idx}' class='tab-panel {active}'>"
                f"<h2>{title}</h2>{div_html}</section>"
            )
        )

    notes_html = "\n".join([f"<li>{note}</li>" for note in notes])

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{run_id} Interactive Attack-Effectivity Dashboard</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #14213d;
      --ink-soft: #4a5d7a;
      --accent: #d95d39;
      --accent-2: #1d4e89;
      --ok: #2a9d8f;
      --line: #dbe3ef;
      --shadow: 0 10px 30px rgba(20, 33, 61, 0.12);
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 20% 0%, #e9f1ff 0%, var(--bg) 42%);
      color: var(--ink);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }}
    .wrap {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 22px 18px 30px;
    }}
    .hero {{
      background: linear-gradient(120deg, #14213d, #1f3b73);
      color: #fff;
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 22px 24px;
      margin-bottom: 16px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 1.5rem;
      letter-spacing: 0.02em;
    }}
    .hero p {{
      margin: 0;
      color: #d6e2ff;
      font-size: 0.95rem;
      max-width: 880px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin: 14px 0 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 13px;
      box-shadow: 0 6px 18px rgba(20,33,61,0.06);
    }}
    .label {{
      font-size: 0.78rem;
      color: var(--ink-soft);
      letter-spacing: 0.03em;
      text-transform: uppercase;
      margin-bottom: 3px;
    }}
    .value {{
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--ink);
    }}
    .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .tab-btn {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink-soft);
      border-radius: 9px;
      padding: 7px 12px;
      cursor: pointer;
      font-size: 0.86rem;
      font-weight: 600;
    }}
    .tab-btn.active {{
      color: #fff;
      background: var(--accent-2);
      border-color: var(--accent-2);
    }}
    .tab-panel {{
      display: none;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      margin-bottom: 12px;
      box-shadow: 0 10px 22px rgba(20,33,61,0.08);
    }}
    .tab-panel.active {{
      display: block;
    }}
    .tab-panel h2 {{
      margin: 2px 0 8px;
      font-size: 1.05rem;
      color: var(--ink);
    }}
    .notes {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 16px;
      box-shadow: 0 10px 22px rgba(20,33,61,0.07);
    }}
    .notes h3 {{
      margin-top: 0;
      font-size: 0.95rem;
    }}
    .notes li {{
      margin: 6px 0;
      color: var(--ink-soft);
      line-height: 1.4;
    }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Interactive Attack-Effectivity Investigation: {run_id}</h1>
      <p>Attacked-only profile-panel simulation: one fixed misinformation attack, repeated opinion leaves per profile, repeated-outcome SEM, and post hoc target-conditional susceptibility indexing.</p>
    </div>

    <div class="cards">{cards_html}</div>

    <div class="nav" id="tabs">{nav_items}</div>

    {''.join(tab_html_blocks)}

    <div class="notes">
      <h3>Methodological Notes</h3>
      <ul>{notes_html}</ul>
    </div>
  </div>

  <script>
    const buttons = Array.from(document.querySelectorAll('.tab-btn'));
    const panels = Array.from(document.querySelectorAll('.tab-panel'));
    function activateTab(id) {{
      buttons.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === id));
      panels.forEach(p => p.classList.toggle('active', p.id === id));
    }}
    buttons.forEach((btn) => btn.addEventListener('click', () => activateTab(btn.dataset.tab)));
    if (buttons.length > 0) {{
      activateTab(buttons[0].dataset.tab);
    }}
  </script>
</body>
</html>
""".strip()


def generate_research_visuals(
    sem_long_csv_path: str | Path,
    sem_result_json_path: str | Path,
    ols_params_csv_path: str | Path,
    output_dir: str | Path,
    run_id: str,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    figures_dir = output_root / "figures"
    data_snapshots_dir = output_root / "data_snapshots"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_snapshots_dir.mkdir(parents=True, exist_ok=True)

    long_df = pd.read_csv(sem_long_csv_path)
    sem_result = json.loads(Path(sem_result_json_path).read_text(encoding="utf-8"))
    ols_params = pd.read_csv(ols_params_csv_path)
    analysis_mode = infer_analysis_mode(long_df)

    stage05_dir = Path(sem_long_csv_path).resolve().parent
    stage06_dir = Path(sem_result_json_path).resolve().parent
    profile_summary_path = stage05_dir / "profile_level_effectivity.csv"
    profile_index_path = stage06_dir / "profile_susceptibility_index.csv"
    exploratory_path = stage06_dir / "exploratory_moderator_comparison.csv"
    weight_path = stage06_dir / "moderator_weight_table.csv"

    profile_df = pd.read_csv(profile_summary_path) if profile_summary_path.exists() else pd.DataFrame()
    profile_index_df = pd.read_csv(profile_index_path) if profile_index_path.exists() else pd.DataFrame()
    exploratory_df = pd.read_csv(exploratory_path) if exploratory_path.exists() else pd.DataFrame()
    weight_df = pd.read_csv(weight_path) if weight_path.exists() else pd.DataFrame()

    fit_indices = sem_result.get("fit_indices", {})
    summary_cards = {
        "Profiles": int(long_df["profile_id"].nunique()) if "profile_id" in long_df.columns else len(profile_df),
        "Attacked Rows": len(long_df),
        "Opinion Leaves": int(long_df["opinion_leaf"].nunique()) if "opinion_leaf" in long_df.columns else "n/a",
        "Mean |Delta|": f"{long_df['abs_delta_score'].mean():.2f}" if "abs_delta_score" in long_df.columns else "n/a",
        "Mean Signed Delta": f"{long_df['delta_score'].mean():.2f}" if "delta_score" in long_df.columns else "n/a",
        "Mean Realism": (
            f"{long_df['attack_realism_score'].dropna().mean():.2f}"
            if "attack_realism_score" in long_df.columns and len(long_df["attack_realism_score"].dropna())
            else "n/a"
        ),
        "CFI": f"{fit_indices.get('CFI', float('nan')):.3f}" if fit_indices.get("CFI") is not None else "n/a",
        "RMSEA": f"{fit_indices.get('RMSEA', float('nan')):.3f}" if fit_indices.get("RMSEA") is not None else "n/a",
    }

    fig_abs_delta = px.box(
        long_df,
        x="opinion_leaf_label",
        y="abs_delta_score",
        points="all",
        color="opinion_leaf_label",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Absolute attacked opinion shift by opinion leaf",
    )
    fig_abs_delta.update_layout(
        template="plotly_white",
        xaxis_title="Opinion leaf",
        yaxis_title="Absolute post-baseline shift",
        showlegend=False,
    )
    fig_abs_delta.update_xaxes(tickangle=-25)

    figure_divs: List[Tuple[str, str]] = []
    visual_files: List[str] = []

    fig_abs_delta_path = figures_dir / "absolute_delta_by_leaf.html"
    visual_files.append(_save_figure_html(fig_abs_delta, fig_abs_delta_path))
    figure_divs.append(("Absolute Delta by Leaf", fig_abs_delta.to_html(include_plotlyjs=False, full_html=False)))

    if not profile_index_df.empty:
        heatmap_df = long_df.merge(
            profile_index_df[["profile_id", "susceptibility_index_pct"]],
            on="profile_id",
            how="left",
        )
        heatmap_df = heatmap_df.sort_values(["susceptibility_index_pct", "profile_id", "opinion_leaf_label"], ascending=[False, True, True])
        matrix = heatmap_df.pivot_table(
            index="profile_id",
            columns="opinion_leaf_label",
            values="abs_delta_score",
            aggfunc="mean",
        )
        matrix = matrix.loc[profile_index_df["profile_id"].tolist()]
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=matrix.fillna(0.0).values,
                x=matrix.columns.tolist(),
                y=matrix.index.tolist(),
                colorscale="YlOrRd",
                colorbar_title="|Delta|",
            )
        )
        fig_heatmap.update_layout(
            title="Per-profile attack effectivity heatmap ordered by empirical susceptibility index",
            template="plotly_white",
            xaxis_title="Opinion leaf",
            yaxis_title="Profile",
            height=max(420, 14 * len(matrix.index) + 180),
        )
        heatmap_path = figures_dir / "profile_effectivity_heatmap.html"
        visual_files.append(_save_figure_html(fig_heatmap, heatmap_path))
        figure_divs.append(("Profile Heatmap", fig_heatmap.to_html(include_plotlyjs=False, full_html=False)))

        fig_index = px.histogram(
            profile_index_df,
            x="susceptibility_index_pct",
            nbins=12,
            color_discrete_sequence=["#d95d39"],
            title="Distribution of post hoc empirical susceptibility index",
        )
        fig_index.update_layout(template="plotly_white", xaxis_title="Susceptibility percentile", yaxis_title="Profiles")
        index_path = figures_dir / "susceptibility_index_distribution.html"
        visual_files.append(_save_figure_html(fig_index, index_path))
        figure_divs.append(("Susceptibility Index", fig_index.to_html(include_plotlyjs=False, full_html=False)))

    if not exploratory_df.empty:
        coeff_df = exploratory_df.copy()
        coeff_df["stars"] = coeff_df["multivariate_p_value"].apply(_significance_stars)
        fig_coeff = go.Figure(
            go.Bar(
                x=coeff_df["multivariate_estimate"],
                y=coeff_df["moderator_label"],
                orientation="h",
                marker_color=np.where(coeff_df["multivariate_estimate"] >= 0, "#1d4e89", "#d95d39"),
                error_x=dict(
                    type="data",
                    array=(coeff_df["multivariate_conf_high"] - coeff_df["multivariate_estimate"]).abs().tolist(),
                    arrayminus=(coeff_df["multivariate_estimate"] - coeff_df["multivariate_conf_low"]).abs().tolist(),
                    visible=True,
                ),
                customdata=coeff_df[["multivariate_p_value", "stars"]].values,
                hovertemplate="%{y}<br>b=%{x:.3f}<br>p=%{customdata[0]:.4f} %{customdata[1]}<extra></extra>",
            )
        )
        fig_coeff.update_layout(
            title="Multivariate profile moderator coefficients on mean absolute opinion shift",
            template="plotly_white",
            xaxis_title="Coefficient estimate",
            yaxis_title="Moderator",
            height=max(420, 26 * len(coeff_df) + 120),
        )
        coeff_path = figures_dir / "moderator_coefficient_forest.html"
        visual_files.append(_save_figure_html(fig_coeff, coeff_path))
        figure_divs.append(("Moderator Coefficients", fig_coeff.to_html(include_plotlyjs=False, full_html=False)))

    if not weight_df.empty:
        grouped_weights = (
            weight_df.groupby("ontology_group", as_index=False)["normalized_weight_pct"]
            .sum()
            .sort_values("normalized_weight_pct", ascending=True)
        )
        fig_weights = go.Figure(
            go.Bar(
                x=grouped_weights["normalized_weight_pct"],
                y=grouped_weights["ontology_group"],
                orientation="h",
                marker_color="#c89b3c",
                hovertemplate="%{y}<br>Weight share=%{x:.2f}%<extra></extra>",
            )
        )
        fig_weights.update_layout(
            title="Ontology-group share of fitted susceptibility weight",
            template="plotly_white",
            xaxis_title="Normalized weight share (%)",
            yaxis_title="Ontology group",
            height=max(360, 42 * len(grouped_weights) + 80),
        )
        weights_path = figures_dir / "moderator_group_weights.html"
        visual_files.append(_save_figure_html(fig_weights, weights_path))
        figure_divs.append(("Moderator Weight Groups", fig_weights.to_html(include_plotlyjs=False, full_html=False)))

    fig_scatter = px.scatter(
        long_df,
        x="baseline_score",
        y="post_score",
        color="opinion_leaf_label",
        hover_data=["scenario_id", "profile_id"],
        title="Baseline versus post-attack opinion scores",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    min_axis = float(min(long_df["baseline_score"].min(), long_df["post_score"].min()))
    max_axis = float(max(long_df["baseline_score"].max(), long_df["post_score"].max()))
    fig_scatter.add_trace(
        go.Scatter(
            x=[min_axis, max_axis],
            y=[min_axis, max_axis],
            mode="lines",
            line=dict(color="#666", dash="dash"),
            name="No change line",
        )
    )
    fig_scatter.update_layout(template="plotly_white")
    scatter_path = figures_dir / "baseline_post_scatter.html"
    visual_files.append(_save_figure_html(fig_scatter, scatter_path))
    figure_divs.append(("Baseline vs Post", fig_scatter.to_html(include_plotlyjs=False, full_html=False)))

    sem_coeff_df = pd.DataFrame(sem_result.get("coefficients", []))
    if not sem_coeff_df.empty:
        sem_coeff_df = sem_coeff_df[sem_coeff_df["op"] == "~"].copy()
        sem_coeff_df["path"] = sem_coeff_df["lhs"] + " ~ " + sem_coeff_df["rhs"]
        sem_coeff_df["stars"] = sem_coeff_df["p_value"].apply(_significance_stars)
        fig_sem = go.Figure(
            go.Bar(
                x=sem_coeff_df["estimate"],
                y=sem_coeff_df["path"],
                orientation="h",
                marker_color=np.where(sem_coeff_df["estimate"] >= 0, "#2a9d8f", "#d95d39"),
                error_x=dict(
                    type="data",
                    array=(sem_coeff_df["std_error"].fillna(0.0) * 1.96).tolist(),
                    visible=True,
                ),
                customdata=sem_coeff_df[["p_value", "stars"]].values,
                hovertemplate="%{y}<br>Estimate=%{x:.3f}<br>p=%{customdata[0]:.4f} %{customdata[1]}<extra></extra>",
            )
        )
        fig_sem.update_layout(
            title="Profile-level SEM coefficients",
            template="plotly_white",
            xaxis_title="Estimate",
            yaxis_title="Path",
            height=max(420, 26 * len(sem_coeff_df) + 120),
        )
        sem_path = figures_dir / "sem_coefficients.html"
        visual_files.append(_save_figure_html(fig_sem, sem_path))
        figure_divs.append(("SEM Coefficients", fig_sem.to_html(include_plotlyjs=False, full_html=False)))

    long_df.to_csv(data_snapshots_dir / "sem_long_encoded_snapshot.csv", index=False)
    if not profile_df.empty:
        profile_df.to_csv(data_snapshots_dir / "profile_level_effectivity_snapshot.csv", index=False)
    if not profile_index_df.empty:
        profile_index_df.to_csv(data_snapshots_dir / "profile_susceptibility_snapshot.csv", index=False)
    if not exploratory_df.empty:
        exploratory_df.to_csv(data_snapshots_dir / "moderator_coefficients_snapshot.csv", index=False)

    notes = [
        "All rows in run_6 are attacked; the dashboard visualizes heterogeneity of attacked opinion movement rather than a treatment-versus-control contrast.",
        "Absolute shift is the primary effectivity outcome because the pilot uses multiple opinion leaves with potentially different signed movement directions.",
        "The empirical susceptibility index is computed post hoc from fitted attack-opinion task models and is therefore descriptive rather than independent inferential evidence.",
        "Moderator weight groups aggregate fitted coefficient importance across ontology-consistent profile components such as age, sex, and Big Five trait families.",
        "The SEM is profile-level and repeated-outcome: multiple attacked opinion shifts are modeled jointly for each profile across the configured opinion-leaf set.",
    ]

    dashboard_path = output_root / "interactive_sem_dashboard.html"
    dashboard_html = _render_dashboard_html(run_id, summary_cards, figure_divs, notes)
    dashboard_path.write_text(dashboard_html, encoding="utf-8")
    visual_files.append(str(dashboard_path))

    return {
        "dashboard_path": str(dashboard_path),
        "visual_files": visual_files,
        "summary_cards": summary_cards,
    }
