from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from src.backend.utils.data_utils import infer_analysis_mode


def _significance_stars(p_value: float | None) -> str:
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _save_figure_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)


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
  <title>{run_id} SEM Interactive Dashboard</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #14213d;
      --ink-soft: #4a5d7a;
      --accent: #ef476f;
      --accent-2: #118ab2;
      --ok: #2a9d8f;
      --warn: #e76f51;
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
      max-width: 1380px;
      margin: 0 auto;
      padding: 22px 18px 30px;
    }}
    .hero {{
      background: linear-gradient(120deg, #14213d, #1f3b73);
      color: #fff;
      border-radius: 16px;
      box-shadow: var(--shadow);
      padding: 22px 24px;
      margin-bottom: 16px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 1.45rem;
      letter-spacing: 0.02em;
    }}
    .hero p {{
      margin: 0;
      color: #d6e2ff;
      font-size: 0.95rem;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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
      <h1>Interactive SEM Investigation: {run_id}</h1>
      <p>Attack moderation pipeline with profile-conditioned simulated opinions and self-supervised attack realism checks.</p>
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

    df = pd.read_csv(sem_long_csv_path)
    sem_result = json.loads(Path(sem_result_json_path).read_text(encoding="utf-8"))
    ols_params = pd.read_csv(ols_params_csv_path)
    analysis_mode = infer_analysis_mode(df)

    df_attack = df[df["attack_present"] == 1].copy()
    df_control = df[df["attack_present"] == 0].copy()

    fit_indices = sem_result.get("fit_indices", {})
    if analysis_mode == "treated_only":
        summary_cards = {
            "Rows": len(df),
            "Design": "Attacked Only",
            "Mean Delta": f"{df['delta_score'].mean():.2f}",
            "Delta SD": f"{df['delta_score'].std(ddof=0):.2f}",
            "Mean Realism": (
                f"{df['attack_realism_score'].dropna().mean():.2f}"
                if "attack_realism_score" in df and len(df["attack_realism_score"].dropna())
                else "n/a"
            ),
            "CFI": f"{fit_indices.get('CFI', float('nan')):.3f}" if fit_indices.get("CFI") is not None else "n/a",
            "RMSEA": f"{fit_indices.get('RMSEA', float('nan')):.3f}" if fit_indices.get("RMSEA") is not None else "n/a",
        }
    else:
        summary_cards = {
            "Rows": len(df),
            "Attack Ratio": f"{df['attack_present'].mean():.2f}",
            "Mean Delta (Attack)": f"{df_attack['delta_score'].mean():.2f}" if len(df_attack) else "n/a",
            "Mean Delta (Control)": f"{df_control['delta_score'].mean():.2f}" if len(df_control) else "n/a",
            "Mean Realism": (
                f"{df_attack['attack_realism_score'].dropna().mean():.2f}"
                if "attack_realism_score" in df_attack and len(df_attack['attack_realism_score'].dropna())
                else "n/a"
            ),
            "SEM Converged": sem_result.get("converged"),
        }

    sem_coeff_df = pd.DataFrame(sem_result.get("coefficients", []))
    sem_coeff_df = sem_coeff_df[sem_coeff_df["op"] == "~"].copy() if not sem_coeff_df.empty else sem_coeff_df
    sem_coeff_df = sem_coeff_df.sort_values("estimate") if not sem_coeff_df.empty else sem_coeff_df

    if not sem_coeff_df.empty:
        sem_coeff_df["label"] = sem_coeff_df["lhs"] + " ~ " + sem_coeff_df["rhs"]
        sem_coeff_df["sig"] = sem_coeff_df["p_value"].apply(_significance_stars)

        fig_sem_coeff = go.Figure(
            go.Bar(
                x=sem_coeff_df["estimate"],
                y=sem_coeff_df["label"],
                orientation="h",
                marker_color=np.where(sem_coeff_df["estimate"] >= 0, "#118ab2", "#ef476f"),
                error_x=dict(
                    type="data",
                    array=(sem_coeff_df["std_error"].fillna(0.0) * 1.96).tolist(),
                    visible=True,
                ),
                customdata=sem_coeff_df[["p_value", "sig"]].values,
                hovertemplate="%{y}<br>Estimate=%{x:.3f}<br>p=%{customdata[0]:.4f} %{customdata[1]}<extra></extra>",
            )
        )
        fig_sem_coeff.update_layout(
            title="SEM Path Coefficients (95% CI)",
            xaxis_title="Estimate",
            yaxis_title="Path",
            template="plotly_white",
            height=500,
        )
    else:
        fig_sem_coeff = go.Figure()

    if analysis_mode == "treated_only":
        if df["primary_moderator_value"].nunique() >= 3:
            work = df.copy()
            work["Susceptibility Group"] = pd.qcut(
                work["primary_moderator_value"],
                q=3,
                labels=["Low", "Mid", "High"],
                duplicates="drop",
            )
            fig_delta = px.violin(
                work,
                x="Susceptibility Group",
                y="delta_score",
                box=True,
                points="all",
                color="Susceptibility Group",
                color_discrete_map={"Low": "#118ab2", "Mid": "#c89b3c", "High": "#ef476f"},
            )
            fig_delta.update_layout(
                title="Delta Score Distribution by Susceptibility Group",
                xaxis_title="Susceptibility Group",
                yaxis_title="Post - Baseline",
                template="plotly_white",
                showlegend=False,
            )
        else:
            fig_delta = px.violin(
                df,
                y="delta_score",
                box=True,
                points="all",
                color_discrete_sequence=["#ef476f"],
                title="Delta Score Distribution",
            )
            fig_delta.update_layout(template="plotly_white", showlegend=False, yaxis_title="Post - Baseline")
    else:
        fig_delta = px.violin(
            df,
            x=df["attack_present"].map({0: "Control", 1: "Attack"}),
            y="delta_score",
            box=True,
            points="all",
            color=df["attack_present"].map({0: "Control", 1: "Attack"}),
            color_discrete_map={"Control": "#118ab2", "Attack": "#ef476f"},
        )
        fig_delta.update_layout(
            title="Delta Score Distribution by Condition",
            xaxis_title="Condition",
            yaxis_title="Post - Baseline",
            template="plotly_white",
            showlegend=False,
        )

    if analysis_mode == "treated_only":
        fig_base_post = px.scatter(
            df,
            x="baseline_score",
            y="post_score",
            color="opinion_leaf_label",
            hover_data=["scenario_id", "opinion_leaf"],
            title="Baseline vs Post Scores",
        )
    else:
        fig_base_post = px.scatter(
            df,
            x="baseline_score",
            y="post_score",
            color=df["attack_present"].map({0: "Control", 1: "Attack"}),
            hover_data=["scenario_id", "opinion_leaf"],
            color_discrete_map={"Control": "#118ab2", "Attack": "#ef476f"},
            title="Baseline vs Post Scores",
        )
    min_axis = float(min(df["baseline_score"].min(), df["post_score"].min()))
    max_axis = float(max(df["baseline_score"].max(), df["post_score"].max()))
    fig_base_post.add_trace(
        go.Scatter(
            x=[min_axis, max_axis],
            y=[min_axis, max_axis],
            mode="lines",
            line=dict(color="#666", dash="dash"),
            name="No change line",
        )
    )
    fig_base_post.update_layout(template="plotly_white")

    if analysis_mode == "treated_only":
        fig_interaction = px.scatter(
            df,
            x="primary_moderator_value",
            y="delta_score",
            color="baseline_score",
            color_continuous_scale="RdBu",
            hover_data=["scenario_id", "opinion_leaf"],
            title="Moderation: Opinion Delta vs Primary Moderator",
        )
        if {"primary_moderator_value", "delta_score"}.issubset(df.columns):
            params_map = {row["term"]: row["estimate"] for _, row in ols_params.iterrows()}
            b0 = float(params_map.get("Intercept", 0.0))
            b_baseline = float(params_map.get("baseline_score", 0.0))
            b_abs = float(params_map.get("baseline_abs_score", 0.0))
            b_mod = float(params_map.get("primary_moderator_z", 0.0))
            b_quality = float(params_map.get("exposure_quality_z", 0.0))
            fixed_effect_terms = [
                column
                for column in df.columns
                if (column.startswith("opinion_leaf_fe_") or column.startswith("opinion_domain_fe_")) and column in params_map
            ]
            fixed_offset = sum(float(params_map.get(column, 0.0)) * float(df[column].mean()) for column in fixed_effect_terms)
            raw_mean = float(df["primary_moderator_value"].mean())
            raw_std = float(df["primary_moderator_value"].std(ddof=0)) or 1.0
            xline_raw = np.linspace(float(df["primary_moderator_value"].min()), float(df["primary_moderator_value"].max()), 60)
            xline_z = (xline_raw - raw_mean) / raw_std
            quality_component = b_quality * float(df["exposure_quality_z"].mean()) if "exposure_quality_z" in df.columns else 0.0
            yline = (
                b0
                + b_baseline * float(df["baseline_score"].mean())
                + b_abs * float(df["baseline_abs_score"].mean())
                + quality_component
            )
            yline = yline + fixed_offset + b_mod * xline_z
            fig_interaction.add_trace(
                go.Scatter(x=xline_raw, y=yline, mode="lines", name="Model trend", line=dict(color="#14213d", width=3))
            )
        fig_interaction.update_layout(template="plotly_white", xaxis_title="Primary moderator", yaxis_title="Opinion delta")
    else:
        fig_interaction = px.scatter(
            df,
            x="moderator_z",
            y="post_score",
            color=df["attack_present"].map({0: "Control", 1: "Attack"}),
            color_discrete_map={"Control": "#118ab2", "Attack": "#ef476f"},
            hover_data=["scenario_id", "opinion_leaf"],
            title="Moderation Interaction: Baseline-adjusted Post Score vs Moderator",
        )

        if set(["attack_present", "moderator_z", "attack_x_moderator"]).issubset(df.columns):
            params_map = {row["term"]: row["estimate"] for _, row in ols_params.iterrows()}
            b0 = float(params_map.get("Intercept", 0.0))
            b1 = float(params_map.get("attack_present", 0.0))
            b_baseline = float(params_map.get("baseline_score", 0.0))
            b2 = float(params_map.get("moderator_z", 0.0))
            b3 = float(params_map.get("attack_x_moderator", 0.0))
            fixed_effect_terms = [
                column
                for column in df.columns
                if (column.startswith("opinion_leaf_fe_") or column.startswith("opinion_domain_fe_")) and column in params_map
            ]
            fixed_offset = sum(float(params_map.get(column, 0.0)) * float(df[column].mean()) for column in fixed_effect_terms)
            baseline_mean = float(df["baseline_score"].mean())

            xline = np.linspace(float(df["moderator_z"].min()), float(df["moderator_z"].max()), 60)
            y_control = b0 + (b_baseline * baseline_mean) + fixed_offset + b2 * xline
            y_attack = b0 + (b_baseline * baseline_mean) + fixed_offset + b1 + (b2 + b3) * xline

            fig_interaction.add_trace(
                go.Scatter(x=xline, y=y_control, mode="lines", name="Control trend", line=dict(color="#118ab2", width=3))
            )
            fig_interaction.add_trace(
                go.Scatter(x=xline, y=y_attack, mode="lines", name="Attack trend", line=dict(color="#ef476f", width=3))
            )
        fig_interaction.update_layout(template="plotly_white")

    if "attack_realism_score" in df.columns:
        realism_source = df if analysis_mode == "treated_only" else df_attack
        fig_realism = px.histogram(
            realism_source,
            x="attack_realism_score",
            nbins=12,
            title="Attack Realism Score Distribution (treated only)",
            color_discrete_sequence=["#2a9d8f"],
        )
        fig_realism.update_layout(template="plotly_white", xaxis_title="Realism score")
    else:
        fig_realism = go.Figure()

    corr_cols = [
        c
        for c in [
            "baseline_score",
            "post_score",
            "delta_score",
            "moderator_z",
            "attack_present",
            "attack_x_moderator",
            "primary_moderator_value",
            "exposure_intensity_hint",
            "exposure_quality_score",
            "attack_realism_score",
            "attack_coherence_score",
        ]
        if c in df.columns
    ]
    corr_df = df[corr_cols].corr(numeric_only=True)
    fig_corr = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap (Key Variables)",
    )
    fig_corr.update_layout(template="plotly_white")

    sem_nodes = ["baseline_score", "baseline_abs_score", "primary_moderator_z", "exposure_quality_z", "delta_score", "post_score"]
    node_map = {name: idx for idx, name in enumerate(sem_nodes)}
    sem_targets = {"baseline_score", "delta_score"} if analysis_mode == "treated_only" else {"baseline_score", "post_score"}
    sem_links = sem_coeff_df[sem_coeff_df["lhs"].isin(sem_targets)].copy() if not sem_coeff_df.empty else pd.DataFrame()
    if not sem_links.empty:
        sem_links = sem_links[sem_links["rhs"].isin(node_map.keys())]
    fig_sankey = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color="#0f172a", width=0.5),
                    label=sem_nodes,
                    color=["#577590", "#8d99ae", "#118ab2", "#2a9d8f", "#ef476f", "#f4a261"],
                ),
                link=dict(
                    source=[node_map.get(rhs, 0) for rhs in sem_links.get("rhs", [])],
                    target=[node_map.get(lhs, 0) for lhs in sem_links.get("lhs", [])],
                    value=[abs(float(v)) + 0.05 for v in sem_links.get("estimate", [])],
                    label=[
                        f"{lhs}~{rhs} ({est:.2f})"
                        for lhs, rhs, est in zip(
                            sem_links.get("lhs", []),
                            sem_links.get("rhs", []),
                            sem_links.get("estimate", []),
                        )
                    ],
                    color=["rgba(17,138,178,0.45)" if float(v) >= 0 else "rgba(239,71,111,0.45)" for v in sem_links.get("estimate", [])],
                ),
            )
        ]
    )
    fig_sankey.update_layout(title_text="SEM Path Structure (magnitude encoded)", font_size=12)

    figures = [
        ("SEM Coefficients", fig_sem_coeff, "01_sem_coefficients.html"),
        ("Delta Distribution", fig_delta, "02_delta_distribution.html"),
        ("Baseline vs Post", fig_base_post, "03_baseline_post_scatter.html"),
        ("Moderation Interaction", fig_interaction, "04_interaction_plot.html"),
        ("Attack Realism", fig_realism, "05_attack_realism_histogram.html"),
        ("Correlation Heatmap", fig_corr, "06_correlation_heatmap.html"),
        ("SEM Structure", fig_sankey, "07_sem_structure_sankey.html"),
    ]

    figure_divs: List[Tuple[str, str]] = []
    visual_files: List[str] = []

    for title, fig, filename in figures:
        _save_figure_html(fig, figures_dir / filename)
        visual_files.append(str((figures_dir / filename).resolve()))
        figure_divs.append((title, pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False})))

    df.head(250).to_csv(data_snapshots_dir / "sem_data_preview.csv", index=False)
    sem_coeff_df.to_csv(data_snapshots_dir / "sem_coefficients_filtered.csv", index=False)

    dashboard_html = _render_dashboard_html(
        run_id=run_id,
        summary_cards=summary_cards,
        figure_divs=figure_divs,
        notes=[
            "Dashboard is diagnostic, not confirmatory; run_5 estimates moderation among attacked individuals only.",
            "Interactive figures combine empirical deltas, path coefficients, and scenario-level quality diagnostics.",
            "Review the methodology audit output before interpreting coefficients substantively.",
        ],
    )
    dashboard_path = output_root / "interactive_sem_dashboard.html"
    dashboard_path.write_text(dashboard_html, encoding="utf-8")

    visual_files.append(str(dashboard_path.resolve()))
    visual_files.append(str((data_snapshots_dir / "sem_data_preview.csv").resolve()))
    visual_files.append(str((data_snapshots_dir / "sem_coefficients_filtered.csv").resolve()))

    return {
        "dashboard_path": str(dashboard_path.resolve()),
        "visual_files": visual_files,
        "summary_cards": summary_cards,
    }
