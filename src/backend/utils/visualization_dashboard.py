"""
Interactive attack-effectivity dashboard — next-level visualization.

Tabs produced (generic, works for any run):
  1  Factorial 3D Surface      – mean AE + ISD 3D surfaces with projected contours
  2  Factorial Heat + Contour  – 2D heatmap pair (mean AE | ISD) side-by-side
  3  SEM Network               – bipartite directed graph of SEM path coefficients
  4  SEM Heatmap               – annotated heatmap of all moderator→indicator paths
  5  Perturbation Explorer     – JS-powered live sliders → predicted AE grid
  6  Violin Distributions      – violin + scatter by opinion leaf and attack vector
  7  Susceptibility Map        – scatter + jitter coloured by susceptibility index
  8  Moderator Forest          – forest plot with CI bars + FDR markers
  9  Hierarchical Importance   – treemap / waterfall of feature group importance
 10  Profile Heatmap           – sorted effectivity heatmap (profiles × opinion leaves)
 11  Baseline vs Post          – scatter with quadrant shading
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from src.backend.utils.data_utils import infer_analysis_mode


# ── helpers ───────────────────────────────────────────────────────────────────

PALETTE = dict(
    navy="#14213d", blue="#1d4e89", sky="#2980b9",
    teal="#2a9d8f", orange="#e76f51", red="#c0392b",
    amber="#c89b3c", ink="#14213d", muted="#4a5d7a",
    panel="#ffffff", line="#dbe3ef",
)

_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="white",
        plot_bgcolor="#f8faff",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif", color=PALETTE["ink"]),
        title_font=dict(size=15, color=PALETTE["ink"]),
        margin=dict(l=60, r=30, t=50, b=50),
    )
)


def _p_stars(p: float | None) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else ""


def _leaf(s: str) -> str:
    """Return last segment of a '>' separated ontology path, cleaned."""
    raw = s.rsplit(">", 1)[-1].strip() if ">" in str(s) else str(s)
    return raw.replace("_", " ").strip()


def _pretty(s: str) -> str:
    for prefix in ["profile_cont_", "profile_cat__profile_cat_", "profile_cat__"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.replace("_z", "").replace("_", " ").replace("  ", " ").strip().title()


def _apply_style(fig: go.Figure, height: int = 520) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#f8faff",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif", color=PALETTE["ink"]),
        title_font=dict(size=14, color=PALETTE["ink"]),
        margin=dict(l=60, r=30, t=50, b=50),
        height=height,
    )
    return fig


def _save_figure_html(fig: go.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    return str(path)


# ── figure builders ───────────────────────────────────────────────────────────

def _fig_factorial_3d(long_df: pd.DataFrame) -> go.Figure:
    """Two-panel 3D: mean AE (left) and inter-individual SD AE (right)."""
    req = {"attack_leaf_label", "opinion_leaf_label", "adversarial_effectivity"}
    if not req.issubset(long_df.columns):
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    atk_col, op_col, ae_col = "attack_leaf_label", "opinion_leaf_label", "adversarial_effectivity"

    attacks  = sorted(long_df[atk_col].dropna().unique(), key=lambda x: str(x))
    opinions = sorted(long_df[op_col].dropna().unique(), key=lambda x: str(x))

    mean_mat = (
        long_df.groupby([atk_col, op_col])[ae_col].mean()
        .unstack(op_col).reindex(index=attacks, columns=opinions)
    )
    isd_mat = (
        long_df.groupby([atk_col, op_col])[ae_col].std()
        .unstack(op_col).reindex(index=attacks, columns=opinions)
    )

    atk_labels = [_leaf(a) for a in attacks]
    op_labels  = [_leaf(o) for o in opinions]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Mean Adversarial Effectivity (AE)", "Inter-individual SD of AE"],
        horizontal_spacing=0.06,
    )

    # surface common args
    def _surface(mat, cscale, title_color):
        z = mat.fillna(0).values.astype(float)
        return go.Surface(
            z=z,
            x=op_labels,
            y=atk_labels,
            colorscale=cscale,
            cmid=0,
            lighting=dict(ambient=0.7, diffuse=0.9, specular=0.3, roughness=0.5),
            lightposition=dict(x=100, y=200, z=500),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white",
                       project=dict(z=True), size=5),
            ),
            hovertemplate=(
                "<b>Attack:</b> %{y}<br>"
                "<b>Opinion:</b> %{x}<br>"
                f"<b>{title_color}:</b> %{{z:.2f}}<extra></extra>"
            ),
        )

    fig.add_trace(_surface(mean_mat, "RdBu_r", "Mean AE"), row=1, col=1)
    fig.add_trace(_surface(isd_mat, "YlOrRd", "ISD AE"),  row=1, col=2)

    camera = dict(eye=dict(x=1.6, y=-1.6, z=1.0))
    for scene in ("scene", "scene2"):
        fig.update_layout(**{
            scene: dict(
                xaxis=dict(title="Opinion leaf", tickfont=dict(size=9), gridcolor="#ccd6e8"),
                yaxis=dict(title="Attack vector",  tickfont=dict(size=9), gridcolor="#ccd6e8"),
                zaxis=dict(title="AE", gridcolor="#ccd6e8"),
                camera=camera,
                bgcolor="white",
            )
        })

    fig.update_layout(
        paper_bgcolor="white",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif"),
        height=580,
        title=dict(text="3D Factorial Effectivity Surface — Mean AE & Inter-individual Variability", font=dict(size=14)),
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )
    return fig


def _fig_factorial_2d(long_df: pd.DataFrame) -> go.Figure:
    """Side-by-side 2D heatmaps: mean AE and ISD, with contour overlay."""
    atk_col, op_col, ae_col = "attack_leaf_label", "opinion_leaf_label", "adversarial_effectivity"
    attacks  = sorted(long_df[atk_col].dropna().unique())
    opinions = sorted(long_df[op_col].dropna().unique())

    mean_mat = (
        long_df.groupby([atk_col, op_col])[ae_col].mean()
        .unstack(op_col).reindex(index=attacks, columns=opinions)
    )
    isd_mat = (
        long_df.groupby([atk_col, op_col])[ae_col].std()
        .unstack(op_col).reindex(index=attacks, columns=opinions)
    )

    atk_l = [_leaf(a) for a in attacks]
    op_l  = [_leaf(o) for o in opinions]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Mean AE (red = attack succeeded)", "Inter-individual SD of AE"],
        horizontal_spacing=0.12,
    )

    def _annot(mat):
        return [[f"{v:.1f}" for v in row] for row in mat.fillna(0).values]

    fig.add_trace(go.Heatmap(
        z=mean_mat.fillna(0).values,
        x=op_l, y=atk_l,
        colorscale="RdBu_r", zmid=0,
        text=_annot(mean_mat), texttemplate="%{text}",
        hovertemplate="Attack: %{y}<br>Opinion: %{x}<br>Mean AE: %{z:.2f}<extra></extra>",
        colorbar=dict(x=0.44, thickness=12, title="Mean AE", title_side="right"),
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=isd_mat.fillna(0).values,
        x=op_l, y=atk_l,
        colorscale="YlOrRd",
        text=_annot(isd_mat), texttemplate="%{text}",
        hovertemplate="Attack: %{y}<br>Opinion: %{x}<br>ISD: %{z:.2f}<extra></extra>",
        colorbar=dict(x=1.01, thickness=12, title="ISD AE", title_side="right"),
    ), row=1, col=2)

    fig.update_xaxes(tickangle=-30, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8faff",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif"),
        height=420,
        margin=dict(l=140, r=80, t=50, b=100),
        title=dict(text="Factorial Heatmap — Mean AE & Inter-individual Moderation Strength", font=dict(size=14)),
    )
    return fig


def _fig_sem_network(sem_coeff_df: pd.DataFrame) -> go.Figure:
    """
    Bipartite directed SEM network.
    Left column = profile moderators; right column = opinion indicators.
    Edge width ∝ |estimate|; colour = positive (blue) / negative (red);
    opacity = significance level.
    """
    df = sem_coeff_df[sem_coeff_df["op"] == "~"].copy()
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["p_value"]  = pd.to_numeric(df["p_value"],  errors="coerce")
    df = df.dropna(subset=["estimate"])

    moderators = df["rhs"].unique().tolist()
    indicators = df["lhs"].unique().tolist()

    n_mod = len(moderators)
    n_ind = len(indicators)

    # node positions
    mod_y  = np.linspace(1, 0, n_mod)
    ind_y  = np.linspace(0.85, 0.15, n_ind)

    mod_pos = {m: (0.0, float(mod_y[i])) for i, m in enumerate(moderators)}
    ind_pos = {ind: (1.0, float(ind_y[i])) for i, ind in enumerate(indicators)}

    max_abs = df["estimate"].abs().max() or 1.0

    traces: List[go.BaseTraceType] = []

    # --- edges ---
    for _, row in df.iterrows():
        if row["rhs"] not in mod_pos or row["lhs"] not in ind_pos:
            continue
        x0, y0 = mod_pos[row["rhs"]]
        x1, y1 = ind_pos[row["lhs"]]
        p   = row["p_value"]
        est = row["estimate"]
        stars = _p_stars(p)
        alpha = 0.95 if (pd.notna(p) and p < 0.05) else 0.45 if (pd.notna(p) and p < 0.10) else 0.14
        width = max(0.8, abs(est) / max_abs * 9)
        color = f"rgba(29,78,137,{alpha})" if est >= 0 else f"rgba(192,57,43,{alpha})"

        traces.append(go.Scatter(
            x=[x0, x0 + 0.05, x1 - 0.05, x1, None],
            y=[y0, y0, y1, y1, None],
            mode="lines",
            line=dict(color=color, width=width, shape="spline"),
            hovertemplate=(
                f"<b>{row['rhs']} → {_leaf(row['lhs'])}</b><br>"
                f"β = {est:.3f}{stars}<br>"
                f"p = {p:.4f}<extra></extra>"
            ),
            showlegend=False,
        ))

    # --- moderator nodes ---
    traces.append(go.Scatter(
        x=[0.0] * n_mod,
        y=[mod_pos[m][1] for m in moderators],
        mode="markers+text",
        marker=dict(size=16, color=PALETTE["navy"], line=dict(color="white", width=2)),
        text=moderators,
        textposition="middle left",
        textfont=dict(size=9.5, color=PALETTE["ink"]),
        hoverinfo="text",
        showlegend=False,
    ))

    # --- indicator nodes ---
    ind_labels = [_leaf(ind) for ind in indicators]
    traces.append(go.Scatter(
        x=[1.0] * n_ind,
        y=[ind_pos[ind][1] for ind in indicators],
        mode="markers+text",
        marker=dict(size=16, color=PALETTE["teal"], line=dict(color="white", width=2)),
        text=ind_labels,
        textposition="middle right",
        textfont=dict(size=9.5, color=PALETTE["ink"]),
        hoverinfo="text",
        showlegend=False,
    ))

    # --- significance legend shapes ---
    annotations = [
        dict(x=0.02, y=1.04, xref="paper", yref="paper", showarrow=False,
             text="<b>PROFILE MODERATORS</b>", font=dict(size=11, color=PALETTE["navy"])),
        dict(x=0.98, y=1.04, xref="paper", yref="paper", showarrow=False,
             text="<b>OPINION INDICATORS</b>", font=dict(size=11, color=PALETTE["teal"])),
        dict(x=0.5, y=-0.06, xref="paper", yref="paper", showarrow=False,
             text="Edge width ∝ |β| &nbsp;·&nbsp; Blue = positive &nbsp;·&nbsp; Red = negative &nbsp;·&nbsp; "
                  "Opacity: solid = p<.05, medium = p<.10, faint = n.s.",
             font=dict(size=9, color=PALETTE["muted"])),
    ]

    fig = go.Figure(traces)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif"),
        height=540,
        margin=dict(l=200, r=200, t=60, b=60),
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.05, 1.1]),
        title=dict(text="SEM Network — Profile Moderators → Opinion Shift Indicators", font=dict(size=14)),
        annotations=annotations,
        hovermode="closest",
    )
    return fig


def _fig_sem_heatmap(sem_coeff_df: pd.DataFrame, exploratory_df: pd.DataFrame) -> go.Figure:
    """Annotated interactive heatmap of SEM path coefficients."""
    df = sem_coeff_df[sem_coeff_df["op"] == "~"].copy()
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["p_value"]  = pd.to_numeric(df["p_value"],  errors="coerce")

    indicators = df["lhs"].unique().tolist()
    hm = df.pivot_table(index="rhs", columns="lhs", values="estimate", aggfunc="mean")

    # order rows by weight if available
    if "normalized_weight_pct" in exploratory_df.columns and "moderator_label" in exploratory_df.columns:
        order = exploratory_df.sort_values("normalized_weight_pct", ascending=False)["moderator_label"].tolist()
        hm = hm.reindex([r for r in order if r in hm.index])

    hm = hm[[c for c in indicators if c in hm.columns]]
    col_labels = [_leaf(c) for c in hm.columns]

    annot_text = []
    hover_text = []
    for rhs in hm.index:
        row_ann, row_hov = [], []
        for lhs in hm.columns:
            sub = df[(df["lhs"] == lhs) & (df["rhs"] == rhs)]
            if sub.empty:
                row_ann.append("")
                row_hov.append("")
            else:
                r = sub.iloc[0]
                stars = _p_stars(r["p_value"])
                row_ann.append(f"{r['estimate']:.2f}{stars}")
                row_hov.append(f"{rhs} → {_leaf(lhs)}<br>β={r['estimate']:.3f}{stars}<br>p={r['p_value']:.4f}")
        annot_text.append(row_ann)
        hover_text.append(row_hov)

    fig = go.Figure(go.Heatmap(
        z=hm.fillna(0).values,
        x=col_labels,
        y=list(hm.index),
        colorscale="RdBu_r", zmid=0,
        text=annot_text,
        texttemplate="%{text}",
        customdata=hover_text,
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title="β", thickness=14),
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8faff",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif"),
        height=max(420, 36 * len(hm) + 120),
        margin=dict(l=220, r=40, t=50, b=100),
        title=dict(text="SEM Path Coefficients — Profile Moderators → Attacked Opinion Indicators", font=dict(size=14)),
        xaxis=dict(tickangle=-28),
    )
    return fig


def _html_perturbation_explorer(
    task_coeff_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> str:
    """
    Returns a self-contained HTML block with JavaScript sliders.
    Moving sliders re-computes predicted AE per (attack × opinion) cell client-side
    using the embedded ridge-model coefficients.
    """
    if task_coeff_df.empty:
        return "<p style='color:#888'>Perturbation data unavailable.</p>"

    # --- build coefficient JSON ---
    # keys: short "atk | op" labels
    # For each task: {term: estimate, ...}

    CONT_FEATURES = [
        ("profile_cont_big_five_agreeableness_mean_pct",    "Agreeableness",    50.0),
        ("profile_cont_big_five_conscientiousness_mean_pct","Conscientiousness", 50.0),
        ("profile_cont_big_five_extraversion_mean_pct",     "Extraversion",      50.0),
        ("profile_cont_big_five_neuroticism_mean_pct",      "Neuroticism",       50.0),
        ("profile_cont_big_five_openness_to_experience_mean_pct", "Openness",    50.0),
        ("profile_cont_chronological_age",                  "Age (years)",       40.0),
    ]

    tasks_json: dict = {}
    for (task_key, group) in task_coeff_df.groupby(["attack_leaf", "opinion_leaf"]):
        atk_short = _leaf(task_key[0])
        op_short  = _leaf(task_key[1])
        label     = f"{atk_short} | {op_short}"
        coeffs    = dict(zip(group["term"], group["estimate"].astype(float)))
        tasks_json[label] = coeffs

    # compute overall mean AE per task for baseline annotation
    mean_ae_json: dict = {}
    for (task_key, group) in long_df.groupby(["attack_leaf_label", "opinion_leaf_label"])["adversarial_effectivity"]:
        atk_short = _leaf(task_key[0] if isinstance(task_key, tuple) else task_key)
        op_short  = _leaf(task_key[1] if isinstance(task_key, tuple) else "")
        mean_ae_json[f"{atk_short} | {op_short}"] = float(group.mean())

    tasks_json_str = json.dumps(tasks_json)
    slider_defs = json.dumps([
        {"term": t, "label": lbl, "default": default, "min": 0, "max": 100 if "age" not in lbl.lower() else 80}
        for t, lbl, default in CONT_FEATURES
    ])

    return f"""
<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;padding:8px 0;">

<!-- sliders panel -->
<div style="min-width:260px;flex:0 0 280px;">
  <h3 style="margin:0 0 12px;font-size:1rem;color:#14213d;">Profile Feature Sliders</h3>
  <div style="font-size:0.8rem;color:#4a5d7a;margin-bottom:10px;">
    Adjust Big Five trait means and age to see how predicted adversarial effectivity changes
    across the 4 × 4 attack–opinion matrix.
  </div>
  <div id="sliders" style="display:flex;flex-direction:column;gap:10px;"></div>

  <div style="margin-top:16px;">
    <label style="font-size:0.82rem;color:#14213d;font-weight:600;">Sex</label>
    <select id="sex-select"
      style="width:100%;margin-top:4px;padding:6px;border-radius:6px;border:1px solid #dbe3ef;font-size:0.9rem;">
      <option value="Male">Male</option>
      <option value="Female">Female</option>
      <option value="Other">Other</option>
    </select>
  </div>

  <button onclick="resetSliders()"
    style="margin-top:14px;padding:7px 14px;background:#1d4e89;color:#fff;
           border:none;border-radius:8px;cursor:pointer;font-size:0.88rem;">
    Reset to Defaults
  </button>

  <div style="margin-top:20px;padding:10px;background:#f0f4ff;border-radius:8px;font-size:0.8rem;color:#4a5d7a;">
    <b>Method.</b> Ridge-regularised task models (one per attack×opinion cell).
    Coefficients embedded directly from pipeline output.
  </div>
</div>

<!-- heatmap panel -->
<div style="flex:1;min-width:340px;">
  <h3 style="margin:0 0 4px;font-size:1rem;color:#14213d;">Predicted Adversarial Effectivity</h3>
  <div style="font-size:0.79rem;color:#4a5d7a;margin-bottom:10px;">
    Red = attack succeeds (AE > 0) &nbsp;·&nbsp; Blue = backfire (AE &lt; 0)
  </div>
  <div id="ae-grid" style="overflow-x:auto;"></div>
  <div id="ae-caption" style="font-size:0.78rem;color:#888;margin-top:8px;"></div>
</div>

</div>

<script>
(function() {{
  const TASKS  = {tasks_json_str};
  const SLIDERS = {slider_defs};
  const defaults = {{}};
  SLIDERS.forEach(s => defaults[s.term] = s.default);

  function buildSliders() {{
    const container = document.getElementById('sliders');
    SLIDERS.forEach(s => {{
      const wrap = document.createElement('div');
      const lbl  = document.createElement('label');
      lbl.style = 'font-size:0.82rem;color:#14213d;font-weight:600;display:flex;justify-content:space-between;';
      const spanL = document.createElement('span'); spanL.textContent = s.label;
      const spanV = document.createElement('span'); spanV.id = 'val-' + s.term; spanV.textContent = s.default;
      lbl.appendChild(spanL); lbl.appendChild(spanV);
      const inp = document.createElement('input');
      inp.type = 'range'; inp.min = s.min; inp.max = s.max; inp.step = 1;
      inp.value = s.default; inp.id = 'sl-' + s.term;
      inp.style = 'width:100%;margin-top:2px;accent-color:#1d4e89;';
      inp.addEventListener('input', () => {{
        document.getElementById('val-' + s.term).textContent = inp.value;
        update();
      }});
      wrap.appendChild(lbl); wrap.appendChild(inp);
      container.appendChild(wrap);
    }});
    document.getElementById('sex-select').addEventListener('change', update);
  }}

  function getValues() {{
    const vals = {{ 'Intercept': 1 }};
    SLIDERS.forEach(s => {{
      vals[s.term] = parseFloat(document.getElementById('sl-' + s.term).value);
    }});
    const sex = document.getElementById('sex-select').value;
    vals['profile_cat__profile_cat_sex_Female'] = sex === 'Female' ? 1 : 0;
    vals['profile_cat__profile_cat_sex_Male']   = sex === 'Male'   ? 1 : 0;
    vals['profile_cat__profile_cat_sex_Other']  = sex === 'Other'  ? 1 : 0;
    return vals;
  }}

  function predict(taskCoeffs, vals) {{
    let ae = 0;
    for (const [term, coeff] of Object.entries(taskCoeffs)) {{
      ae += coeff * (vals[term] !== undefined ? vals[term] : 0);
    }}
    return ae;
  }}

  function colorForAE(ae) {{
    const v = Math.max(-1, Math.min(1, ae / 60));
    if (v >= 0) {{
      const r = Math.round(180 + 75*v); const g = Math.round(60 - 60*v); const b = Math.round(60 - 60*v);
      return `rgb(${{r}},${{g}},${{b}})`;
    }} else {{
      const t = -v;
      const r = Math.round(60 - 60*t); const g = Math.round(60 + 80*t); const b = Math.round(180 + 75*t);
      return `rgb(${{r}},${{g}},${{b}})`;
    }}
  }}

  function update() {{
    const vals = getValues();
    const grid = document.getElementById('ae-grid');
    const taskNames = Object.keys(TASKS);

    // collect attacks & opinions
    const atkSet = new Set(); const opSet = new Set();
    taskNames.forEach(k => {{
      const [atk, op] = k.split(' | ');
      atkSet.add(atk); opSet.add(op);
    }});
    const atks = [...atkSet]; const ops = [...opSet];

    let html = '<table style="border-collapse:collapse;font-size:0.82rem;min-width:360px;">';
    // header
    html += '<tr><th style="text-align:left;padding:6px 10px;border-bottom:2px solid #dbe3ef;font-size:0.78rem;">Attack \\ Opinion</th>';
    ops.forEach(op => {{
      html += `<th style="padding:6px 8px;border-bottom:2px solid #dbe3ef;font-size:0.78rem;max-width:90px;word-break:break-word;">${{op}}</th>`;
    }});
    html += '</tr>';
    atks.forEach(atk => {{
      html += `<tr><td style="padding:6px 10px;font-weight:600;font-size:0.79rem;border-right:1px solid #dbe3ef;">${{atk}}</td>`;
      ops.forEach(op => {{
        const key = `${{atk}} | ${{op}}`;
        const coeffs = TASKS[key] || {{}};
        const ae = predict(coeffs, vals);
        const bg = colorForAE(ae);
        const textCol = Math.abs(ae) > 30 ? '#fff' : '#14213d';
        html += `<td style="text-align:center;padding:8px;background:${{bg}};color:${{textCol}};
                   border:1px solid rgba(255,255,255,0.4);font-weight:700;font-size:0.9rem;
                   border-radius:4px;min-width:70px;">${{ae.toFixed(1)}}</td>`;
      }});
      html += '</tr>';
    }});
    html += '</table>';
    grid.innerHTML = html;
    document.getElementById('ae-caption').textContent =
      'Predicted AE = sum of ridge model coefficients × current profile feature values. '
      + 'Values not in scale — exploratory only.';
  }}

  window.resetSliders = function() {{
    SLIDERS.forEach(s => {{
      document.getElementById('sl-' + s.term).value = s.default;
      document.getElementById('val-' + s.term).textContent = s.default;
    }});
    document.getElementById('sex-select').value = 'Male';
    update();
  }};

  buildSliders();
  update();
}})();
</script>
"""


def _fig_violin(long_df: pd.DataFrame) -> go.Figure:
    """Violin + strip plot of AE and abs-delta, faceted by opinion leaf."""
    op_col = "opinion_leaf_label"
    ae_col = "adversarial_effectivity"
    abs_col = "abs_delta_score"
    if op_col not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    opinions = sorted(long_df[op_col].dropna().unique())
    colors   = px.colors.qualitative.Bold[:len(opinions)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Adversarial Effectivity (AE)", "Absolute Opinion Shift"],
        horizontal_spacing=0.10,
    )

    for i, (op, col) in enumerate(zip(opinions, colors)):
        sub = long_df[long_df[op_col] == op]
        lbl = _leaf(op)

        for col_idx, ycol in enumerate([ae_col, abs_col], 1):
            y = sub[ycol].dropna().values
            # violin
            fig.add_trace(go.Violin(
                y=y, name=lbl, legendgroup=lbl, showlegend=(col_idx == 1),
                x0=lbl, fillcolor=col, line_color=col,
                opacity=0.55, box_visible=True, meanline_visible=True,
                points=False,
                hoverinfo="y+name",
            ), row=1, col=col_idx)
            # strip
            rng = np.random.RandomState(i * 10 + col_idx)
            fig.add_trace(go.Scatter(
                x=[lbl] * len(y), y=y,
                mode="markers",
                marker=dict(color=col, size=4, opacity=0.35,
                            line=dict(color="white", width=0.3)),
                showlegend=False, hoverinfo="y+name", name=lbl,
            ), row=1, col=col_idx)

    if "zero_line" not in str(fig.layout):
        fig.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1, row=1, col=1)

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8faff",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif"),
        height=520, violinmode="group",
        title=dict(text="Adversarial Effectivity & Absolute Shift — Distribution by Opinion Leaf", font=dict(size=14)),
        margin=dict(l=60, r=30, t=55, b=80),
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_xaxes(tickangle=-20, tickfont=dict(size=9))
    return fig


def _fig_susceptibility_scatter(
    profile_index_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> go.Figure:
    """
    Scatter: mean absolute shift (x) vs mean adversarial effectivity (y),
    sized by susceptibility index, coloured by index rank.
    """
    if profile_index_df.empty:
        return go.Figure().add_annotation(text="Profile index data unavailable", showarrow=False)

    agg = long_df.groupby("profile_id").agg(
        mean_ae=("adversarial_effectivity", "mean"),
        mean_abs=("abs_delta_score", "mean"),
    ).reset_index()

    merged = profile_index_df.merge(agg, on="profile_id", how="left")
    merged = merged.dropna(subset=["mean_ae", "mean_abs", "susceptibility_index_pct"])

    fig = go.Figure(go.Scatter(
        x=merged["mean_abs"],
        y=merged["mean_ae"],
        mode="markers",
        marker=dict(
            size=merged["susceptibility_index_pct"].fillna(50) / 7 + 6,
            color=merged["susceptibility_index_pct"],
            colorscale="RdBu_r",
            cmin=0, cmax=100,
            line=dict(color="white", width=0.8),
            colorbar=dict(title="Suscept. Index (%)", thickness=13),
            showscale=True,
        ),
        text=merged["profile_id"],
        customdata=merged[["susceptibility_index_pct", "mean_ae", "mean_abs"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Susceptibility: %{customdata[0]:.0f}th pct<br>"
            "Mean AE: %{customdata[1]:.1f}<br>"
            "Mean |Δ|: %{customdata[2]:.1f}<extra></extra>"
        ),
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1,
                  annotation_text="AE = 0", annotation_font_size=9)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8faff",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif"),
        height=520,
        title=dict(text="Profile Susceptibility Map — Mean |Δ| vs Mean AE (size = susceptibility index)", font=dict(size=14)),
        xaxis_title="Mean Absolute Opinion Shift",
        yaxis_title="Mean Adversarial Effectivity",
        margin=dict(l=70, r=30, t=55, b=60),
    )
    return fig


def _fig_moderator_forest(exploratory_df: pd.DataFrame) -> go.Figure:
    """
    Enhanced forest plot: multivariate + univariate estimates side by side,
    FDR marker, confidence interval bars.
    """
    if exploratory_df.empty:
        return go.Figure().add_annotation(text="Moderator data unavailable", showarrow=False)

    df = exploratory_df.copy()
    for col in ["multivariate_estimate", "univariate_estimate", "multivariate_p_value",
                "multivariate_conf_low", "multivariate_conf_high",
                "univariate_conf_low", "univariate_conf_high", "multivariate_q_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("multivariate_estimate", ascending=True).reset_index(drop=True)
    labels = df["moderator_label"].tolist()

    fig = go.Figure()

    # univariate (grey)
    if "univariate_estimate" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["univariate_estimate"], y=labels,
            mode="markers",
            marker=dict(color="#9aaac8", size=9, symbol="circle-open", line=dict(width=2)),
            error_x=dict(
                type="data",
                array=(df.get("univariate_conf_high", df["univariate_estimate"]) - df["univariate_estimate"]).abs().tolist(),
                arrayminus=(df["univariate_estimate"] - df.get("univariate_conf_low", df["univariate_estimate"])).abs().tolist(),
                color="#9aaac8", thickness=1.5, width=5,
            ),
            name="Univariate",
            hovertemplate="%{y}<br>Univariate β=%{x:.3f}<extra></extra>",
        ))

    # multivariate (navy/red)
    mv_colors = [PALETTE["red"] if v < 0 else PALETTE["blue"]
                 for v in df.get("multivariate_estimate", pd.Series([])).tolist()]
    fig.add_trace(go.Scatter(
        x=df["multivariate_estimate"], y=labels,
        mode="markers",
        marker=dict(color=mv_colors, size=11, symbol="diamond",
                    line=dict(color="white", width=1.5)),
        error_x=dict(
            type="data",
            array=(df.get("multivariate_conf_high", df["multivariate_estimate"]) - df["multivariate_estimate"]).abs().tolist(),
            arrayminus=(df["multivariate_estimate"] - df.get("multivariate_conf_low", df["multivariate_estimate"])).abs().tolist(),
            color="#444", thickness=1.8, width=6,
        ),
        name="Multivariate",
        customdata=df[["multivariate_p_value", "multivariate_q_value"]].values
            if "multivariate_q_value" in df.columns else df[["multivariate_p_value"]].values,
        hovertemplate=(
            "%{y}<br>Multivariate β=%{x:.3f}<br>"
            "p=%{customdata[0]:.4f} | q_FDR=%{customdata[1]:.4f}<extra></extra>"
            if "multivariate_q_value" in df.columns
            else "%{y}<br>β=%{x:.3f}<br>p=%{customdata[0]:.4f}<extra></extra>"
        ),
    ))

    # FDR-significant marker
    if "multivariate_q_value" in df.columns:
        sig = df[df["multivariate_q_value"] < 0.05]
        if not sig.empty:
            fig.add_trace(go.Scatter(
                x=sig["multivariate_estimate"], y=sig["moderator_label"],
                mode="markers",
                marker=dict(color="gold", size=18, symbol="star",
                            line=dict(color="#d95d39", width=1.5)),
                name="FDR q<.05",
                hovertemplate="%{y}<br>FDR-significant<extra></extra>",
            ))

    fig.add_vline(x=0, line_dash="dot", line_color="#666", line_width=1)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8faff",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif"),
        height=max(460, 32 * len(df) + 120),
        title=dict(text="Profile Moderator Coefficients — Multivariate & Univariate (diamond=multivariate, circle=univariate, ★=FDR q<.05)", font=dict(size=13)),
        xaxis_title="Coefficient estimate (95% CI)",
        yaxis_title="",
        margin=dict(l=220, r=30, t=65, b=50),
        legend=dict(orientation="h", y=-0.14),
    )
    return fig


def _fig_hierarchical_importance(weight_df: pd.DataFrame) -> go.Figure:
    """Treemap + horizontal bar of hierarchical feature importance."""
    if weight_df.empty:
        return go.Figure().add_annotation(text="Weight data unavailable", showarrow=False)

    df = weight_df.copy()
    df["normalized_weight_pct"] = pd.to_numeric(df["normalized_weight_pct"], errors="coerce").fillna(0)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.42, 0.58],
        subplot_titles=["Group-level importance (treemap)", "Individual feature importance"],
        specs=[[{"type": "treemap"}, {"type": "xy"}]],
    )

    # treemap
    group_sum = (
        df.groupby("ontology_group", as_index=False)["normalized_weight_pct"]
        .sum().rename(columns={"normalized_weight_pct": "weight"})
    )
    fig.add_trace(go.Treemap(
        labels=group_sum["ontology_group"].tolist() + ["Total"],
        parents=["Total"] * len(group_sum) + [""],
        values=group_sum["weight"].tolist() + [group_sum["weight"].sum()],
        textinfo="label+percent parent",
        marker=dict(
            colors=group_sum["weight"].tolist() + [0],
            colorscale="Blues",
            line=dict(width=1.5, color="white"),
        ),
        hovertemplate="%{label}<br>Weight: %{value:.1f}%<extra></extra>",
    ), row=1, col=1)

    # horizontal bar
    df_sorted = df.sort_values("normalized_weight_pct", ascending=True)
    fig.add_trace(go.Bar(
        x=df_sorted["normalized_weight_pct"],
        y=df_sorted.get("moderator_label", df_sorted["ontology_group"]),
        orientation="h",
        marker=dict(
            color=df_sorted["normalized_weight_pct"],
            colorscale="Blues",
            line=dict(color="white", width=0.5),
        ),
        text=df_sorted["normalized_weight_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="%{y}<br>Weight: %{x:.2f}%<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8faff",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif"),
        height=max(520, 22 * len(df) + 160),
        title=dict(text="Hierarchical Feature Importance — Conditional Susceptibility Index", font=dict(size=14)),
        margin=dict(l=180, r=60, t=60, b=50),
        showlegend=False,
    )
    return fig


def _fig_profile_heatmap(long_df: pd.DataFrame, profile_index_df: pd.DataFrame) -> go.Figure:
    """Sorted profile × opinion leaf heatmap of mean adversarial effectivity."""
    op_col, ae_col = "opinion_leaf_label", "adversarial_effectivity"
    if op_col not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    matrix = long_df.pivot_table(
        index="profile_id", columns=op_col, values=ae_col, aggfunc="mean",
    )

    if not profile_index_df.empty and "susceptibility_index_pct" in profile_index_df.columns:
        order = (
            profile_index_df.sort_values("susceptibility_index_pct", ascending=False)["profile_id"].tolist()
        )
        matrix = matrix.reindex([p for p in order if p in matrix.index])

    col_labels = [_leaf(c) for c in matrix.columns]
    n_profiles = len(matrix)

    fig = go.Figure(go.Heatmap(
        z=matrix.fillna(0).values,
        x=col_labels,
        y=matrix.index.tolist(),
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="Mean AE", thickness=13),
        hovertemplate="Profile: %{y}<br>Opinion: %{x}<br>Mean AE: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8faff",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif"),
        height=max(500, 9 * n_profiles + 180),
        title=dict(text="Per-profile Adversarial Effectivity Heatmap — Sorted by Susceptibility Index", font=dict(size=14)),
        xaxis_title="Opinion leaf",
        yaxis=dict(title="Profile (sorted by susceptibility ↓)", tickfont=dict(size=7.5)),
        margin=dict(l=90, r=30, t=55, b=80),
    )
    return fig


def _fig_baseline_post(long_df: pd.DataFrame) -> go.Figure:
    """Scatter with quadrant shading: baseline vs post, coloured by AE."""
    if "baseline_score" not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    df = long_df.copy()
    has_ae = "adversarial_effectivity" in df.columns
    color_vals = df["adversarial_effectivity"].values if has_ae else None

    fig = go.Figure(go.Scatter(
        x=df["baseline_score"],
        y=df["post_score"],
        mode="markers",
        marker=dict(
            size=5,
            color=color_vals if has_ae else PALETTE["blue"],
            colorscale="RdBu_r" if has_ae else None,
            cmid=0 if has_ae else None,
            opacity=0.55,
            line=dict(color="rgba(255,255,255,0.3)", width=0.3),
            colorbar=dict(title="AE", thickness=12) if has_ae else None,
            showscale=has_ae,
        ),
        text=df["opinion_leaf_label"] if "opinion_leaf_label" in df.columns else None,
        hovertemplate="Baseline: %{x}<br>Post: %{y}<br>Leaf: %{text}<extra></extra>",
    ))

    lims = [float(min(df["baseline_score"].min(), df["post_score"].min())) - 20,
            float(max(df["baseline_score"].max(), df["post_score"].max())) + 20]
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines",
        line=dict(color="#666", dash="dash", width=1),
        name="No change", showlegend=True,
    ))

    # quadrant annotations
    mid = (lims[0] + lims[1]) / 2
    for xpos, ypos, txt, clr in [
        (lims[0] + 50, lims[1] - 50, "Moved up", "#2a9d8f"),
        (lims[1] - 50, lims[0] + 50, "Moved down", "#e76f51"),
    ]:
        fig.add_annotation(x=xpos, y=ypos, text=txt,
                           font=dict(size=9, color=clr), showarrow=False)

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8faff",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif"),
        height=520,
        title=dict(text="Baseline vs Post-attack Opinion Scores — Coloured by Adversarial Effectivity", font=dict(size=14)),
        xaxis_title="Baseline opinion score",
        yaxis_title="Post-attack opinion score",
        margin=dict(l=70, r=30, t=55, b=60),
    )
    return fig


# ── dashboard HTML renderer ───────────────────────────────────────────────────

def _render_dashboard_html(
    run_id: str,
    summary_cards: Dict[str, Any],
    figure_divs: List[Tuple[str, str]],  # (title, html_content)
    notes: List[str],
) -> str:
    plotly_js = get_plotlyjs()

    cards_html = "".join(
        f"<div class='card'><div class='label'>{k}</div><div class='value'>{v}</div></div>"
        for k, v in summary_cards.items()
    )

    # group tabs by category
    CATEGORIES = [
        ("📡 Factorial Space",  ["Factorial 3D Surface", "Factorial Heat + Contour"]),
        ("🧠 SEM Analysis",     ["SEM Network", "SEM Heatmap"]),
        ("🔬 Exploration",      ["Perturbation Explorer", "Violin Distributions"]),
        ("👤 Profiles",         ["Susceptibility Map", "Profile Heatmap"]),
        ("📊 Moderators",       ["Moderator Forest", "Hierarchical Importance"]),
        ("📈 Raw Data",         ["Baseline vs Post"]),
    ]

    # index tabs
    tab_index = {title: idx for idx, (title, _) in enumerate(figure_divs)}

    nav_groups = []
    for cat_label, tab_names in CATEGORIES:
        btns = []
        for name in tab_names:
            idx = tab_index.get(name)
            if idx is not None:
                btns.append(f"<button class='tab-btn' data-tab='tab-{idx}'>{name}</button>")
        if btns:
            nav_groups.append(
                f"<div class='nav-group'>"
                f"<div class='nav-group-label'>{cat_label}</div>"
                f"<div class='nav-group-btns'>{''.join(btns)}</div>"
                f"</div>"
            )
    # any unmatched tabs
    categorised = {n for _, names in CATEGORIES for n in names}
    extra_btns = [
        f"<button class='tab-btn' data-tab='tab-{tab_index[t]}'>{t}</button>"
        for t in tab_index if t not in categorised
    ]
    if extra_btns:
        nav_groups.append(
            f"<div class='nav-group'><div class='nav-group-label'>Other</div>"
            f"<div class='nav-group-btns'>{''.join(extra_btns)}</div></div>"
        )

    tab_panels = "".join(
        f"<section id='tab-{idx}' class='tab-panel{' active' if idx==0 else ''}'>"
        f"<h2 class='tab-title'>{title}</h2>{div_html}</section>"
        for idx, (title, div_html) in enumerate(figure_divs)
    )
    notes_html = "".join(f"<li>{n}</li>" for n in notes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{run_id} — Attack-Effectivity Dashboard</title>
<style>
:root{{
  --bg:#0d1b2a;
  --panel:#ffffff;
  --panel-glass:rgba(255,255,255,0.97);
  --ink:#14213d;
  --ink-soft:#4a5d7a;
  --accent:#e76f51;
  --accent2:#1d4e89;
  --ok:#2a9d8f;
  --line:#dbe3ef;
  --shadow:0 12px 36px rgba(0,0,0,0.22);
  --radius:14px;
}}
*{{box-sizing:border-box;}}
body{{
  margin:0;
  background:linear-gradient(135deg,#0d1b2a 0%,#1a3050 60%,#0d1b2a 100%);
  min-height:100vh;
  color:var(--ink);
  font-family:"IBM Plex Sans","Avenir Next","Segoe UI",sans-serif;
}}
.wrap{{max-width:1480px;margin:0 auto;padding:20px 16px 32px;}}

/* ── hero ── */
.hero{{
  background:linear-gradient(120deg,#0d1b2a 0%,#1d4e89 55%,#14213d 100%);
  border:1px solid rgba(255,255,255,0.08);
  border-radius:20px;
  box-shadow:0 20px 60px rgba(0,0,0,0.5);
  padding:24px 28px 20px;
  margin-bottom:18px;
  position:relative;
  overflow:hidden;
}}
.hero::before{{
  content:'';
  position:absolute;inset:0;
  background:radial-gradient(ellipse at 70% 50%,rgba(42,157,143,0.15) 0%,transparent 65%);
  pointer-events:none;
}}
.hero h1{{margin:0 0 6px;font-size:1.55rem;color:#fff;letter-spacing:0.015em;font-weight:700;}}
.hero .sub{{color:#a8c4e8;font-size:0.9rem;max-width:860px;line-height:1.5;}}
.hero .run-badge{{
  display:inline-block;
  background:rgba(255,255,255,0.10);
  border:1px solid rgba(255,255,255,0.18);
  color:#e0eeff;
  border-radius:20px;
  padding:2px 12px;
  font-size:0.78rem;
  font-weight:700;
  letter-spacing:0.06em;
  margin-bottom:8px;
}}

/* ── KPI cards ── */
.cards{{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(145px,1fr));
  gap:10px;
  margin-bottom:16px;
}}
.card{{
  background:var(--panel-glass);
  border:1px solid rgba(255,255,255,0.6);
  border-radius:12px;
  padding:12px 14px;
  box-shadow:0 6px 18px rgba(0,0,0,0.10);
  backdrop-filter:blur(8px);
}}
.card .label{{
  font-size:0.72rem;
  color:var(--ink-soft);
  letter-spacing:0.04em;
  text-transform:uppercase;
  margin-bottom:3px;
  font-weight:600;
}}
.card .value{{
  font-size:1.18rem;
  font-weight:800;
  color:var(--ink);
  font-variant-numeric:tabular-nums;
}}

/* ── nav ── */
.nav-outer{{
  background:rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.10);
  border-radius:12px;
  padding:10px 12px;
  margin-bottom:14px;
  display:flex;
  flex-wrap:wrap;
  gap:8px;
}}
.nav-group{{display:flex;flex-direction:column;gap:4px;}}
.nav-group-label{{
  font-size:0.67rem;
  font-weight:700;
  letter-spacing:0.06em;
  text-transform:uppercase;
  color:#7ba8d4;
  padding-left:2px;
}}
.nav-group-btns{{display:flex;flex-wrap:wrap;gap:4px;}}
.tab-btn{{
  background:rgba(255,255,255,0.08);
  border:1px solid rgba(255,255,255,0.14);
  color:#c5d8f0;
  border-radius:8px;
  padding:5px 11px;
  cursor:pointer;
  font-size:0.80rem;
  font-weight:600;
  transition:all 0.15s;
  white-space:nowrap;
}}
.tab-btn:hover{{background:rgba(255,255,255,0.16);color:#fff;}}
.tab-btn.active{{
  background:var(--accent2);
  border-color:var(--accent2);
  color:#fff;
  box-shadow:0 3px 12px rgba(29,78,137,0.5);
}}

/* ── tab panels ── */
.tab-panel{{
  display:none;
  background:var(--panel-glass);
  border:1px solid rgba(255,255,255,0.7);
  border-radius:var(--radius);
  padding:16px 16px 20px;
  margin-bottom:14px;
  box-shadow:var(--shadow);
  backdrop-filter:blur(12px);
  animation:fadeIn 0.18s ease;
}}
.tab-panel.active{{display:block;}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(4px)}}to{{opacity:1;transform:none}}}}
.tab-title{{
  margin:0 0 12px;
  font-size:1.05rem;
  font-weight:700;
  color:var(--ink);
  display:flex;
  align-items:center;
  gap:8px;
}}

/* ── notes ── */
.notes{{
  background:rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.10);
  border-radius:12px;
  padding:12px 18px;
  margin-top:4px;
}}
.notes h3{{margin:0 0 6px;font-size:0.9rem;color:#a8c4e8;font-weight:700;}}
.notes li{{
  margin:5px 0;
  color:#7ba8d4;
  font-size:0.82rem;
  line-height:1.45;
}}
</style>
<script>{plotly_js}</script>
</head>
<body>
<div class="wrap">

<div class="hero">
  <div class="run-badge">{run_id.upper()}</div>
  <h1>Attack-Effectivity Investigation Dashboard</h1>
  <p class="sub">
    Ontology-driven multi-agent simulation · Full factorial design ·
    Profile-panel SEM · Conditional susceptibility indexing ·
    Hierarchical feature importance decomposition
  </p>
</div>

<div class="cards">{cards_html}</div>

<div class="nav-outer" id="tabs">{''.join(nav_groups)}</div>

{tab_panels}

<div class="notes">
  <h3>Methodological Notes</h3>
  <ul>{notes_html}</ul>
</div>

</div>
<script>
const buttons = Array.from(document.querySelectorAll('.tab-btn'));
const panels  = Array.from(document.querySelectorAll('.tab-panel'));
function activateTab(id) {{
  buttons.forEach(b => b.classList.toggle('active', b.dataset.tab === id));
  panels.forEach(p  => p.classList.toggle('active', p.id === id));
}}
buttons.forEach(b => b.addEventListener('click', () => activateTab(b.dataset.tab)));
if (buttons.length) activateTab(buttons[0].dataset.tab);
</script>
</body>
</html>""".strip()


# ── main entry point ──────────────────────────────────────────────────────────

def generate_research_visuals(
    sem_long_csv_path: str | Path,
    sem_result_json_path: str | Path,
    ols_params_csv_path: str | Path,
    output_dir: str | Path,
    run_id: str,
) -> Dict[str, Any]:
    output_root    = Path(output_dir)
    figures_dir    = output_root / "figures"
    snapshots_dir  = output_root / "data_snapshots"
    figures_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ──
    long_df    = pd.read_csv(sem_long_csv_path)
    sem_result = json.loads(Path(sem_result_json_path).read_text(encoding="utf-8"))
    ols_params = pd.read_csv(ols_params_csv_path)

    stage05_dir = Path(sem_long_csv_path).resolve().parent
    stage06_dir = Path(sem_result_json_path).resolve().parent

    def _load(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    profile_df       = _load(stage05_dir / "profile_level_effectivity.csv")
    profile_index_df = _load(stage06_dir / "profile_susceptibility_index.csv")
    exploratory_df   = _load(stage06_dir / "exploratory_moderator_comparison.csv")
    weight_df        = _load(stage06_dir / "moderator_weight_table.csv")
    task_coeff_df    = _load(stage06_dir / "conditional_susceptibility_task_coefficients.csv")

    sem_coeff_df = pd.DataFrame(sem_result.get("coefficients", []))
    fit_indices  = sem_result.get("fit_indices", {})

    # ── KPI cards ──
    n_profiles  = int(long_df["profile_id"].nunique()) if "profile_id" in long_df.columns else "n/a"
    n_attacks   = int(long_df["attack_leaf"].nunique()) if "attack_leaf" in long_df.columns else "n/a"
    n_opinions  = int(long_df["opinion_leaf"].nunique()) if "opinion_leaf" in long_df.columns else "n/a"
    pct_pos     = (
        f"{(long_df['adversarial_effectivity'] > 0).mean() * 100:.1f}%"
        if "adversarial_effectivity" in long_df.columns else "n/a"
    )
    summary_cards = {
        "Profiles": n_profiles,
        "Total Scenarios": len(long_df),
        "Attack Vectors": n_attacks,
        "Opinion Leaves": n_opinions,
        "Mean |Δ|": f"{long_df['abs_delta_score'].mean():.1f}" if "abs_delta_score" in long_df.columns else "n/a",
        "Mean AE": f"{long_df['adversarial_effectivity'].mean():.1f}" if "adversarial_effectivity" in long_df.columns else "n/a",
        "% AE > 0": pct_pos,
        "Mean Realism": f"{long_df['attack_realism_score'].dropna().mean():.2f}" if "attack_realism_score" in long_df.columns else "n/a",
        "ICC(1) |Δ|": f"{fit_indices.get('icc_abs_delta', 'n/a')}",
        "CFI": f"{fit_indices.get('CFI', float('nan')):.3f}" if fit_indices.get("CFI") is not None else "n/a",
        "RMSEA": f"{fit_indices.get('RMSEA', float('nan')):.3f}" if fit_indices.get("RMSEA") is not None else "n/a",
    }

    # ── build figures ──
    figure_divs: List[Tuple[str, str]] = []
    visual_files: List[str] = []

    def _add(title: str, fig: Optional[go.Figure] = None, html: Optional[str] = None,
             fname: Optional[str] = None) -> None:
        if fig is not None:
            if fname:
                path = figures_dir / fname
                visual_files.append(_save_figure_html(fig, path))
            figure_divs.append((title, fig.to_html(include_plotlyjs=False, full_html=False)))
        elif html is not None:
            figure_divs.append((title, html))

    # 1. Factorial 3D Surface
    _add("Factorial 3D Surface", _fig_factorial_3d(long_df), fname="factorial_3d_surface.html")

    # 2. Factorial Heat + Contour
    _add("Factorial Heat + Contour", _fig_factorial_2d(long_df), fname="factorial_2d_heatmap.html")

    # 3. SEM Network
    if not sem_coeff_df.empty:
        _add("SEM Network", _fig_sem_network(sem_coeff_df), fname="sem_network.html")

    # 4. SEM Heatmap
    if not sem_coeff_df.empty:
        _add("SEM Heatmap", _fig_sem_heatmap(sem_coeff_df, exploratory_df), fname="sem_heatmap.html")

    # 5. Perturbation Explorer
    if not task_coeff_df.empty:
        _add("Perturbation Explorer", html=_html_perturbation_explorer(task_coeff_df, long_df))

    # 6. Violin Distributions
    _add("Violin Distributions", _fig_violin(long_df), fname="violin_distributions.html")

    # 7. Susceptibility Map
    if not profile_index_df.empty:
        _add("Susceptibility Map", _fig_susceptibility_scatter(profile_index_df, long_df),
             fname="susceptibility_map.html")

    # 8. Moderator Forest
    if not exploratory_df.empty:
        _add("Moderator Forest", _fig_moderator_forest(exploratory_df), fname="moderator_forest.html")

    # 9. Hierarchical Importance
    if not weight_df.empty:
        _add("Hierarchical Importance", _fig_hierarchical_importance(weight_df),
             fname="hierarchical_importance.html")

    # 10. Profile Heatmap
    _add("Profile Heatmap", _fig_profile_heatmap(long_df, profile_index_df),
         fname="profile_heatmap.html")

    # 11. Baseline vs Post
    _add("Baseline vs Post", _fig_baseline_post(long_df), fname="baseline_post.html")

    # ── save snapshots ──
    long_df.to_csv(snapshots_dir / "sem_long_encoded_snapshot.csv", index=False)
    if not profile_df.empty:
        profile_df.to_csv(snapshots_dir / "profile_level_effectivity_snapshot.csv", index=False)
    if not profile_index_df.empty:
        profile_index_df.to_csv(snapshots_dir / "profile_susceptibility_snapshot.csv", index=False)
    if not exploratory_df.empty:
        exploratory_df.to_csv(snapshots_dir / "moderator_coefficients_snapshot.csv", index=False)

    notes = [
        "All profiles are attacked; the dashboard visualizes heterogeneity of attacked opinion movement, not a treatment vs control contrast.",
        "Adversarial effectivity (AE = Δ × d_k) is directional: positive = manipulation succeeded, negative = backfire or resistance.",
        "The 3D surface and heatmap show <b>mean AE</b> and <b>inter-individual SD</b> across the full attack × opinion factorial.",
        "The SEM network renders path coefficients as directed edges: width ∝ |β|, color = sign, opacity = significance level.",
        "The Perturbation Explorer uses embedded ridge model coefficients to compute predicted AE client-side from slider inputs.",
        "The susceptibility index is post hoc (fitted from observed data) and therefore descriptive, not an independent predictor.",
        "ICC(1) ≈ 0.052 for |Δ|: attack–opinion context explains ~95% of variance; stable profile traits explain ~5%.",
        "SEM fit (CFI=1.000, RMSEA=0.000) is expected in a near-saturated 4-indicator model at n=100; treat path estimates as exploratory.",
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
