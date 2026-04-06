"""
Interactive attack-effectivity dashboard — next-level visualization.

Tabs (generic for any run):
  📡 Factorial Space   → Factorial 3D Surface, Factorial Heat + Contour
  🧠 SEM Analysis      → SEM Network (interactive), SEM Heatmap
  🔬 Estimation        → Conditional Susceptibility Estimator ★ (new), Perturbation Explorer
  👤 Profiles          → Susceptibility Map, Profile Heatmap
  📊 Moderators        → Moderator Forest, Hierarchical Importance
  📈 Raw Data          → Violin Distributions, Baseline vs Post
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


# ─── palette & helpers ────────────────────────────────────────────────────────

PALETTE = dict(
    navy="#0f2240", blue="#1d4e89", sky="#2980b9",
    teal="#2a9d8f", orange="#e76f51", red="#c0392b",
    amber="#c89b3c", ink="#14213d", muted="#4a5d7a",
    panel="#ffffff", line="#dbe3ef", gold="#f0c040",
)


def _leaf(s: str) -> str:
    raw = s.rsplit(">", 1)[-1].strip() if ">" in str(s) else str(s)
    return raw.replace("_", " ").strip()


def _pretty(s: str) -> str:
    for prefix in ["profile_cont_", "profile_cat__profile_cat_", "profile_cat__"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.replace("_z", "").replace("_", " ").replace("  ", " ").strip().title()


def _p_stars(p: Any) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else ""


def _safe_col(df: pd.DataFrame, col: str, fallback: Any = None):
    return df[col] if col in df.columns else (fallback if fallback is not None else pd.Series(dtype=float))


def _apply_style(fig: go.Figure, height: int = 520) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif", size=12),
        height=height, margin=dict(l=60, r=30, t=52, b=50),
    )
    return fig


def _save_figure_html(fig: go.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    return str(path)


# ─── figure builders ──────────────────────────────────────────────────────────

def _fig_factorial_3d(long_df: pd.DataFrame) -> go.Figure:
    """Dual go.Surface: mean AE (RdBu_r) + ISD of AE (YlOrRd)."""
    atk_col, op_col, ae_col = "attack_leaf_label", "opinion_leaf_label", "adversarial_effectivity"
    for c in (atk_col, op_col, ae_col):
        if c not in long_df.columns:
            return go.Figure().add_annotation(text=f"Column '{c}' missing", showarrow=False)

    attacks  = sorted(long_df[atk_col].dropna().unique())
    opinions = sorted(long_df[op_col].dropna().unique())

    def _matrix(func):
        return (
            long_df.groupby([atk_col, op_col])[ae_col].agg(func)
            .unstack(op_col).reindex(index=attacks, columns=opinions).fillna(0)
        )

    mean_mat = _matrix("mean")
    isd_mat  = _matrix("std")
    atk_l    = [_leaf(a) for a in attacks]
    op_l     = [_leaf(o) for o in opinions]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Mean Adversarial Effectivity (AE)", "Inter-individual Variability (SD of AE)"],
        horizontal_spacing=0.04,
    )

    def _surf(mat, cscale, zlabel):
        z = mat.values.astype(float)
        return go.Surface(
            z=z, x=op_l, y=atk_l,
            colorscale=cscale, cmid=0,
            lighting=dict(ambient=0.75, diffuse=0.9, specular=0.4, roughness=0.4),
            lightposition=dict(x=100, y=200, z=500),
            contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True))),
            hovertemplate=f"<b>%{{y}}</b> → <b>%{{x}}</b><br>{zlabel}: %{{z:.2f}}<extra></extra>",
        )

    fig.add_trace(_surf(mean_mat, "RdBu_r", "Mean AE"), row=1, col=1)
    fig.add_trace(_surf(isd_mat,  "YlOrRd",  "SD AE"),   row=1, col=2)

    cam = dict(eye=dict(x=1.55, y=-1.55, z=1.05))
    for scene in ("scene", "scene2"):
        fig.update_layout(**{scene: dict(
            xaxis=dict(title="Opinion leaf", tickfont=dict(size=8.5), gridcolor="#ccd8ee"),
            yaxis=dict(title="Attack vector", tickfont=dict(size=8.5), gridcolor="#ccd8ee"),
            zaxis=dict(gridcolor="#ccd8ee"),
            camera=cam, bgcolor="white",
        )})
    fig.update_layout(
        paper_bgcolor="white", font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=600, showlegend=False,
        title=dict(text="3D Factorial Surface — Mean AE & Inter-individual Variability", font_size=14),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def _fig_factorial_2d(long_df: pd.DataFrame) -> go.Figure:
    """Side-by-side 2D annotated heatmaps (mean AE | ISD)."""
    atk_col, op_col, ae_col = "attack_leaf_label", "opinion_leaf_label", "adversarial_effectivity"
    if ae_col not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    attacks  = sorted(long_df[atk_col].dropna().unique())
    opinions = sorted(long_df[op_col].dropna().unique())
    atk_l    = [_leaf(a) for a in attacks]
    op_l     = [_leaf(o) for o in opinions]

    mean_m = (long_df.groupby([atk_col, op_col])[ae_col].mean()
              .unstack(op_col).reindex(index=attacks, columns=opinions).fillna(0))
    isd_m  = (long_df.groupby([atk_col, op_col])[ae_col].std()
              .unstack(op_col).reindex(index=attacks, columns=opinions).fillna(0))

    fig = make_subplots(1, 2, subplot_titles=["Mean AE  (red = manipulation succeeded)",
                                               "Inter-individual SD of AE"], horizontal_spacing=0.14)
    common = dict(texttemplate="%{text}", textfont=dict(size=11, color="white"),
                  hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>%{z:.1f}<extra></extra>")

    fig.add_trace(go.Heatmap(z=mean_m.values, x=op_l, y=atk_l,
                             colorscale="RdBu_r", zmid=0,
                             text=[[f"{v:.1f}" for v in row] for row in mean_m.values],
                             colorbar=dict(x=0.44, thickness=11, title="AE", title_side="right"),
                             **common), row=1, col=1)
    fig.add_trace(go.Heatmap(z=isd_m.values, x=op_l, y=atk_l,
                             colorscale="YlOrRd",
                             text=[[f"{v:.1f}" for v in row] for row in isd_m.values],
                             colorbar=dict(x=1.01, thickness=11, title="SD", title_side="right"),
                             **common), row=1, col=2)

    fig.update_xaxes(tickangle=-28, tickfont_size=9)
    fig.update_yaxes(tickfont_size=9)
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="#f4f7ff",
                      font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
                      height=420, margin=dict(l=150, r=90, t=52, b=100),
                      title=dict(text="Factorial Heatmap — Mean AE & Inter-individual Moderation Strength",
                                 font_size=14))
    return fig


def _fig_sem_network(sem_coeff_df: pd.DataFrame) -> go.Figure:
    """
    Bipartite SEM network. Edge: width ∝ |β|, colour = sign, opacity = significance.
    Click a moderator node → highlight only its edges. Click background → reset.
    Buttons: All paths | Significant (p<.05) | Very significant (p<.01) | Reset highlight.
    """
    df = sem_coeff_df[sem_coeff_df["op"] == "~"].copy()
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["p_value"]  = pd.to_numeric(df["p_value"],  errors="coerce")
    df = df.dropna(subset=["estimate"])
    if df.empty:
        return go.Figure().add_annotation(text="No SEM path data", showarrow=False)

    moderators = df["rhs"].unique().tolist()
    indicators = df["lhs"].unique().tolist()
    n_mod, n_ind = len(moderators), len(indicators)

    mod_y  = np.linspace(0.92, 0.08, n_mod)
    ind_y  = np.linspace(0.88, 0.12, n_ind)
    mod_pos = {m: (0.0, float(mod_y[i])) for i, m in enumerate(moderators)}
    ind_pos = {ind: (1.0, float(ind_y[i])) for i, ind in enumerate(indicators)}

    max_abs = df["estimate"].abs().max() or 1.0
    traces: List[go.BaseTraceType] = []
    edge_trace_meta: List[dict] = []   # {rhs, lhs, p, est}

    for _, row in df.iterrows():
        rhs, lhs = str(row["rhs"]), str(row["lhs"])
        if rhs not in mod_pos or lhs not in ind_pos:
            continue
        x0, y0 = mod_pos[rhs]
        x1, y1 = ind_pos[lhs]
        p    = row["p_value"]
        est  = float(row["estimate"])
        stars = _p_stars(p)
        alpha = 0.92 if (pd.notna(p) and p < 0.05) else 0.48 if (pd.notna(p) and p < 0.10) else 0.13
        width = max(0.7, abs(est) / max_abs * 10)
        col   = f"rgba(29,78,137,{alpha})" if est >= 0 else f"rgba(192,57,43,{alpha})"

        traces.append(go.Scatter(
            x=[x0, (x0*2 + x1) / 3, (x0 + x1*2) / 3, x1, None],
            y=[y0, y0, y1, y1, None],
            mode="lines",
            line=dict(color=col, width=width, shape="spline"),
            hovertemplate=(
                f"<b>{rhs}</b> → <b>{_leaf(lhs)}</b><br>"
                f"β = {est:.3f} {stars}<br>p = {p:.4f}<extra></extra>"
            ),
            showlegend=False,
            customdata=[[rhs, lhs, p, est]],
        ))
        edge_trace_meta.append({"rhs": rhs, "lhs": lhs, "p": float(p) if pd.notna(p) else 1.0, "est": est})

    n_edges = len(traces)

    # moderator nodes
    traces.append(go.Scatter(
        x=[0.0] * n_mod, y=[mod_pos[m][1] for m in moderators],
        mode="markers+text",
        marker=dict(size=18, color=PALETTE["navy"], symbol="circle",
                    line=dict(color="white", width=2.5)),
        text=moderators, textposition="middle left", textfont=dict(size=9.5, color=PALETTE["ink"]),
        hovertemplate="<b>%{text}</b><extra>Moderator</extra>",
        name="Moderators", showlegend=False,
    ))
    mod_node_idx = n_edges

    # indicator nodes
    ind_labels = [_leaf(ind) for ind in indicators]
    traces.append(go.Scatter(
        x=[1.0] * n_ind, y=[ind_pos[ind][1] for ind in indicators],
        mode="markers+text",
        marker=dict(size=18, color=PALETTE["teal"], symbol="diamond",
                    line=dict(color="white", width=2.5)),
        text=ind_labels, textposition="middle right", textfont=dict(size=9.5, color=PALETTE["ink"]),
        hovertemplate="<b>%{text}</b><extra>Opinion Indicator</extra>",
        name="Indicators", showlegend=False,
    ))

    fig = go.Figure(traces)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=560,
        xaxis=dict(visible=False, range=[-0.08, 1.08]),
        yaxis=dict(visible=False, range=[0.0, 1.0]),
        title=dict(text="SEM Network — Profile Moderators → Opinion Shift Indicators", font_size=14),
        margin=dict(l=210, r=200, t=60, b=70),
        hovermode="closest",
        annotations=[
            dict(x=0.02, y=1.06, xref="paper", yref="paper", showarrow=False,
                 text="<b>PROFILE MODERATORS</b>", font=dict(size=10.5, color=PALETTE["navy"])),
            dict(x=0.98, y=1.06, xref="paper", yref="paper", showarrow=False,
                 text="<b>OPINION INDICATORS</b>", font=dict(size=10.5, color=PALETTE["teal"])),
            dict(x=0.5, y=-0.08, xref="paper", yref="paper", showarrow=False,
                 text="Width ∝ |β| &nbsp;·&nbsp; <span style='color:#1d4e89'>■</span> positive &nbsp;"
                      "<span style='color:#c0392b'>■</span> negative &nbsp;·&nbsp; "
                      "Opacity: solid p<.05 · medium p<.10 · faint n.s.<br>"
                      "<i>Click a moderator node to highlight its paths — click background to reset</i>",
                 font=dict(size=8.5, color=PALETTE["muted"])),
        ],
        # filter buttons
        updatemenus=[dict(
            type="buttons", direction="right",
            x=0.5, xanchor="center", y=1.14, yanchor="top",
            buttons=[
                dict(label="All paths",
                     method="restyle",
                     args=[{"opacity": [1.0] * n_edges + [1.0, 1.0]}]),
                dict(label="Significant (p<.05)",
                     method="restyle",
                     args=[{"opacity": [
                         1.0 if meta["p"] < 0.05 else 0.06
                         for meta in edge_trace_meta
                     ] + [1.0, 1.0]}]),
                dict(label="Very significant (p<.01)",
                     method="restyle",
                     args=[{"opacity": [
                         1.0 if meta["p"] < 0.01 else 0.06
                         for meta in edge_trace_meta
                     ] + [1.0, 1.0]}]),
                dict(label="Positive only",
                     method="restyle",
                     args=[{"opacity": [
                         1.0 if meta["est"] > 0 else 0.06
                         for meta in edge_trace_meta
                     ] + [1.0, 1.0]}]),
                dict(label="Negative only",
                     method="restyle",
                     args=[{"opacity": [
                         1.0 if meta["est"] < 0 else 0.06
                         for meta in edge_trace_meta
                     ] + [1.0, 1.0]}]),
            ],
            bgcolor=PALETTE["panel"], bordercolor=PALETTE["line"],
            font=dict(size=10), pad=dict(l=4, r=4, t=4, b=4),
        )],
    )
    return fig


def _fig_sem_heatmap(sem_coeff_df: pd.DataFrame, exploratory_df: pd.DataFrame) -> go.Figure:
    df = sem_coeff_df[sem_coeff_df["op"] == "~"].copy()
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["p_value"]  = pd.to_numeric(df["p_value"],  errors="coerce")
    indicators = df["lhs"].unique().tolist()
    hm = df.pivot_table(index="rhs", columns="lhs", values="estimate", aggfunc="mean")

    if not exploratory_df.empty and "normalized_weight_pct" in exploratory_df.columns:
        order = exploratory_df.sort_values("normalized_weight_pct", ascending=False)["moderator_label"].tolist()
        hm = hm.reindex([r for r in order if r in hm.index])

    hm = hm[[c for c in indicators if c in hm.columns]]
    col_labels = [_leaf(c) for c in hm.columns]

    annot, hover = [], []
    for rhs in hm.index:
        row_a, row_h = [], []
        for lhs in hm.columns:
            sub = df[(df["lhs"] == lhs) & (df["rhs"] == rhs)]
            if sub.empty:
                row_a.append("")
                row_h.append(f"{rhs} → {_leaf(lhs)}<br>No data")
            else:
                r = sub.iloc[0]
                row_a.append(f"{r['estimate']:.2f}{_p_stars(r['p_value'])}")
                row_h.append(f"<b>{rhs}</b> → <b>{_leaf(lhs)}</b><br>"
                             f"β = {r['estimate']:.3f} {_p_stars(r['p_value'])}<br>"
                             f"p = {r['p_value']:.4f}")
        annot.append(row_a)
        hover.append(row_h)

    fig = go.Figure(go.Heatmap(
        z=hm.fillna(0).values, x=col_labels, y=list(hm.index),
        colorscale="RdBu_r", zmid=0,
        text=annot, texttemplate="%{text}", textfont=dict(size=10.5),
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title="β", thickness=13),
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=max(420, 38 * len(hm) + 130),
        margin=dict(l=230, r=40, t=52, b=110),
        title=dict(text="SEM Path Coefficients — Moderators → Opinion Indicators (★ p<.05)", font_size=14),
        xaxis=dict(tickangle=-30),
    )
    return fig


def _fig_violin(long_df: pd.DataFrame) -> go.Figure:
    """Violin + strip scatter of AE and |Δ| by opinion leaf."""
    op_col, ae_col, abs_col = "opinion_leaf_label", "adversarial_effectivity", "abs_delta_score"
    if op_col not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    opinions = sorted(long_df[op_col].dropna().unique())
    colors   = px.colors.qualitative.Bold[:max(len(opinions), 4)]

    fig = make_subplots(1, 2,
                        subplot_titles=["Adversarial Effectivity (AE)",
                                        "Absolute Opinion Shift |Δ|"],
                        horizontal_spacing=0.09)

    for i, (op, clr) in enumerate(zip(opinions, colors)):
        sub = long_df[long_df[op_col] == op]
        lbl = _leaf(op)

        for col_idx, ycol in enumerate([ae_col, abs_col], 1):
            if ycol not in sub.columns:
                continue
            y = sub[ycol].dropna().values
            if len(y) == 0:
                continue

            fig.add_trace(go.Violin(
                y=y, x0=lbl, name=lbl, legendgroup=lbl, showlegend=(col_idx == 1),
                fillcolor=clr, line_color=clr, opacity=0.55,
                box=dict(visible=True, width=0.18),
                meanline=dict(visible=True, color="#333"),
                points=False,
                hoverinfo="y+name",
            ), row=1, col=col_idx)

            rng = np.random.RandomState(i * 7 + col_idx)
            jit = rng.uniform(-0.14, 0.14, len(y))
            fig.add_trace(go.Scatter(
                x=[lbl] * len(y), y=y,
                mode="markers",
                marker=dict(color=clr, size=4.5, opacity=0.38,
                            line=dict(color="white", width=0.3)),
                showlegend=False, name=lbl, hoverinfo="y",
            ), row=1, col=col_idx)

    fig.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1, row=1, col=1)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=520, violinmode="group",
        title=dict(text="Distribution of Outcomes by Opinion Leaf", font_size=14),
        margin=dict(l=60, r=30, t=52, b=90),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(tickangle=-20, tickfont_size=9)
    return fig


def _fig_susceptibility_scatter(profile_index_df: pd.DataFrame,
                                long_df: pd.DataFrame) -> go.Figure:
    if profile_index_df.empty:
        return go.Figure().add_annotation(text="Profile index unavailable", showarrow=False)

    agg = long_df.groupby("profile_id").agg(
        mean_ae=("adversarial_effectivity", "mean"),
        mean_abs=("abs_delta_score", "mean"),
    ).reset_index()
    merged = profile_index_df.merge(agg, on="profile_id", how="left").dropna(
        subset=["mean_ae", "mean_abs", "susceptibility_index_pct"])

    fig = go.Figure(go.Scatter(
        x=merged["mean_abs"], y=merged["mean_ae"],
        mode="markers",
        marker=dict(
            size=merged["susceptibility_index_pct"].fillna(50) / 6 + 7,
            color=merged["susceptibility_index_pct"],
            colorscale="RdBu_r", cmin=0, cmax=100,
            opacity=0.82, line=dict(color="white", width=0.8),
            colorbar=dict(title="Susceptibility<br>Index (%)", thickness=13),
            showscale=True,
        ),
        text=merged["profile_id"],
        customdata=np.column_stack([merged["susceptibility_index_pct"],
                                    merged["mean_ae"], merged["mean_abs"]]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Susceptibility: %{customdata[0]:.0f}th pct<br>"
            "Mean AE: %{customdata[1]:.1f}<br>"
            "Mean |Δ|: %{customdata[2]:.1f}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#666", line_width=1,
                  annotation_text="AE = 0 (no net manipulation)", annotation_font_size=9)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=520,
        title=dict(text="Profile Susceptibility Map — size & color = susceptibility index", font_size=14),
        xaxis_title="Mean Absolute Opinion Shift",
        yaxis_title="Mean Adversarial Effectivity",
        margin=dict(l=70, r=30, t=52, b=60),
    )
    return fig


def _fig_moderator_forest(exploratory_df: pd.DataFrame) -> go.Figure:
    if exploratory_df.empty:
        return go.Figure().add_annotation(text="Moderator data unavailable", showarrow=False)

    df = exploratory_df.copy()
    for c in ["multivariate_estimate", "univariate_estimate", "multivariate_p_value",
               "multivariate_conf_low", "multivariate_conf_high",
               "univariate_conf_low", "univariate_conf_high", "multivariate_q_value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("multivariate_estimate", ascending=True).reset_index(drop=True)
    labels = df["moderator_label"].tolist()

    has_uni   = "univariate_estimate"    in df.columns
    has_q     = "multivariate_q_value"  in df.columns
    has_ci_mv = ("multivariate_conf_high" in df.columns and "multivariate_conf_low" in df.columns)
    has_ci_uv = ("univariate_conf_high"   in df.columns and "univariate_conf_low"   in df.columns)

    fig = go.Figure()

    if has_uni:
        fig.add_trace(go.Scatter(
            x=df["univariate_estimate"], y=labels, mode="markers", name="Univariate",
            marker=dict(color="#9aaac8", size=9, symbol="circle-open", line=dict(width=2)),
            error_x=dict(
                type="data",
                array=(df["univariate_conf_high"] - df["univariate_estimate"]).abs() if has_ci_uv else None,
                arrayminus=(df["univariate_estimate"] - df["univariate_conf_low"]).abs() if has_ci_uv else None,
                color="#9aaac8", thickness=1.5, width=4, visible=True,
            ),
            hovertemplate="%{y}<br>Univariate β=%{x:.3f}<extra></extra>",
        ))

    mv_colors = [PALETTE["red"] if v < 0 else PALETTE["blue"]
                 for v in df["multivariate_estimate"].tolist()]
    hov_extra = (df[["multivariate_p_value", "multivariate_q_value"]].values
                 if has_q else df[["multivariate_p_value"]].values)
    hov_tmpl  = ("%{y}<br>β=%{x:.3f}<br>p=%{customdata[0]:.4f}<br>q_FDR=%{customdata[1]:.4f}<extra></extra>"
                 if has_q else "%{y}<br>β=%{x:.3f}<br>p=%{customdata[0]:.4f}<extra></extra>")

    fig.add_trace(go.Scatter(
        x=df["multivariate_estimate"], y=labels, mode="markers", name="Multivariate",
        marker=dict(color=mv_colors, size=12, symbol="diamond",
                    line=dict(color="white", width=1.5)),
        error_x=dict(
            type="data",
            array=(df["multivariate_conf_high"] - df["multivariate_estimate"]).abs() if has_ci_mv else None,
            arrayminus=(df["multivariate_estimate"] - df["multivariate_conf_low"]).abs() if has_ci_mv else None,
            color="#555", thickness=1.8, width=6, visible=True,
        ),
        customdata=hov_extra,
        hovertemplate=hov_tmpl,
    ))

    if has_q:
        sig = df[df["multivariate_q_value"] < 0.05]
        if not sig.empty:
            fig.add_trace(go.Scatter(
                x=sig["multivariate_estimate"], y=sig["moderator_label"],
                mode="markers", name="FDR q<.05",
                marker=dict(color=PALETTE["gold"], size=20, symbol="star",
                            line=dict(color="#d95d39", width=1.5)),
                hovertemplate="%{y}<br>FDR-significant (q<.05)<extra></extra>",
            ))

    fig.add_vline(x=0, line_dash="dot", line_color="#777", line_width=1)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=max(460, 33 * len(df) + 130),
        title=dict(
            text="Moderator Coefficients (◇ multivariate · ○ univariate · ★ FDR q<.05)",
            font_size=13),
        xaxis_title="Coefficient estimate (95% CI)",
        margin=dict(l=230, r=30, t=65, b=50),
        legend=dict(orientation="h", y=-0.13, x=0.5, xanchor="center"),
    )
    return fig


def _fig_hierarchical_importance(weight_df: pd.DataFrame) -> go.Figure:
    if weight_df.empty:
        return go.Figure().add_annotation(text="Weight data unavailable", showarrow=False)

    df = weight_df.copy()
    df["normalized_weight_pct"] = pd.to_numeric(df["normalized_weight_pct"], errors="coerce").fillna(0)
    df_pos = df[df["normalized_weight_pct"] > 0].copy()
    if df_pos.empty:
        return go.Figure().add_annotation(text="All weights zero", showarrow=False)

    group_sum = (df_pos.groupby("ontology_group", as_index=False)["normalized_weight_pct"]
                 .sum().rename(columns={"normalized_weight_pct": "weight"})
                 .sort_values("weight", ascending=False))

    fig = make_subplots(1, 2,
                        column_widths=[0.38, 0.62],
                        subplot_titles=["Feature Group (treemap)", "Individual Feature Importance"],
                        specs=[[{"type": "treemap"}, {"type": "xy"}]])

    labels  = group_sum["ontology_group"].tolist() + ["Root"]
    parents = ["Root"] * len(group_sum)            + [""]
    values  = group_sum["weight"].tolist()         + [group_sum["weight"].sum()]

    fig.add_trace(go.Treemap(
        labels=labels, parents=parents, values=values,
        textinfo="label+percent parent",
        marker=dict(colors=values, colorscale="Blues",
                    line=dict(width=1.5, color="white")),
        hovertemplate="%{label}<br>Weight: %{value:.1f}%<extra></extra>",
        root_color="lightgrey",
    ), row=1, col=1)

    df_s = df_pos.sort_values("normalized_weight_pct", ascending=True)
    bar_labels = (df_s["moderator_label"].tolist()
                  if "moderator_label" in df_s.columns else df_s["ontology_group"].tolist())
    fig.add_trace(go.Bar(
        x=df_s["normalized_weight_pct"], y=bar_labels, orientation="h",
        marker=dict(color=df_s["normalized_weight_pct"], colorscale="Blues",
                    line=dict(color="white", width=0.5)),
        text=df_s["normalized_weight_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="%{y}<br>%{x:.1f}%<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=max(520, 24 * len(df_pos) + 160), showlegend=False,
        title=dict(text="Hierarchical Feature Importance — Conditional Susceptibility Model", font_size=14),
        margin=dict(l=200, r=70, t=60, b=50),
    )
    return fig


def _fig_profile_heatmap(long_df: pd.DataFrame, profile_index_df: pd.DataFrame) -> go.Figure:
    op_col, ae_col = "opinion_leaf_label", "adversarial_effectivity"
    if op_col not in long_df.columns or ae_col not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    matrix = long_df.pivot_table(index="profile_id", columns=op_col,
                                  values=ae_col, aggfunc="mean")
    if not profile_index_df.empty and "susceptibility_index_pct" in profile_index_df.columns:
        order = (profile_index_df.sort_values("susceptibility_index_pct", ascending=False)
                 ["profile_id"].tolist())
        matrix = matrix.reindex([p for p in order if p in matrix.index])

    col_labels = [_leaf(c) for c in matrix.columns]
    n = len(matrix)
    fig = go.Figure(go.Heatmap(
        z=matrix.fillna(0).values, x=col_labels, y=matrix.index.tolist(),
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="Mean AE", thickness=13),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=max(500, 9 * n + 180),
        title=dict(text="Per-profile AE Heatmap — Sorted by Susceptibility Index (↓ most susceptible)",
                   font_size=14),
        xaxis_title="Opinion leaf",
        yaxis=dict(title="Profile ID", tickfont_size=7.5),
        margin=dict(l=95, r=30, t=52, b=80),
    )
    return fig


def _fig_baseline_post(long_df: pd.DataFrame) -> go.Figure:
    if "baseline_score" not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    has_ae = "adversarial_effectivity" in long_df.columns
    has_leaf = "opinion_leaf_label" in long_df.columns

    fig = go.Figure(go.Scatter(
        x=long_df["baseline_score"],
        y=long_df["post_score"],
        mode="markers",
        marker=dict(
            size=5.5, opacity=0.55,
            color=long_df["adversarial_effectivity"].values if has_ae else PALETTE["blue"],
            colorscale="RdBu_r" if has_ae else None,
            cmid=0 if has_ae else None,
            line=dict(color="rgba(255,255,255,0.25)", width=0.3),
            colorbar=dict(title="AE", thickness=12) if has_ae else None,
            showscale=has_ae,
        ),
        text=long_df["opinion_leaf_label"] if has_leaf else None,
        hovertemplate="Baseline: %{x}<br>Post: %{y}<br>%{text}<extra></extra>",
    ))

    lo = float(min(long_df["baseline_score"].min(), long_df["post_score"].min())) - 30
    hi = float(max(long_df["baseline_score"].max(), long_df["post_score"].max())) + 30
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="#666", dash="dash", width=1.2),
                             name="No change", showlegend=True))
    fig.add_annotation(x=lo + 60, y=hi - 30, text="Opinion moved up",
                       font=dict(size=9, color=PALETTE["teal"]), showarrow=False)
    fig.add_annotation(x=hi - 60, y=lo + 30, text="Opinion moved down",
                       font=dict(size=9, color=PALETTE["orange"]), showarrow=False)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=520,
        title=dict(text="Baseline vs Post-attack Opinion — Coloured by Adversarial Effectivity", font_size=14),
        xaxis_title="Baseline opinion score",
        yaxis_title="Post-attack opinion score",
        margin=dict(l=70, r=30, t=52, b=60),
    )
    return fig


# ─── perturbation explorer (simple) ──────────────────────────────────────────

def _html_perturbation_explorer(task_coeff_df: pd.DataFrame,
                                long_df: pd.DataFrame) -> str:
    if task_coeff_df.empty:
        return "<p>No coefficient data available.</p>"

    SLIDERS = [
        ("profile_cont_big_five_agreeableness_mean_pct",         "Agreeableness",    50, 0, 100),
        ("profile_cont_big_five_conscientiousness_mean_pct",     "Conscientiousness", 50, 0, 100),
        ("profile_cont_big_five_extraversion_mean_pct",          "Extraversion",      50, 0, 100),
        ("profile_cont_big_five_neuroticism_mean_pct",           "Neuroticism",       50, 0, 100),
        ("profile_cont_big_five_openness_to_experience_mean_pct","Openness",          50, 0, 100),
        ("profile_cont_chronological_age",                       "Age",               40, 18, 80),
    ]

    tasks_json: dict = {}
    for (ak, ok), grp in task_coeff_df.groupby(["attack_leaf", "opinion_leaf"]):
        key = f"{_leaf(ak)} | {_leaf(ok)}"
        tasks_json[key] = dict(zip(grp["term"].tolist(), grp["estimate"].astype(float).tolist()))

    slider_defs = json.dumps([{"term": t, "label": lbl, "default": d, "min": mn, "max": mx}
                               for t, lbl, d, mn, mx in SLIDERS])

    return f"""
<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;padding:4px 0">
<div style="min-width:250px;flex:0 0 265px">
  <h3 style="margin:0 0 10px;font-size:0.95rem;color:{PALETTE['ink']}">Profile Sliders</h3>
  <div id="pe-sliders" style="display:flex;flex-direction:column;gap:8px;"></div>
  <label style="display:block;margin-top:12px;font-size:0.82rem;font-weight:600;">Sex
    <select id="pe-sex" style="width:100%;margin-top:3px;padding:5px;border-radius:6px;border:1px solid #dbe3ef">
      <option>Male</option><option>Female</option><option>Other</option>
    </select></label>
  <button onclick="pe_reset()" style="margin-top:12px;padding:6px 14px;background:{PALETTE['blue']};color:#fff;border:none;border-radius:7px;cursor:pointer;font-size:0.85rem">Reset</button>
</div>
<div style="flex:1;min-width:320px">
  <h3 style="margin:0 0 4px;font-size:0.95rem;color:{PALETTE['ink']}">Predicted AE Grid</h3>
  <div style="font-size:0.78rem;color:{PALETTE['muted']};margin-bottom:8px">Red = manipulation succeeded · Blue = resistance/backfire</div>
  <div id="pe-grid"></div>
</div>
</div>
<script>
(function(){{
const TASKS={json.dumps(tasks_json)};
const SL={slider_defs};
function vals(){{
  const v={{'Intercept':1}};
  SL.forEach(s=>v[s.term]=parseFloat(document.getElementById('pe-sl-'+s.term).value));
  const sx=document.getElementById('pe-sex').value;
  v['profile_cat__profile_cat_sex_Female']=sx==='Female'?1:0;
  v['profile_cat__profile_cat_sex_Male']=sx==='Male'?1:0;
  v['profile_cat__profile_cat_sex_Other']=sx==='Other'?1:0;
  return v;
}}
function cellBg(ae){{
  const t=Math.max(-1,Math.min(1,ae/60));
  if(t>=0)return `rgb(${{Math.round(190+65*t)}},${{Math.round(60-40*t)}},${{Math.round(60-40*t)}})`;
  const u=-t;return `rgb(${{Math.round(60-40*u)}},${{Math.round(60+60*u)}},${{Math.round(190+65*u)}})`;
}}
function upd(){{
  const v=vals();const atks=new Set(),ops=new Set();
  Object.keys(TASKS).forEach(k=>{{const[a,o]=k.split(' | ');atks.add(a);ops.add(o);}});
  let h='<table style="border-collapse:collapse;font-size:0.82rem">';
  h+='<tr><th style="padding:5px 8px;border-bottom:2px solid #dbe3ef;text-align:left;font-size:0.76rem">Attack \\ Opinion</th>';
  [...ops].forEach(o=>h+=`<th style="padding:5px 7px;border-bottom:2px solid #dbe3ef;font-size:0.76rem;min-width:80px">${{o}}</th>`);
  h+='</tr>';
  [...atks].forEach(a=>{{
    h+=`<tr><td style="padding:5px 8px;font-weight:600;border-right:1px solid #dbe3ef;font-size:0.78rem;white-space:nowrap">${{a}}</td>`;
    [...ops].forEach(o=>{{
      const k=`${{a}} | ${{o}}`;const c=TASKS[k]||{{}};
      let ae=0;Object.entries(c).forEach(([t,e])=>ae+=e*(v[t]||0));
      const bg=cellBg(ae),tc=Math.abs(ae)>28?'#fff':'#14213d';
      h+=`<td style="text-align:center;padding:7px;background:${{bg}};color:${{tc}};border:2px solid rgba(255,255,255,0.4);border-radius:4px;font-weight:700;font-size:0.9rem">${{ae.toFixed(1)}}</td>`;
    }});h+='</tr>';
  }});
  document.getElementById('pe-grid').innerHTML=h+'</table>';
}}
function build(){{
  const c=document.getElementById('pe-sliders');
  SL.forEach(s=>{{
    const d=document.createElement('div');
    d.innerHTML=`<label style="font-size:0.81rem;font-weight:600;color:{PALETTE['ink']};display:flex;justify-content:space-between">
      <span>${{s.label}}</span><span id="pe-v-${{s.term}}">${{s.default}}</span></label>
      <input type="range" id="pe-sl-${{s.term}}" min="${{s.min}}" max="${{s.max}}" value="${{s.default}}"
        style="width:100%;accent-color:{PALETTE['blue']}"
        oninput="document.getElementById('pe-v-${{s.term}}').textContent=this.value;peUpd()">`;
    c.appendChild(d);
  }});
  document.getElementById('pe-sex').onchange=()=>peUpd();
}}
window.peUpd=upd;window.pe_reset=function(){{SL.forEach(s=>{{document.getElementById('pe-sl-'+s.term).value=s.default;document.getElementById('pe-v-'+s.term).textContent=s.default;}});document.getElementById('pe-sex').value='Male';upd();}};
build();upd();
}})();
</script>"""


# ─── conditional susceptibility estimator ────────────────────────────────────

def _html_cs_estimator(
    task_coeff_df: pd.DataFrame,
    task_summary_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> str:
    """
    Full conditional susceptibility estimation tool embedded in the dashboard.
    Features:
    - Profile builder: Big Five (mean + collapsible facets), age, sex, heuristic/resilience
    - Task selector: any subset of attack × opinion combinations
    - Live AE prediction grid (attack × opinion, RdBu colour)
    - Susceptibility gauge (percentile vs re-computed distribution on selected tasks)
    - Radar chart: user profile vs population average (Plotly)
    - Feature contribution waterfall: top positive & negative drivers
    - Load random profile · Reset · Quick presets
    """
    if task_coeff_df.empty:
        return "<p>Conditional susceptibility data unavailable.</p>"

    # ── extract feature metadata ──
    feat_cols = sorted([c for c in long_df.columns
                        if (c.startswith("profile_cont_") or c.startswith("profile_cat__"))
                        and c != "profile_cont_heuristic_shift_sensitivity_proxy"])

    profile_feats_df = long_df.groupby("profile_id")[feat_cols].first().reset_index()
    feat_means = {c: float(profile_feats_df[c].mean()) for c in feat_cols}
    feat_stds  = {c: float(max(profile_feats_df[c].std(), 0.01)) for c in feat_cols}

    # profiles as JSON dict {profile_id: {term: value}}
    profiles_json: dict = {}
    for _, row in profile_feats_df.iterrows():
        profiles_json[str(row["profile_id"])] = {c: float(row[c]) for c in feat_cols
                                                   if not pd.isna(row[c])}

    # task coefficients: {short_label: {term: coeff}}
    tasks_json: dict = {}
    weights_json: dict = {}
    for (ak, ok), grp in task_coeff_df.groupby(["attack_leaf", "opinion_leaf"]):
        key = f"{_leaf(ak)} | {_leaf(ok)}"
        tasks_json[key] = dict(zip(grp["term"].tolist(), grp["estimate"].astype(float).tolist()))

    if not task_summary_df.empty:
        for _, row in task_summary_df.iterrows():
            key = f"{_leaf(str(row['attack_leaf']))} | {_leaf(str(row['opinion_leaf']))}"
            weights_json[key] = float(row.get("reliability_weight", 1.0))
    else:
        weights_json = {k: 1.0 for k in tasks_json}

    all_attacks  = sorted({k.split(" | ")[0] for k in tasks_json})
    all_opinions = sorted({k.split(" | ")[1] for k in tasks_json})

    # Big Five structure for grouped sliders
    BIG5_GROUPS = [
        ("agreeableness", "Agreeableness", [
            "altruism", "compliance", "modesty", "straightforwardness",
            "tender_mindedness", "trust",
        ]),
        ("conscientiousness", "Conscientiousness", [
            "achievement_striving", "competence", "deliberation",
            "dutifulness", "order", "self_discipline",
        ]),
        ("extraversion", "Extraversion", [
            "activity_level", "assertiveness", "excitement_seeking",
            "gregariousness", "positive_emotions", "warmth",
        ]),
        ("neuroticism", "Neuroticism", [
            "anger_hostility", "anxiety", "depression",
            "impulsiveness", "self_consciousness", "stress_vulnerability",
        ]),
        ("openness_to_experience", "Openness to Experience", [
            "actions", "aesthetics", "fantasy",
            "feelings", "ideas", "values",
        ]),
    ]

    radar_labels = [g[1] for g in BIG5_GROUPS]
    radar_means  = [round(feat_means.get(
        f"profile_cont_big_five_{g[0]}_mean_pct", 50.0), 1) for g in BIG5_GROUPS]

    return f"""
<div id="cse-root" style="display:grid;grid-template-columns:300px 1fr;grid-template-rows:auto;gap:16px;align-items:start">

<!-- ══ LEFT: profile builder ══ -->
<div style="display:flex;flex-direction:column;gap:10px">

  <div style="background:#f0f5ff;border-radius:10px;padding:12px 14px">
    <div style="font-weight:700;font-size:0.92rem;color:{PALETTE['navy']};margin-bottom:10px">
      🧬 Profile Configuration
    </div>

    <!-- Big Five groups -->
    {''.join(f"""
    <div class="cse-group" id="cse-g-{g[0]}" style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;align-items:center;cursor:pointer"
           onclick="cse_toggle('{g[0]}')">
        <span style="font-weight:600;font-size:0.83rem;color:{PALETTE['ink']}">{g[1]}</span>
        <span style="display:flex;gap:6px;align-items:center">
          <span id="cse-mv-{g[0]}"
            style="font-size:0.75rem;font-weight:700;color:{PALETTE['blue']};min-width:32px;text-align:right">50</span>
          <span id="cse-arr-{g[0]}" style="font-size:0.7rem;color:{PALETTE['muted']}">▶</span>
        </span>
      </div>
      <!-- mean slider (visible by default) -->
      <input type="range" id="cse-sl-{g[0]}-mean" min="0" max="100" value="50" step="1"
        style="width:100%;margin-top:3px;accent-color:{PALETTE['blue']}"
        oninput="cse_mean_change('{g[0]}',this.value)">
      <!-- facet sliders (hidden by default) -->
      <div id="cse-facets-{g[0]}" style="display:none;margin-top:6px;padding:6px 8px;background:rgba(255,255,255,0.7);border-radius:7px">
        {''.join(f"""
        <div style="margin-bottom:5px">
          <div style="display:flex;justify-content:space-between;font-size:0.76rem;color:{PALETTE['muted']}">
            <span>{facet.replace("_"," ").title()}</span>
            <span id="cse-fv-{g[0]}-{facet}">50</span>
          </div>
          <input type="range" id="cse-sf-{g[0]}-{facet}" min="0" max="100" value="50" step="1"
            style="width:100%;accent-color:{PALETTE['sky']}"
            oninput="cse_facet_change('{g[0]}','{facet}',this.value)">
        </div>""" for facet in g[2])}
      </div>
    </div>""" for g in BIG5_GROUPS)}

    <!-- Age -->
    <div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;font-size:0.83rem;font-weight:600;color:{PALETTE['ink']}">
        <span>Age</span><span id="cse-v-age">40</span>
      </div>
      <input type="range" id="cse-sl-age" min="18" max="80" value="40" step="1"
        style="width:100%;accent-color:{PALETTE['blue']}"
        oninput="document.getElementById('cse-v-age').textContent=this.value;cse_update()">
    </div>

    <!-- Sex -->
    <div style="margin-bottom:6px">
      <label style="font-size:0.83rem;font-weight:600;color:{PALETTE['ink']}">Sex
        <select id="cse-sex" onchange="cse_update()"
          style="width:100%;margin-top:3px;padding:5px;border-radius:6px;border:1px solid #dbe3ef;font-size:0.88rem">
          <option>Male</option><option>Female</option><option>Other</option>
        </select>
      </label>
    </div>

    <!-- Action buttons -->
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px">
      <button onclick="cse_reset()" style="flex:1;padding:6px;background:{PALETTE['blue']};color:#fff;border:none;border-radius:7px;cursor:pointer;font-size:0.80rem;font-weight:600">Reset</button>
      <button onclick="cse_random()" style="flex:1;padding:6px;background:{PALETTE['teal']};color:#fff;border:none;border-radius:7px;cursor:pointer;font-size:0.80rem;font-weight:600">Random</button>
      <button onclick="cse_preset('high_c')" style="flex:1;padding:6px;background:#f0f0f0;color:{PALETTE['ink']};border:1px solid #dbe3ef;border-radius:7px;cursor:pointer;font-size:0.75rem">High C</button>
      <button onclick="cse_preset('low_c')"  style="flex:1;padding:6px;background:#f0f0f0;color:{PALETTE['ink']};border:1px solid #dbe3ef;border-radius:7px;cursor:pointer;font-size:0.75rem">Low C</button>
    </div>
  </div>

  <!-- Task selector -->
  <div style="background:#f0f5ff;border-radius:10px;padding:12px 14px">
    <div style="font-weight:700;font-size:0.92rem;color:{PALETTE['navy']};margin-bottom:8px">
      🎯 Task Selector
    </div>
    <div style="font-size:0.78rem;font-weight:600;color:{PALETTE['muted']};margin-bottom:5px">
      Attack vectors
      <a onclick="cse_all_atk(true)" style="cursor:pointer;color:{PALETTE['blue']};margin-left:6px">All</a>
      <a onclick="cse_all_atk(false)" style="cursor:pointer;color:{PALETTE['orange']};margin-left:4px">None</a>
    </div>
    <div id="cse-atk-checks" style="display:flex;flex-direction:column;gap:4px;margin-bottom:10px">
      {''.join(f"""<label style="font-size:0.82rem;display:flex;align-items:center;gap:6px;cursor:pointer">
        <input type="checkbox" checked id="cse-atk-{i}" onchange="cse_update()" style="accent-color:{PALETTE['blue']}">
        {atk}</label>""" for i, atk in enumerate(all_attacks))}
    </div>
    <div style="font-size:0.78rem;font-weight:600;color:{PALETTE['muted']};margin-bottom:5px">
      Opinion domains
      <a onclick="cse_all_op(true)" style="cursor:pointer;color:{PALETTE['blue']};margin-left:6px">All</a>
      <a onclick="cse_all_op(false)" style="cursor:pointer;color:{PALETTE['orange']};margin-left:4px">None</a>
    </div>
    <div id="cse-op-checks" style="display:flex;flex-direction:column;gap:4px">
      {''.join(f"""<label style="font-size:0.82rem;display:flex;align-items:center;gap:6px;cursor:pointer">
        <input type="checkbox" checked id="cse-op-{i}" onchange="cse_update()" style="accent-color:{PALETTE['blue']}">
        {op}</label>""" for i, op in enumerate(all_opinions))}
    </div>
  </div>

</div><!-- end left panel -->

<!-- ══ RIGHT: results ══ -->
<div style="display:flex;flex-direction:column;gap:14px">

  <!-- AE Grid -->
  <div style="background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:14px">
    <div style="font-weight:700;font-size:0.88rem;color:{PALETTE['navy']};margin-bottom:8px">
      📊 Predicted AE per Task
      <span style="font-weight:400;font-size:0.76rem;color:{PALETTE['muted']};margin-left:8px">
        red = manipulation succeeded · blue = backfire
      </span>
    </div>
    <div id="cse-grid"></div>
  </div>

  <!-- Gauge + Radar side by side -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">

    <div style="background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:14px">
      <div style="font-weight:700;font-size:0.88rem;color:{PALETTE['navy']};margin-bottom:8px">
        🎯 Susceptibility Score
      </div>
      <div id="cse-gauge-wrap" style="text-align:center;padding:6px 0"></div>
      <div id="cse-gauge-text" style="text-align:center;font-size:0.78rem;color:{PALETTE['muted']};margin-top:4px"></div>
    </div>

    <div style="background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:14px">
      <div style="font-weight:700;font-size:0.88rem;color:{PALETTE['navy']};margin-bottom:4px">
        🕸 Profile Radar
        <span style="font-size:0.72rem;font-weight:400;color:{PALETTE['muted']}"> vs population avg</span>
      </div>
      <div id="cse-radar" style="height:200px"></div>
    </div>

  </div>

  <!-- Feature contributions -->
  <div style="background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:14px">
    <div style="font-weight:700;font-size:0.88rem;color:{PALETTE['navy']};margin-bottom:8px">
      📈 Feature Contributions to Susceptibility
      <span style="font-size:0.72rem;font-weight:400;color:{PALETTE['muted']}">
        (vs population mean · shows marginal effect)
      </span>
    </div>
    <div id="cse-contrib" style="font-size:0.82rem"></div>
  </div>

</div><!-- end right panel -->
</div><!-- end grid -->

<script>
(function(){{
// ── embedded data ──────────────────────────────────────────────────────────
const TASKS    = {json.dumps(tasks_json)};
const WEIGHTS  = {json.dumps(weights_json)};
const PROFILES = {json.dumps(profiles_json)};
const FEAT_MEANS = {json.dumps(feat_means)};
const RADAR_LABELS = {json.dumps(radar_labels)};
const RADAR_MEANS  = {json.dumps(radar_means)};
const ALL_ATTACKS  = {json.dumps(all_attacks)};
const ALL_OPINIONS = {json.dumps(all_opinions)};
const B5_GROUPS    = {json.dumps([g[0] for g in BIG5_GROUPS])};
const B5_LABELS    = {json.dumps([g[1] for g in BIG5_GROUPS])};
const B5_FACETS    = {json.dumps({g[0]: g[2] for g in BIG5_GROUPS})};
const BIG5_NAMES   = {json.dumps({g[0]: f"profile_cont_big_five_{g[0]}_mean_pct" for g in BIG5_GROUPS})};
const BIG5_FACET_NAMES = {json.dumps(
    {g[0]: {f: f"profile_cont_big_five_{g[0]}_{f}_pct" for f in g[2]} for g in BIG5_GROUPS})};

let _radar_initialized = false;

// ── helpers ────────────────────────────────────────────────────────────────
function getVals() {{
  const v = {{'Intercept': 1}};
  // Big Five means/facets
  B5_GROUPS.forEach(g => {{
    const facetDiv = document.getElementById('cse-facets-'+g);
    const facetsVisible = facetDiv && facetDiv.style.display !== 'none';
    if (facetsVisible) {{
      // use facet values, compute mean
      let fsum = 0, fn = 0;
      (B5_FACETS[g] || []).forEach(f => {{
        const fv = parseFloat(document.getElementById('cse-sf-'+g+'-'+f)?.value || 50);
        v[BIG5_FACET_NAMES[g][f]] = fv;
        fsum += fv; fn++;
      }});
      const meanV = fn > 0 ? fsum / fn : 50;
      v[BIG5_NAMES[g]] = meanV;
      document.getElementById('cse-mv-'+g).textContent = meanV.toFixed(0);
      document.getElementById('cse-sl-'+g+'-mean').value = meanV.toFixed(0);
    }} else {{
      const mv = parseFloat(document.getElementById('cse-sl-'+g+'-mean')?.value || 50);
      v[BIG5_NAMES[g]] = mv;
      // also set facets to mean value (so they're consistent)
      (B5_FACETS[g] || []).forEach(f => {{
        v[BIG5_FACET_NAMES[g][f]] = mv;
      }});
    }}
  }});
  // age
  v['profile_cont_chronological_age'] = parseFloat(document.getElementById('cse-sl-age').value || 40);
  // sex
  const sx = document.getElementById('cse-sex').value;
  v['profile_cat__profile_cat_sex_Female'] = sx==='Female'?1:0;
  v['profile_cat__profile_cat_sex_Male']   = sx==='Male'  ?1:0;
  v['profile_cat__profile_cat_sex_Other']  = sx==='Other' ?1:0;
  return v;
}}

function getSelectedTasks() {{
  const atks = ALL_ATTACKS.filter((_,i) => document.getElementById('cse-atk-'+i)?.checked);
  const ops  = ALL_OPINIONS.filter((_,i) => document.getElementById('cse-op-'+i)?.checked);
  return Object.keys(TASKS).filter(k => {{
    const [a,o] = k.split(' | ');
    return atks.includes(a) && ops.includes(o);
  }});
}}

function predictAE(pf, taskKey) {{
  const c = TASKS[taskKey] || {{}};
  let ae = 0;
  Object.entries(c).forEach(([t,e]) => ae += e * (pf[t] ?? 0));
  return ae;
}}

function computeScore(pf, selectedKeys) {{
  if (selectedKeys.length === 0) return {{ae_map:{{}}, raw:0, pct:50, dist:[]}};
  const ae_map = {{}};
  let wsum = 0, wtot = 0;
  selectedKeys.forEach(k => {{
    const ae = predictAE(pf, k);
    ae_map[k] = ae;
    const w = WEIGHTS[k] || 1;
    wsum += ae * w; wtot += w;
  }});
  const raw = wtot > 0 ? wsum / wtot : 0;

  // distribution: re-score all 100 original profiles on selected tasks
  const dist = Object.values(PROFILES).map(pfOrig => {{
    let ws=0, wt=0;
    selectedKeys.forEach(k => {{ const w=WEIGHTS[k]||1; ws+=predictAE(pfOrig,k)*w; wt+=w; }});
    return wt > 0 ? ws/wt : 0;
  }}).sort((a,b)=>a-b);

  const below = dist.filter(v => v <= raw).length;
  const pct   = Math.round(below / dist.length * 100);
  return {{ae_map, raw, pct, dist}};
}}

function aeColor(ae) {{
  const t = Math.max(-1, Math.min(1, ae/60));
  if (t>=0) return `rgb(${{Math.round(190+65*t)}},${{Math.round(55-30*t)}},${{Math.round(55-30*t)}})`;
  const u=-t; return `rgb(${{Math.round(55-30*u)}},${{Math.round(70+60*u)}},${{Math.round(190+65*u)}})`;
}}

function renderGrid(ae_map) {{
  const selAtks = ALL_ATTACKS.filter((_,i) => document.getElementById('cse-atk-'+i)?.checked);
  const selOps  = ALL_OPINIONS.filter((_,i) => document.getElementById('cse-op-'+i)?.checked);
  if (!selAtks.length || !selOps.length) {{
    document.getElementById('cse-grid').innerHTML='<p style="color:#aaa;font-size:0.82rem">Select at least one attack and one opinion.</p>';
    return;
  }}
  let h='<table style="border-collapse:collapse;font-size:0.82rem;width:100%">';
  h+=`<tr><th style="padding:6px 10px;border-bottom:2px solid #dbe3ef;text-align:left;font-size:0.75rem;color:{PALETTE['muted']}">Attack \\ Opinion</th>`;
  selOps.forEach(o=>h+=`<th style="padding:6px 8px;border-bottom:2px solid #dbe3ef;font-size:0.75rem;color:{PALETTE['muted']};min-width:90px">${{o}}</th>`);
  h+='</tr>';
  selAtks.forEach(a=>{{
    h+=`<tr><td style="padding:7px 10px;font-weight:600;border-right:1px solid #dbe3ef;white-space:nowrap;font-size:0.80rem;color:{PALETTE['ink']}">${{a}}</td>`;
    selOps.forEach(o=>{{
      const k=`${{a}} | ${{o}}`;
      const ae = ae_map[k] ?? 0;
      const bg=aeColor(ae), tc=Math.abs(ae)>25?'#fff':'{PALETTE['ink']}';
      const lbl = ae>5?'↑ succ':ae<-5?'↓ back':'≈ neut';
      h+=`<td style="text-align:center;padding:8px 6px;background:${{bg}};color:${{tc}};border:2px solid rgba(255,255,255,0.35);border-radius:5px;font-weight:700;font-size:0.92rem" title="${{k}}">
        ${{ae.toFixed(1)}}<br><span style="font-size:0.65rem;opacity:0.85">${{lbl}}</span></td>`;
    }});
    h+='</tr>';
  }});
  document.getElementById('cse-grid').innerHTML=h+'</table>';
}}

function renderGauge(pct, raw, nProfiles) {{
  const gc = pct<33?'{PALETTE['teal']}':pct<67?'{PALETTE['amber']}':'{PALETTE['red']}';
  const label = pct>=75?'High susceptibility':pct>=50?'Moderate-high':pct>=25?'Moderate-low':'Low susceptibility';
  document.getElementById('cse-gauge-wrap').innerHTML=`
    <div style="position:relative;display:inline-block;width:180px">
      <div style="background:#e8edf5;border-radius:20px;height:18px;overflow:hidden;width:180px">
        <div style="background:${{gc}};height:100%;width:${{pct}}%;border-radius:20px;transition:width 0.35s ease"></div>
      </div>
      <div style="margin-top:8px;font-size:1.6rem;font-weight:800;color:${{gc}}">${{pct}}th</div>
      <div style="font-size:0.78rem;color:{PALETTE['muted']}">${{label}}</div>
      <div style="font-size:0.72rem;color:{PALETTE['muted']};margin-top:2px">raw score: ${{raw.toFixed(1)}}</div>
    </div>`;
  document.getElementById('cse-gauge-text').textContent=
    `Ranked ${{pct}}th percentile vs ${{nProfiles}} original profiles (task-subset re-scored)`;
}}

function renderRadar(pf) {{
  const userVals = B5_GROUPS.map(g => pf[BIG5_NAMES[g]] ?? 50);
  const closed_u = [...userVals, userVals[0]];
  const closed_m = [...RADAR_MEANS, RADAR_MEANS[0]];
  const closed_l = [...RADAR_LABELS, RADAR_LABELS[0]];
  const data = [
    {{type:'scatterpolar',r:closed_u,theta:closed_l,fill:'toself',name:'Your profile',
      fillcolor:'rgba(29,78,137,0.18)',line:{{color:'{PALETTE['blue']}',width:2}}}},
    {{type:'scatterpolar',r:closed_m,theta:closed_l,fill:'toself',name:'Population avg',
      fillcolor:'rgba(42,157,143,0.12)',line:{{color:'{PALETTE['teal']}',width:2,dash:'dot'}}}}
  ];
  const layout = {{
    polar:{{radialaxis:{{visible:true,range:[0,100],tickfont:{{size:8}}}},
            angularaxis:{{tickfont:{{size:9}}}}}},
    showlegend:true,legend:{{x:0.5,xanchor:'center',y:-0.15,orientation:'h',font:{{size:8}}}},
    margin:{{l:30,r:30,t:10,b:30}},paper_bgcolor:'white',font_family:'IBM Plex Sans,sans-serif',
    height:200,
  }};
  if (!_radar_initialized) {{
    Plotly.newPlot('cse-radar', data, layout, {{displayModeBar:false, responsive:true}});
    _radar_initialized = true;
  }} else {{
    Plotly.react('cse-radar', data, layout);
  }}
}}

function renderContrib(pf, selectedKeys) {{
  const contribs = {{}};
  selectedKeys.forEach(k => {{
    const c = TASKS[k] || {{}};
    Object.entries(c).forEach(([term, coef]) => {{
      if (term==='Intercept') return;
      const delta = (pf[term]??0) - (FEAT_MEANS[term]??0);
      const contrib = coef * delta;
      if (!contribs[term]) contribs[term] = 0;
      contribs[term] += contrib;
    }});
  }});
  const sorted = Object.entries(contribs).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,10);
  const maxAbs = Math.max(...sorted.map(([,v])=>Math.abs(v)), 0.01);

  let h='<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">';
  const pos = sorted.filter(([,v])=>v>0).slice(0,5);
  const neg = sorted.filter(([,v])=>v<0).slice(0,5);

  function renderSide(items, title, color) {{
    let s=`<div><div style="font-weight:700;font-size:0.78rem;color:${{color}};margin-bottom:5px">${{title}}</div>`;
    items.forEach(([term,val])=>{{
      const lbl=term.replace('profile_cont_','').replace('profile_cat__profile_cat_','').replace(/_/g,' ').replace(' z','').replace('big five ','').trim();
      const w=Math.round(Math.abs(val)/maxAbs*100);
      s+=`<div style="margin-bottom:4px">
        <div style="display:flex;justify-content:space-between;font-size:0.76rem">
          <span style="color:{PALETTE['ink']};text-overflow:ellipsis;overflow:hidden;max-width:140px" title="${{lbl}}">${{lbl}}</span>
          <span style="font-weight:700;color:${{color}}">${{val>0?'+':''}}${{val.toFixed(1)}}</span>
        </div>
        <div style="background:#f0f0f5;border-radius:4px;height:6px;overflow:hidden">
          <div style="background:${{color}};height:100%;width:${{w}}%;border-radius:4px"></div>
        </div></div>`;
    }});
    return s+'</div>';
  }}

  h+=renderSide(pos, '↑ Increases susceptibility', '{PALETTE['red']}');
  h+=renderSide(neg, '↓ Decreases susceptibility', '{PALETTE['teal']}');
  document.getElementById('cse-contrib').innerHTML=h+'</div>';
}}

// ── main update ─────────────────────────────────────────────────────────────
function cse_update() {{
  const pf   = getVals();
  const keys = getSelectedTasks();
  const {{ae_map, raw, pct, dist}} = computeScore(pf, keys);
  renderGrid(ae_map);
  renderGauge(pct, raw, dist.length);
  renderRadar(pf);
  renderContrib(pf, keys);
}}
window.cse_update = cse_update;

// ── interactions ────────────────────────────────────────────────────────────
window.cse_toggle = function(g) {{
  const fd = document.getElementById('cse-facets-'+g);
  const arr = document.getElementById('cse-arr-'+g);
  const ms = document.getElementById('cse-sl-'+g+'-mean');
  if (fd.style.display==='none') {{
    fd.style.display='block'; arr.textContent='▼'; ms.style.opacity='0.4';
    // sync facets to current mean
    const mv = parseFloat(ms.value);
    (B5_FACETS[g]||[]).forEach(f => {{
      const el=document.getElementById('cse-sf-'+g+'-'+f);
      if(el){{el.value=mv;document.getElementById('cse-fv-'+g+'-'+f).textContent=mv;}}
    }});
  }} else {{
    fd.style.display='none'; arr.textContent='▶'; ms.style.opacity='1';
  }}
  cse_update();
}};

window.cse_mean_change = function(g, val) {{
  document.getElementById('cse-mv-'+g).textContent = val;
  cse_update();
}};

window.cse_facet_change = function(g, f, val) {{
  document.getElementById('cse-fv-'+g+'-'+f).textContent = val;
  // recompute mean from all facets
  const facets = B5_FACETS[g]||[];
  let sum=0;
  facets.forEach(ff=>{{
    sum+=parseFloat(document.getElementById('cse-sf-'+g+'-'+ff)?.value||50);
  }});
  const mean = (sum/facets.length).toFixed(0);
  document.getElementById('cse-mv-'+g).textContent=mean;
  document.getElementById('cse-sl-'+g+'-mean').value=mean;
  cse_update();
}};

window.cse_reset = function() {{
  B5_GROUPS.forEach(g=>{{
    const el=document.getElementById('cse-sl-'+g+'-mean');
    if(el){{el.value=50;document.getElementById('cse-mv-'+g).textContent=50;}}
    document.getElementById('cse-facets-'+g).style.display='none';
    document.getElementById('cse-arr-'+g).textContent='▶';
    if(el)el.style.opacity='1';
    (B5_FACETS[g]||[]).forEach(f=>{{
      const fe=document.getElementById('cse-sf-'+g+'-'+f);
      if(fe){{fe.value=50;document.getElementById('cse-fv-'+g+'-'+f).textContent=50;}}
    }});
  }});
  document.getElementById('cse-sl-age').value=40;
  document.getElementById('cse-v-age').textContent=40;
  document.getElementById('cse-sex').value='Male';
  cse_update();
}};

window.cse_random = function() {{
  const pids=Object.keys(PROFILES);
  const pf=PROFILES[pids[Math.floor(Math.random()*pids.length)]];
  B5_GROUPS.forEach(g=>{{
    const term=BIG5_NAMES[g]; const v=Math.round(pf[term]??50);
    document.getElementById('cse-sl-'+g+'-mean').value=v;
    document.getElementById('cse-mv-'+g).textContent=v;
    (B5_FACETS[g]||[]).forEach(f=>{{
      const ft=BIG5_FACET_NAMES[g][f]; const fv=Math.round(pf[ft]??v);
      const fe=document.getElementById('cse-sf-'+g+'-'+f);
      if(fe){{fe.value=fv;document.getElementById('cse-fv-'+g+'-'+f).textContent=fv;}}
    }});
  }});
  const age=Math.round(pf['profile_cont_chronological_age']??40);
  document.getElementById('cse-sl-age').value=age;
  document.getElementById('cse-v-age').textContent=age;
  const sex=pf['profile_cat__profile_cat_sex_Female']>0.5?'Female':pf['profile_cat__profile_cat_sex_Other']>0.5?'Other':'Male';
  document.getElementById('cse-sex').value=sex;
  cse_update();
}};

window.cse_preset = function(name) {{
  cse_reset();
  if(name==='high_c'){{document.getElementById('cse-sl-conscientiousness-mean').value=85;document.getElementById('cse-mv-conscientiousness').textContent=85;}}
  if(name==='low_c'){{document.getElementById('cse-sl-conscientiousness-mean').value=15;document.getElementById('cse-mv-conscientiousness').textContent=15;}}
  cse_update();
}};

window.cse_all_atk = function(sel){{ALL_ATTACKS.forEach((_,i)=>{{const el=document.getElementById('cse-atk-'+i);if(el)el.checked=sel;}});cse_update();}};
window.cse_all_op  = function(sel){{ALL_OPINIONS.forEach((_,i)=>{{const el=document.getElementById('cse-op-'+i);if(el)el.checked=sel;}});cse_update();}};

// initialise
cse_update();
}})();
</script>"""


# ─── dashboard HTML ───────────────────────────────────────────────────────────

def _render_dashboard_html(
    run_id: str,
    summary_cards: Dict[str, Any],
    figure_divs: List[Tuple[str, str]],
    notes: List[str],
) -> str:
    plotly_js = get_plotlyjs()

    cards_html = "".join(
        f"<div class='card'><div class='label'>{k}</div><div class='value'>{v}</div></div>"
        for k, v in summary_cards.items()
    )

    CATEGORIES = [
        ("📡 Factorial Space",   ["Factorial 3D Surface", "Factorial Heat + Contour"]),
        ("🧠 SEM Analysis",      ["SEM Network", "SEM Heatmap"]),
        ("🔬 Estimation",        ["Conditional Susceptibility Estimator", "Perturbation Explorer"]),
        ("👤 Profiles",          ["Susceptibility Map", "Profile Heatmap"]),
        ("📊 Moderators",        ["Moderator Forest", "Hierarchical Importance"]),
        ("📈 Raw Data",          ["Violin Distributions", "Baseline vs Post"]),
    ]

    tab_index = {title: idx for idx, (title, _) in enumerate(figure_divs)}
    categorised: set = {n for _, ns in CATEGORIES for n in ns}

    nav_groups = []
    for cat_label, tab_names in CATEGORIES:
        btns = [f"<button class='tab-btn' data-tab='tab-{tab_index[n]}'>{n}</button>"
                for n in tab_names if n in tab_index]
        if btns:
            nav_groups.append(
                f"<div class='nav-group'><div class='nav-label'>{cat_label}</div>"
                f"<div class='nav-btns'>{''.join(btns)}</div></div>")

    extra = [f"<button class='tab-btn' data-tab='tab-{tab_index[t]}'>{t}</button>"
             for t in tab_index if t not in categorised]
    if extra:
        nav_groups.append(
            f"<div class='nav-group'><div class='nav-label'>Other</div>"
            f"<div class='nav-btns'>{''.join(extra)}</div></div>")

    panels = "".join(
        f"<section id='tab-{i}' class='tab-panel{' active' if i==0 else ''}'>"
        f"<h2 class='tab-title'>{t}</h2>{h}</section>"
        for i, (t, h) in enumerate(figure_divs)
    )
    notes_html = "".join(f"<li>{n}</li>" for n in notes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{run_id} — Cognitive Warfare Susceptibility Dashboard</title>
<style>
:root{{
  --bg:#0a1628;--panel:#fff;--glass:rgba(255,255,255,0.97);
  --ink:#14213d;--muted:#4a5d7a;--accent:#e76f51;--blue:#1d4e89;--teal:#2a9d8f;
  --line:#dbe3ef;--shadow:0 12px 40px rgba(0,0,0,0.26);--r:14px;
}}
*{{box-sizing:border-box;}}
body{{margin:0;min-height:100vh;color:var(--ink);
  background:linear-gradient(150deg,#080f1e 0%,#0f2240 45%,#061020 100%);
  font-family:"IBM Plex Sans","Avenir Next","Segoe UI",sans-serif;}}
.wrap{{max-width:1500px;margin:0 auto;padding:18px 14px 30px;}}

/* hero */
.hero{{
  background:linear-gradient(118deg,#0a1628 0%,#1d4e89 55%,#12213a 100%);
  border:1px solid rgba(255,255,255,0.07);border-radius:20px;
  box-shadow:0 24px 64px rgba(0,0,0,0.55);padding:22px 26px 18px;
  margin-bottom:14px;position:relative;overflow:hidden;
}}
.hero::before{{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at 75% 50%,rgba(42,157,143,0.14) 0%,transparent 62%);
  pointer-events:none;}}
.hero h1{{margin:0 0 5px;font-size:1.50rem;color:#fff;font-weight:700;letter-spacing:0.01em;}}
.hero .sub{{color:#a4c3e8;font-size:0.87rem;line-height:1.5;max-width:900px;}}
.badge{{display:inline-block;background:rgba(255,255,255,0.09);border:1px solid rgba(255,255,255,0.18);
  color:#d6eaff;border-radius:20px;padding:2px 11px;font-size:0.74rem;font-weight:700;
  letter-spacing:0.06em;margin-bottom:7px;}}

/* cards */
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-bottom:14px;}}
.card{{background:var(--glass);border:1px solid rgba(255,255,255,0.6);border-radius:11px;
  padding:11px 13px;box-shadow:0 4px 16px rgba(0,0,0,0.08);backdrop-filter:blur(6px);}}
.card .label{{font-size:0.70rem;color:var(--muted);letter-spacing:0.04em;
  text-transform:uppercase;margin-bottom:2px;font-weight:600;}}
.card .value{{font-size:1.10rem;font-weight:800;color:var(--ink);font-variant-numeric:tabular-nums;}}

/* nav */
.nav-outer{{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.09);
  border-radius:11px;padding:9px 11px;margin-bottom:12px;
  display:flex;flex-wrap:wrap;gap:10px;}}
.nav-group{{display:flex;flex-direction:column;gap:3px;}}
.nav-label{{font-size:0.64rem;font-weight:700;letter-spacing:0.06em;
  text-transform:uppercase;color:#6899cc;padding-left:1px;}}
.nav-btns{{display:flex;flex-wrap:wrap;gap:4px;}}
.tab-btn{{background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);
  color:#bdd4ee;border-radius:7px;padding:5px 10px;cursor:pointer;
  font-size:0.79rem;font-weight:600;transition:all 0.15s;white-space:nowrap;}}
.tab-btn:hover{{background:rgba(255,255,255,0.15);color:#fff;}}
.tab-btn.active{{background:var(--blue);border-color:var(--blue);color:#fff;
  box-shadow:0 3px 14px rgba(29,78,137,0.55);}}

/* panels */
.tab-panel{{display:none;background:var(--glass);border:1px solid rgba(255,255,255,0.65);
  border-radius:var(--r);padding:15px 15px 20px;margin-bottom:12px;
  box-shadow:var(--shadow);backdrop-filter:blur(14px);
  animation:fadeUp 0.20s ease;}}
.tab-panel.active{{display:block;}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:none}}}}
.tab-title{{margin:0 0 12px;font-size:1.0rem;font-weight:700;color:var(--ink);}}

/* notes */
.notes{{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
  border-radius:11px;padding:11px 16px;margin-top:6px;}}
.notes h3{{margin:0 0 5px;font-size:0.86rem;color:#7aadd4;font-weight:700;}}
.notes li{{margin:4px 0;color:#6899cc;font-size:0.79rem;line-height:1.45;}}
</style>
<script>{plotly_js}</script>
</head>
<body>
<div class="wrap">
<div class="hero">
  <div class="badge">{run_id.upper()}</div>
  <h1>Cognitive Warfare Susceptibility Dashboard</h1>
  <p class="sub">
    Ontology-driven multi-agent simulation · Full 4×4 factorial design ·
    Profile-panel repeated-outcome SEM · Conditional susceptibility estimation ·
    Hierarchical feature importance decomposition
  </p>
</div>
<div class="cards">{cards_html}</div>
<div class="nav-outer" id="tabs">{''.join(nav_groups)}</div>
{panels}
<div class="notes">
  <h3>Methodological Notes</h3>
  <ul>{notes_html}</ul>
</div>
</div>
<script>
const btns=Array.from(document.querySelectorAll('.tab-btn'));
const pans=Array.from(document.querySelectorAll('.tab-panel'));
function activate(id){{btns.forEach(b=>b.classList.toggle('active',b.dataset.tab===id));pans.forEach(p=>p.classList.toggle('active',p.id===id));}}
btns.forEach(b=>b.addEventListener('click',()=>activate(b.dataset.tab)));
if(btns.length)activate(btns[0].dataset.tab);
</script>
</body>
</html>""".strip()


# ─── main entry point ─────────────────────────────────────────────────────────

def generate_research_visuals(
    sem_long_csv_path: str | Path,
    sem_result_json_path: str | Path,
    ols_params_csv_path: str | Path,
    output_dir: str | Path,
    run_id: str,
) -> Dict[str, Any]:
    output_root  = Path(output_dir)
    figures_dir  = output_root / "figures"
    snap_dir     = output_root / "data_snapshots"
    figures_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    long_df    = pd.read_csv(sem_long_csv_path)
    sem_result = json.loads(Path(sem_result_json_path).read_text(encoding="utf-8"))
    ols_params = pd.read_csv(ols_params_csv_path)

    s05 = Path(sem_long_csv_path).resolve().parent
    s06 = Path(sem_result_json_path).resolve().parent

    def _load(p: Path) -> pd.DataFrame:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    profile_df       = _load(s05 / "profile_level_effectivity.csv")
    profile_index_df = _load(s06 / "profile_susceptibility_index.csv")
    exploratory_df   = _load(s06 / "exploratory_moderator_comparison.csv")
    weight_df        = _load(s06 / "moderator_weight_table.csv")
    task_coeff_df    = _load(s06 / "conditional_susceptibility_task_coefficients.csv")
    task_summary_df  = _load(s06 / "conditional_susceptibility_task_summary.csv")

    sem_coeff_df = pd.DataFrame(sem_result.get("coefficients", []))
    fit          = sem_result.get("fit_indices", {})
    icc_data     = {}
    icc_path     = s06 / "intraclass_correlation.json"
    if icc_path.exists():
        try:
            icc_data = json.loads(icc_path.read_text())
        except Exception:
            pass

    icc_str = "n/a"
    for key in ("icc1_abs_delta", "icc_abs_delta", "icc1", "ICC1"):
        if key in icc_data:
            try:
                icc_str = f"{float(icc_data[key]):.3f}"
            except Exception:
                pass
            break

    n_profiles = int(long_df["profile_id"].nunique()) if "profile_id" in long_df.columns else "n/a"
    pct_pos    = (
        f"{(long_df['adversarial_effectivity'] > 0).mean() * 100:.1f}%"
        if "adversarial_effectivity" in long_df.columns else "n/a"
    )
    summary_cards: Dict[str, Any] = {
        "Profiles":        n_profiles,
        "Scenarios":       len(long_df),
        "Attack Vectors":  int(long_df["attack_leaf"].nunique()) if "attack_leaf" in long_df.columns else "n/a",
        "Opinion Leaves":  int(long_df["opinion_leaf"].nunique()) if "opinion_leaf" in long_df.columns else "n/a",
        "Mean |Δ|":        f"{long_df['abs_delta_score'].mean():.1f}" if "abs_delta_score" in long_df.columns else "n/a",
        "Mean AE":         f"{long_df['adversarial_effectivity'].mean():.1f}" if "adversarial_effectivity" in long_df.columns else "n/a",
        "% AE > 0":        pct_pos,
        "Mean Realism":    f"{long_df['attack_realism_score'].dropna().mean():.2f}" if "attack_realism_score" in long_df.columns else "n/a",
        "ICC(1) |Δ|":      icc_str,
        "CFI":             f"{fit['CFI']:.3f}" if fit.get("CFI") is not None else "n/a",
        "RMSEA":           f"{fit['RMSEA']:.3f}" if fit.get("RMSEA") is not None else "n/a",
    }

    figure_divs: List[Tuple[str, str]] = []
    visual_files: List[str] = []

    def _add_fig(title: str, fig: go.Figure, fname: str) -> None:
        visual_files.append(_save_figure_html(fig, figures_dir / fname))
        figure_divs.append((title, fig.to_html(include_plotlyjs=False, full_html=False)))

    def _add_html(title: str, html: str) -> None:
        figure_divs.append((title, html))

    _add_fig("Factorial 3D Surface",    _fig_factorial_3d(long_df),          "factorial_3d.html")
    _add_fig("Factorial Heat + Contour", _fig_factorial_2d(long_df),          "factorial_2d.html")

    if not sem_coeff_df.empty:
        _add_fig("SEM Network",  _fig_sem_network(sem_coeff_df),               "sem_network.html")
        _add_fig("SEM Heatmap",  _fig_sem_heatmap(sem_coeff_df, exploratory_df),"sem_heatmap.html")

    if not task_coeff_df.empty:
        _add_html("Conditional Susceptibility Estimator",
                  _html_cs_estimator(task_coeff_df, task_summary_df, long_df))
        _add_html("Perturbation Explorer",
                  _html_perturbation_explorer(task_coeff_df, long_df))

    _add_fig("Violin Distributions", _fig_violin(long_df),               "violin.html")

    if not profile_index_df.empty:
        _add_fig("Susceptibility Map", _fig_susceptibility_scatter(profile_index_df, long_df),
                 "susceptibility_map.html")

    if not exploratory_df.empty:
        _add_fig("Moderator Forest",       _fig_moderator_forest(exploratory_df),           "moderator_forest.html")
    if not weight_df.empty:
        _add_fig("Hierarchical Importance", _fig_hierarchical_importance(weight_df),        "hierarchical_importance.html")

    _add_fig("Profile Heatmap",  _fig_profile_heatmap(long_df, profile_index_df), "profile_heatmap.html")
    _add_fig("Baseline vs Post", _fig_baseline_post(long_df),                     "baseline_post.html")

    # snapshots
    long_df.to_csv(snap_dir / "sem_long_encoded_snapshot.csv", index=False)
    if not profile_df.empty:
        profile_df.to_csv(snap_dir / "profile_level_effectivity_snapshot.csv", index=False)
    if not profile_index_df.empty:
        profile_index_df.to_csv(snap_dir / "profile_susceptibility_snapshot.csv", index=False)
    if not exploratory_df.empty:
        exploratory_df.to_csv(snap_dir / "moderator_coefficients_snapshot.csv", index=False)

    notes = [
        "All profiles are attacked; the dashboard visualizes heterogeneity of manipulation outcomes, not a treatment vs control contrast.",
        "Adversarial effectivity (AE = Δ × d_k): <b>positive = manipulation succeeded</b>, negative = backfire or resistance.",
        "The 3D surface shows mean AE and inter-individual SD across the full 4×4 attack–opinion factorial.",
        "SEM Network: edge width ∝ |β| · blue = positive · red = negative · opacity = significance. Use filter buttons to focus.",
        "Conditional Susceptibility Estimator: configure any profile, select any task subset, get predicted AE grid and percentile rank.",
        "The susceptibility percentile is re-computed on the fly for the selected task subset vs the 100 original profiles.",
        "ICC(1) ≈ 0.052 for |Δ|: attack–opinion context explains ~95% of variance; stable profile traits explain ~5%.",
        "SEM fit (CFI=1.000, RMSEA=0.000) is expected in a near-saturated 4-indicator model at n=100.",
    ]

    dashboard_path = output_root / "interactive_sem_dashboard.html"
    dashboard_path.write_text(
        _render_dashboard_html(run_id, summary_cards, figure_divs, notes),
        encoding="utf-8",
    )
    visual_files.append(str(dashboard_path))

    return {
        "dashboard_path": str(dashboard_path),
        "visual_files":   visual_files,
        "summary_cards":  summary_cards,
    }
