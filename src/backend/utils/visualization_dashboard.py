"""
Interactive attack-effectivity dashboard — next-level visualization.

Tabs (generic for any run):
  🗂 Ontologies        → Ontology Explorer
  📡 Factorial Space   → Factorial 3D Surface, Factorial Heat + Contour
  🧠 SEM Analysis      → SEM Network (interactive), SEM Heatmap
  🔬 Estimation        → Conditional Susceptibility Estimator ★ (new), Perturbation Explorer
  👤 Profiles          → Susceptibility Map, Profile Heatmap
  📊 Moderators        → Moderator Forest, Hierarchical Importance
  📈 Raw Data          → Violin Distributions, Baseline vs Post
"""
from __future__ import annotations

import re
from collections import Counter
import json
from pathlib import Path
from textwrap import wrap
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


def _pretty_indicator(s: str) -> str:
    for prefix in ["adversarial_delta_indicator__", "abs_delta_indicator__"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.replace("_", " ").replace("  ", " ").strip().title()


def _wrap_label(s: str, width: int = 18) -> str:
    text = str(s).replace("<br>", " ").strip()
    lines = wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return "<br>".join(lines) if lines else text


def _clip_label(s: str, max_len: int = 42) -> str:
    text = re.sub(r"\s+", " ", str(s)).strip()
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _path_parts(s: str) -> List[str]:
    raw = str(s)
    if ">" in raw:
        return [_leaf(part) for part in raw.split(">") if str(part).strip()]
    return [_leaf(raw)]


def _path_context(s: str, keep: int = 2) -> str:
    parts = _path_parts(s)
    if len(parts) <= 1:
        return ""
    trimmed = [
        part for idx, part in enumerate(parts[:-1])
        if idx > 0 or part.lower() not in {"attack vectors", "issue position taxonomy", "profile"}
    ]
    return " / ".join(trimmed[-keep:])


def _unique_display_map(values: List[str]) -> Dict[str, str]:
    leaves = [_leaf(v) for v in values]
    counts = Counter(leaves)
    labels: Dict[str, str] = {}
    for value, leaf_name in zip(values, leaves):
        if counts[leaf_name] <= 1:
            labels[value] = leaf_name
            continue
        context = _path_context(value, keep=1)
        labels[value] = f"{context} • {leaf_name}" if context else leaf_name
    return labels


def _moderator_hierarchy(label: str, ontology_group: str | None = None) -> List[str]:
    group_parts = [part.strip() for part in str(ontology_group or "").split(":") if part.strip()]
    clean_label = re.sub(r"\s+", " ", str(label)).strip()
    leaf_label = clean_label
    if group_parts:
        last = group_parts[-1]
        leaf_label = re.sub(rf"(?i)\b{re.escape(last)}\b", "", leaf_label, count=1)
        leaf_label = re.sub(r"(?i)\bBig Five\b", "", leaf_label)
        leaf_label = re.sub(r"\s+", " ", leaf_label).strip(" -:%")
    if not leaf_label:
        leaf_label = clean_label

    segments = ["Profile Features"]
    segments.extend(group_parts if group_parts else ["Other Moderators"])
    if not segments or segments[-1] != leaf_label:
        segments.append(leaf_label)
    deduped: List[str] = []
    for seg in segments:
        if not deduped or deduped[-1] != seg:
            deduped.append(seg)
    return deduped


def _infer_sem_moderator_groups(label: str) -> Tuple[str, str]:
    txt = re.sub(r"\s+", " ", str(label)).strip()
    if txt.startswith("Big Five "):
        remainder = txt[len("Big Five "):].replace("%", "").strip()
        subgroup = remainder.split(" Mean")[0].strip()
        return "Profile Traits", subgroup or "Big Five"
    if txt.startswith("Sex "):
        return "Demographics", "Sex"
    if "Age" in txt:
        return "Demographics", "Age"
    if "Baseline" in txt:
        return "Model Controls", "Baseline"
    if "Exposure" in txt or "Realism" in txt or "Plausibility" in txt:
        return "Model Controls", "Exposure / Quality"
    return "Other Moderators", txt.split(" ")[0]


def _humanize_ontology_label(label: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(label).replace("_", " ").strip())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if text.lower() == text:
        specials = {"ai", "ui", "id", "uk", "us", "eu", "vat", "ngo", "api"}
        parts = [
            part.upper() if part.lower() in specials else part.capitalize()
            for part in text.split(" ")
        ]
        return " ".join(parts)
    return text


def _split_ontology_children(node: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(node, dict):
        return {}, {"value": node}
    structural: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    for key, value in node.items():
        if str(key).startswith("_"):
            metadata[str(key)] = value
        elif isinstance(value, dict):
            structural[str(key)] = value
        else:
            metadata[str(key)] = value
    return structural, metadata


def _build_ontology_payload(
    tree: Dict[str, Any],
    *,
    env: str,
    ontology_key: str,
    sampled_paths: set[str],
    sampled_leaf_names: set[str],
) -> Dict[str, Any]:
    root_label = {
        "ATTACK": "Attack Ontology",
        "OPINION": "Opinion Ontology",
        "PROFILE": "Profile Ontology",
    }.get(ontology_key, ontology_key.title())

    nodes: List[Dict[str, Any]] = []
    node_counter = 0

    def _walk(
        label: str,
        payload: Any,
        parent_id: Optional[str],
        raw_parts: Tuple[str, ...],
        depth: int,
    ) -> Tuple[str, int, int, int, int, int, int]:
        nonlocal node_counter
        node_id = f"{env}:{ontology_key}:{node_counter}"
        node_counter += 1

        structural, metadata = _split_ontology_children(payload)
        metadata_preview: List[List[str]] = []
        for key, value in list(metadata.items())[:6]:
            if isinstance(value, (dict, list)):
                value_txt = json.dumps(value, ensure_ascii=False)
            else:
                value_txt = str(value)
            metadata_preview.append([
                _humanize_ontology_label(key),
                _clip_label(value_txt.replace("\n", " "), 120),
            ])

        raw_path = " > ".join(raw_parts)
        human_path = " > ".join(_humanize_ontology_label(part) for part in raw_parts) if raw_parts else root_label
        display_label = _humanize_ontology_label(label)

        node: Dict[str, Any] = {
            "id": node_id,
            "parent": parent_id,
            "name": str(label),
            "label": display_label,
            "short": _clip_label(display_label, 30),
            "tiny": _clip_label(display_label, 18),
            "depth": depth,
            "path": raw_path,
            "path_label": human_path,
            "children": [],
            "metadata_preview": metadata_preview,
            "metadata_count": len(metadata),
            "kind": "branch",
            "sample_exact": False,
            "sample_aligned": False,
        }
        nodes.append(node)

        subtree_nodes = 1
        subtree_leaves = 0
        subtree_metadata_leaves = 0
        subtree_exact = 0
        subtree_aligned = 0
        max_depth = depth

        if not structural:
            node["kind"] = "leaf_meta" if metadata else "leaf"
            subtree_leaves = 1
            if metadata:
                subtree_metadata_leaves = 1
            leaf_token = raw_parts[-1] if raw_parts else str(label)
            node["sample_exact"] = bool(raw_path and raw_path in sampled_paths)
            node["sample_aligned"] = bool(
                raw_path and not node["sample_exact"] and leaf_token in sampled_leaf_names
            )
            subtree_exact = 1 if node["sample_exact"] else 0
            subtree_aligned = 1 if node["sample_aligned"] else 0
        else:
            for child_name, child_payload in structural.items():
                child_id, n_nodes, n_leaves, n_meta_leaves, child_max_depth, n_exact, n_aligned = _walk(
                    child_name,
                    child_payload,
                    node_id,
                    raw_parts + (child_name,),
                    depth + 1,
                )
                node["children"].append(child_id)
                subtree_nodes += n_nodes
                subtree_leaves += n_leaves
                subtree_metadata_leaves += n_meta_leaves
                subtree_exact += n_exact
                subtree_aligned += n_aligned
                max_depth = max(max_depth, child_max_depth)

        node["child_count"] = len(node["children"])
        node["leaf_count"] = subtree_leaves
        node["subtree_node_count"] = subtree_nodes
        node["metadata_leaf_count"] = subtree_metadata_leaves
        node["max_subtree_depth"] = max_depth
        node["sample_exact_subtree"] = subtree_exact
        node["sample_aligned_subtree"] = subtree_aligned

        return (
            node_id,
            subtree_nodes,
            subtree_leaves,
            subtree_metadata_leaves,
            max_depth,
            subtree_exact,
            subtree_aligned,
        )

    root_id, node_count, leaf_count, metadata_leaf_count, max_depth, exact_count, aligned_count = _walk(
        root_label,
        tree,
        None,
        (),
        0,
    )

    branch_count = sum(1 for node in nodes if node["kind"] == "branch")
    recommended_depth = 2 if leaf_count > 1500 else 3 if leaf_count > 200 else min(4, max_depth)

    return {
        "root_id": root_id,
        "nodes": nodes,
        "summary": {
            "node_count": int(node_count),
            "leaf_count": int(leaf_count),
            "branch_count": int(branch_count),
            "metadata_leaf_count": int(metadata_leaf_count),
            "max_depth": int(max_depth),
            "recommended_depth": int(max(recommended_depth, 1)),
            "sample_exact_count": int(exact_count),
            "sample_aligned_count": int(aligned_count),
        },
    }


def _load_dashboard_ontology_payload(ontology_catalog: Dict[str, Any]) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    ontology_root = str(ontology_catalog.get("ontology_root", ""))
    run_source = "test" if "separate/test" in ontology_root else "production"

    selected_attack_paths = {str(v) for v in ontology_catalog.get("selected_attack_leaves", [])}
    if not selected_attack_paths and ontology_catalog.get("selected_attack_leaf"):
        selected_attack_paths.add(str(ontology_catalog["selected_attack_leaf"]))
    selected_opinion_paths = {str(v) for v in ontology_catalog.get("selected_opinion_leaves", [])}

    sampled_paths_by_key: Dict[str, set[str]] = {
        "ATTACK": selected_attack_paths,
        "OPINION": selected_opinion_paths,
        "PROFILE": set(),
    }
    sampled_leaf_names_by_key: Dict[str, set[str]] = {
        key: {path.split(">")[-1].strip() for path in paths if str(path).strip()}
        for key, paths in sampled_paths_by_key.items()
    }

    sources: Dict[str, Dict[str, Any]] = {}
    for env in ("production", "test"):
        env_root = project_root / "src" / "backend" / "ontology" / "separate" / env
        env_sources: Dict[str, Any] = {}
        for ontology_key, rel in {
            "ATTACK": env_root / "ATTACK" / "attack.json",
            "OPINION": env_root / "OPINION" / "opinion.json",
            "PROFILE": env_root / "PROFILE" / "profile.json",
        }.items():
            raw_tree = json.loads(rel.read_text(encoding="utf-8"))
            env_sources[ontology_key] = _build_ontology_payload(
                raw_tree,
                env=env,
                ontology_key=ontology_key,
                sampled_paths=sampled_paths_by_key.get(ontology_key, set()),
                sampled_leaf_names=sampled_leaf_names_by_key.get(ontology_key, set()),
            )
        sources[env] = env_sources

    return {
        "current_run_source": run_source,
        "selected_paths": {
            "ATTACK": sorted(selected_attack_paths),
            "OPINION": sorted(selected_opinion_paths),
            "PROFILE": [],
        },
        "sources": sources,
    }


def _build_tree_nodes(paths_with_values: List[Tuple[List[str], float]]) -> Tuple[List[str], List[str], List[str], List[float], List[str]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    for path, value in paths_with_values:
        for idx, segment in enumerate(path):
            node_id = " | ".join(path[: idx + 1])
            parent_id = " | ".join(path[:idx]) if idx else ""
            if node_id not in nodes:
                nodes[node_id] = {
                    "label": segment,
                    "parent": parent_id,
                    "value": 0.0,
                    "path": " → ".join(path[: idx + 1]),
                }
            nodes[node_id]["value"] += float(value)
    ids = list(nodes.keys())
    labels = [nodes[node_id]["label"] for node_id in ids]
    parents = [nodes[node_id]["parent"] for node_id in ids]
    values = [nodes[node_id]["value"] for node_id in ids]
    paths = [nodes[node_id]["path"] for node_id in ids]
    return ids, labels, parents, values, paths


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


def _save_html_block(content: str, path: Path, title: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plotly_js = get_plotlyjs()
    path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{title}</title>
<style>
body{{margin:0;padding:16px;background:#ffffff;font-family:"IBM Plex Sans","Avenir Next","Segoe UI",sans-serif;color:#14213d;}}
</style>
<script>{plotly_js}</script>
</head>
<body>
{content}
</body>
</html>""",
        encoding="utf-8",
    )
    return str(path)


# ─── figure builders ──────────────────────────────────────────────────────────

def _html_ontology_explorer(ontology_payload: Dict[str, Any]) -> str:
    payload_json = json.dumps(ontology_payload, ensure_ascii=False)
    return f"""
<div id="ontx-root">
  <style>
    #ontx-root .ontx-shell{{display:grid;grid-template-columns:minmax(300px,340px) minmax(0,1fr);gap:16px;align-items:start}}
    #ontx-root .ontx-card,#ontx-root .ontx-panel,#ontx-root .ontx-canvas-card{{background:#f7faff;border:1px solid #dbe3ef;border-radius:14px;box-shadow:0 3px 14px rgba(20,33,61,0.05)}}
    #ontx-root .ontx-card{{padding:12px 13px}}
    #ontx-root .ontx-card + .ontx-card{{margin-top:10px}}
    #ontx-root .ontx-title{{font-weight:800;font-size:0.92rem;color:{PALETTE['navy']};margin-bottom:8px}}
    #ontx-root .ontx-sub{{font-size:0.76rem;line-height:1.45;color:{PALETTE['muted']};margin-bottom:8px}}
    #ontx-root .ontx-segment{{display:flex;flex-wrap:wrap;gap:6px}}
    #ontx-root .ontx-btn{{padding:6px 10px;border-radius:999px;border:1px solid #c8d7ec;background:#fff;color:{PALETTE['ink']};cursor:pointer;font-size:0.75rem;font-weight:700}}
    #ontx-root .ontx-btn.active{{background:{PALETTE['blue']};border-color:{PALETTE['blue']};color:#fff}}
    #ontx-root .ontx-focus{{display:flex;flex-direction:column;gap:8px}}
    #ontx-root .ontx-focus-item{{padding:10px 11px;border-radius:12px;border:1px solid #dbe3ef;background:#fff;cursor:pointer;transition:transform 120ms ease,border-color 120ms ease,box-shadow 120ms ease}}
    #ontx-root .ontx-focus-item:hover{{transform:translateY(-1px);box-shadow:0 6px 14px rgba(20,33,61,0.08)}}
    #ontx-root .ontx-focus-item.active{{border-color:{PALETTE['blue']};box-shadow:0 0 0 2px rgba(29,78,137,0.08)}}
    #ontx-root .ontx-focus-top{{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:5px}}
    #ontx-root .ontx-focus-top strong{{font-size:0.82rem;color:{PALETTE['ink']}}}
    #ontx-root .ontx-focus-pill{{display:inline-flex;align-items:center;padding:2px 7px;border-radius:999px;font-size:0.66rem;font-weight:800;letter-spacing:0.02em}}
    #ontx-root .ontx-focus-meta{{display:flex;gap:10px;flex-wrap:wrap;font-size:0.72rem;color:{PALETTE['muted']}}}
    #ontx-root .ontx-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
    #ontx-root .ontx-select{{width:100%;padding:7px 8px;border-radius:9px;border:1px solid #dbe3ef;background:#fff;color:{PALETTE['ink']};font-size:0.80rem}}
    #ontx-root .ontx-slider-wrap{{background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:9px 10px}}
    #ontx-root .ontx-slider-meta{{display:flex;justify-content:space-between;gap:10px;align-items:center;font-size:0.76rem;color:{PALETTE['muted']};font-weight:700;margin-bottom:6px}}
    #ontx-root input[type="range"]{{width:100%;accent-color:{PALETTE['blue']}}}
    #ontx-root .ontx-toggle{{display:flex;align-items:center;gap:7px;font-size:0.77rem;color:{PALETTE['ink']};font-weight:600}}
    #ontx-root .ontx-toggle-list{{display:flex;flex-direction:column;gap:8px}}
    #ontx-root .ontx-search{{width:100%;padding:9px 10px;border-radius:10px;border:1px solid #dbe3ef;background:#fff;font-size:0.82rem;color:{PALETTE['ink']}}}
    #ontx-root .ontx-results{{display:flex;flex-direction:column;gap:7px;max-height:220px;overflow:auto;margin-top:10px;padding-right:4px}}
    #ontx-root .ontx-result{{padding:8px 9px;border-radius:10px;background:#fff;border:1px solid #dbe3ef;cursor:pointer}}
    #ontx-root .ontx-result strong{{display:block;font-size:0.77rem;color:{PALETTE['ink']}}}
    #ontx-root .ontx-result span{{display:block;font-size:0.71rem;color:{PALETTE['muted']};line-height:1.4;margin-top:3px}}
    #ontx-root .ontx-stage{{display:flex;flex-direction:column;gap:12px}}
    #ontx-root .ontx-banner{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;background:linear-gradient(135deg,#f8fbff 0%,#eef5ff 100%);border:1px solid #dbe3ef;border-radius:12px;padding:11px 13px}}
    #ontx-root .ontx-status{{font-size:0.79rem;line-height:1.5;color:{PALETTE['muted']}}}
    #ontx-root .ontx-status strong{{color:{PALETTE['ink']}}}
    #ontx-root .ontx-legend{{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}}
    #ontx-root .ontx-legend-item{{display:flex;align-items:center;gap:6px;font-size:0.72rem;color:{PALETTE['muted']}}}
    #ontx-root .ontx-swatch{{display:inline-block;width:28px;height:10px;border-radius:999px}}
    #ontx-root .ontx-swatch.attack{{background:linear-gradient(90deg,rgba(231,111,81,0.22),rgba(231,111,81,0.92))}}
    #ontx-root .ontx-swatch.opinion{{background:linear-gradient(90deg,rgba(42,157,143,0.22),rgba(42,157,143,0.92))}}
    #ontx-root .ontx-swatch.profile{{background:linear-gradient(90deg,rgba(29,78,137,0.22),rgba(29,78,137,0.92))}}
    #ontx-root .ontx-compare{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}
    #ontx-root .ontx-compare-card{{padding:11px 12px;border-radius:12px;border:1px solid #dbe3ef;background:#fff}}
    #ontx-root .ontx-compare-card.active{{border-color:{PALETTE['blue']};box-shadow:0 0 0 2px rgba(29,78,137,0.08)}}
    #ontx-root .ontx-compare-top{{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:7px}}
    #ontx-root .ontx-compare-top strong{{font-size:0.82rem;color:{PALETTE['ink']}}}
    #ontx-root .ontx-chip{{display:inline-flex;align-items:center;padding:2px 7px;border-radius:999px;font-size:0.66rem;font-weight:800;letter-spacing:0.02em}}
    #ontx-root .ontx-compare-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}}
    #ontx-root .ontx-metric{{padding:7px 8px;border-radius:10px;background:#f8fbff;border:1px solid #e2eaf5}}
    #ontx-root .ontx-metric .k{{font-size:0.67rem;text-transform:uppercase;letter-spacing:0.06em;color:{PALETTE['muted']}}}
    #ontx-root .ontx-metric .v{{font-size:0.90rem;font-weight:800;color:{PALETTE['ink']};margin-top:2px}}
    #ontx-root .ontx-canvas-card{{padding:0;overflow:hidden}}
    #ontx-root .ontx-canvas-head{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;padding:12px 14px;border-bottom:1px solid #dbe3ef;background:linear-gradient(180deg,#ffffff 0%,#fbfdff 100%)}}
    #ontx-root .ontx-canvas-head strong{{display:block;font-size:0.94rem;color:{PALETTE['navy']}}}
    #ontx-root .ontx-canvas-head span{{display:block;font-size:0.74rem;color:{PALETTE['muted']};margin-top:3px;line-height:1.45}}
    #ontx-root .ontx-chipline{{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}}
    #ontx-root .ontx-chipline .ontx-chip{{background:rgba(29,78,137,0.08);color:{PALETTE['blue']}}}
    #ontx-root #ontx-canvas-wrap{{background:
      radial-gradient(circle at 12% 16%, rgba(42,157,143,0.08), transparent 28%),
      radial-gradient(circle at 88% 14%, rgba(231,111,81,0.08), transparent 26%),
      linear-gradient(180deg,#ffffff 0%,#f9fbff 100%);
      min-height:760px;overflow:auto;padding:16px;
      cursor:grab;user-select:none;-webkit-user-select:none}}
    #ontx-root #ontx-svg{{display:block}}
    #ontx-root .ontx-bottom{{display:grid;grid-template-columns:1.15fr 1fr 0.95fr;gap:12px}}
    #ontx-root .ontx-panel{{padding:12px 13px}}
    #ontx-root .ontx-panel h4{{margin:0 0 8px;font-size:0.82rem;color:{PALETTE['navy']}}}
    #ontx-root .ontx-kv{{display:grid;grid-template-columns:1fr 1fr;gap:7px}}
    #ontx-root .ontx-kv .ontx-metric{{padding:8px 9px}}
    #ontx-root .ontx-meta-list{{display:flex;flex-direction:column;gap:6px;margin-top:10px}}
    #ontx-root .ontx-meta-item{{padding:7px 8px;border-radius:10px;background:#fff;border:1px solid #dbe3ef}}
    #ontx-root .ontx-meta-item strong{{display:block;font-size:0.75rem;color:{PALETTE['ink']};margin-bottom:2px}}
    #ontx-root .ontx-meta-item span{{font-size:0.72rem;color:{PALETTE['muted']};line-height:1.4;word-break:break-word}}
    #ontx-root .ontx-path{{padding:8px 9px;border-radius:10px;background:#fff;border:1px solid #dbe3ef;font-size:0.73rem;line-height:1.45;color:{PALETTE['muted']};word-break:break-word}}
    #ontx-root .ontx-highlight-list{{display:flex;flex-direction:column;gap:7px}}
    #ontx-root .ontx-highlight-item{{padding:8px 9px;border-radius:10px;background:#fff;border:1px solid #dbe3ef}}
    #ontx-root .ontx-highlight-item strong{{display:block;font-size:0.76rem;color:{PALETTE['ink']}}}
    #ontx-root .ontx-highlight-item span{{display:block;font-size:0.71rem;color:{PALETTE['muted']};line-height:1.4;margin-top:3px}}
    #ontx-root .ontx-note{{font-size:0.73rem;line-height:1.45;color:{PALETTE['muted']}}}
    #ontx-root .ontx-legend-stack{{display:flex;flex-direction:column;gap:8px}}
    #ontx-root .ontx-legend-row{{display:flex;align-items:center;gap:9px;font-size:0.74rem;color:{PALETTE['muted']}}}
    #ontx-root .ontx-node-demo{{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border-radius:999px;flex:0 0 auto}}
    #ontx-root .ontx-actions{{display:flex;flex-wrap:wrap;gap:6px}}
    @media (max-width: 1180px) {{
      #ontx-root .ontx-shell{{grid-template-columns:1fr}}
      #ontx-root .ontx-bottom{{grid-template-columns:1fr}}
      #ontx-root .ontx-compare{{grid-template-columns:1fr}}
    }}
  </style>

  <div class="ontx-shell">
    <div>
      <div class="ontx-card">
        <div class="ontx-title">Ontology Source</div>
        <div class="ontx-sub">Production is the default explorer surface. This run used the test ontology, so the source toggle lets you compare the research-scale production hierarchy against the exact pilot ontology used in run 8.</div>
        <div class="ontx-segment" id="ontx-source">
          <button class="ontx-btn active" data-source="production">Production</button>
          <button class="ontx-btn" data-source="test">Test</button>
        </div>
      </div>

      <div class="ontx-card">
        <div class="ontx-title">Ontology Focus</div>
        <div class="ontx-sub">Choose which hierarchical state space to inspect: cybermanipulation attacks, opinion targets, or profile space.</div>
        <div id="ontx-focus" class="ontx-focus"></div>
      </div>

      <div class="ontx-card">
        <div class="ontx-title">Layout</div>
        <div class="ontx-sub">Switch between directional tree flow, top-down structure, and radial orbit when the hierarchy gets dense.</div>
        <div class="ontx-segment" id="ontx-layout">
          <button class="ontx-btn active" data-layout="flow">Left → Right</button>
          <button class="ontx-btn" data-layout="vertical">Top ↓ Bottom</button>
          <button class="ontx-btn" data-layout="radial">Radial</button>
        </div>
      </div>

      <div class="ontx-card">
        <div class="ontx-title">Depth & Labels</div>
        <div class="ontx-slider-wrap">
          <div class="ontx-slider-meta"><span>Visible depth</span><span id="ontx-depth-display">3</span></div>
          <input type="range" id="ontx-depth" min="1" max="8" step="1" value="3">
        </div>
        <div class="ontx-sub" style="margin:10px 0 6px">Label density</div>
        <select id="ontx-label-mode" class="ontx-select">
          <option value="compact" selected>Compact</option>
          <option value="branches">Branches only</option>
          <option value="full">Full labels</option>
        </select>
      </div>

      <div class="ontx-card">
        <div class="ontx-title">Visibility</div>
        <div class="ontx-toggle-list">
          <label class="ontx-toggle"><input type="checkbox" id="ontx-show-meta" checked> Show metadata halos for annotated leaves</label>
          <label class="ontx-toggle"><input type="checkbox" id="ontx-highlight-run" checked> Highlight run-aligned leaves and branch matches</label>
          <label class="ontx-toggle"><input type="checkbox" id="ontx-relevant-only"> Restrict view to run-aligned branches</label>
        </div>
      </div>

      <div class="ontx-card">
        <div class="ontx-title">Search</div>
        <input id="ontx-search" class="ontx-search" type="text" placeholder="Search branches, leaves, or paths">
        <div id="ontx-results" class="ontx-results"></div>
      </div>

      <div class="ontx-card">
        <div class="ontx-title">Actions</div>
        <div class="ontx-actions">
          <button class="ontx-btn" id="ontx-expand-all">Expand all</button>
          <button class="ontx-btn" id="ontx-collapse-all">Collapse all</button>
          <button class="ontx-btn" id="ontx-reset-depth">Reset to depth</button>
          <button class="ontx-btn" id="ontx-zoom-out">Zoom −</button>
          <button class="ontx-btn" id="ontx-zoom-in">Zoom +</button>
          <button class="ontx-btn" id="ontx-zoom-reset">100%</button>
        </div>
      </div>
    </div>

    <div class="ontx-stage">
      <div class="ontx-banner">
        <div class="ontx-status" id="ontx-status"></div>
        <div class="ontx-legend">
          <div class="ontx-legend-item"><span class="ontx-swatch attack"></span><span>attack ontology accent</span></div>
          <div class="ontx-legend-item"><span class="ontx-swatch opinion"></span><span>opinion ontology accent</span></div>
          <div class="ontx-legend-item"><span class="ontx-swatch profile"></span><span>profile ontology accent</span></div>
        </div>
      </div>

      <div id="ontx-compare" class="ontx-compare"></div>

      <div class="ontx-canvas-card">
        <div class="ontx-canvas-head">
          <div>
            <strong id="ontx-active-title"></strong>
            <span id="ontx-active-sub"></span>
          </div>
          <div class="ontx-chipline" id="ontx-chipline"></div>
        </div>
        <div id="ontx-canvas-wrap"></div>
      </div>

      <div class="ontx-bottom">
        <div class="ontx-panel">
          <h4>Selected Node</h4>
          <div id="ontx-inspector"></div>
        </div>
        <div class="ontx-panel">
          <h4>Path Highlights</h4>
          <div id="ontx-highlights" class="ontx-highlight-list"></div>
        </div>
        <div class="ontx-panel">
          <h4>Legend & Use</h4>
          <div class="ontx-legend-stack">
            <div class="ontx-legend-row"><span class="ontx-node-demo" style="background:{PALETTE['panel']};border:2px solid {PALETTE['ink']}"></span><span>Branch node. Click to expand or collapse the subtree.</span></div>
            <div class="ontx-legend-row"><span class="ontx-node-demo" style="background:{PALETTE['panel']};border:2px dashed {PALETTE['amber']}"></span><span>Dashed halo marks annotated test leaves with metadata such as descriptions or adversarial direction.</span></div>
            <div class="ontx-legend-row"><span class="ontx-node-demo" style="background:{PALETTE['panel']};border:2px solid {PALETTE['gold']};box-shadow:0 0 0 3px rgba(240,192,64,0.16)"></span><span>Gold ring marks exact run-aligned leaves in the ontology source used by the run.</span></div>
            <div class="ontx-legend-row"><span class="ontx-node-demo" style="background:{PALETTE['panel']};border:2px solid {PALETTE['amber']}"></span><span>Amber ring marks leaf-name alignment across sources, useful when the production ontology extends the test ontology.</span></div>
            <div class="ontx-note">Use lower visible depth for the production ATTACK and PROFILE ontologies, then search or expand selected branches to inspect local structure without overwhelming the canvas.</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
  (function(){{
    const DATA = {payload_json};
    const root = document.getElementById('ontx-root');
    const COLORS = {{
      ATTACK:  {{ hue: 18,  accent: '{PALETTE['orange']}', soft: 'rgba(231,111,81,0.14)', edge: 'rgba(231,111,81,0.24)' }},
      OPINION: {{ hue: 168, accent: '{PALETTE['teal']}',   soft: 'rgba(42,157,143,0.14)', edge: 'rgba(42,157,143,0.24)' }},
      PROFILE: {{ hue: 214, accent: '{PALETTE['blue']}',   soft: 'rgba(29,78,137,0.14)', edge: 'rgba(29,78,137,0.22)' }},
    }};
    const expanded = {{}};
    const depthStore = {{}};
    const state = {{
      source: 'production',
      ontology: 'ATTACK',
      layout: 'flow',
      maxDepth: 3,
      labelMode: 'compact',
      showMetadata: true,
      highlightRun: true,
      relevantOnly: false,
      zoom: 1,
      search: '',
      selectedId: null,
    }};

    function escapeHtml(txt) {{
      return String(txt ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}
    function datasetKey() {{
      return `${{state.source}}:${{state.ontology}}`;
    }}
    function dataset() {{
      return DATA.sources[state.source][state.ontology];
    }}
    function activeSpec() {{
      return COLORS[state.ontology];
    }}
    function initDatasets() {{
      Object.entries(DATA.sources).forEach(([env, envObj]) => {{
        Object.entries(envObj).forEach(([ontology, ds]) => {{
          ds.nodeMap = Object.fromEntries(ds.nodes.map(n => [n.id, n]));
          const key = `${{env}}:${{ontology}}`;
          depthStore[key] = ds.summary.recommended_depth || 3;
          /* Always open layers 1 & 2: expand all branch nodes at depth 0 and 1 */
          expanded[key] = new Set(
            ds.nodes
              .filter(n => n.kind === 'branch' && n.depth <= 1)
              .map(n => n.id)
          );
        }});
      }});
      state.maxDepth = depthStore[datasetKey()] || 3;
      state.selectedId = dataset().root_id;
    }}
    function badgeHtml(text, bg, fg) {{
      return `<span class="ontx-chip" style="background:${{bg}};color:${{fg}}">${{escapeHtml(text)}}</span>`;
    }}
    function setSource(source) {{
      state.source = source;
      state.maxDepth = depthStore[datasetKey()] || dataset().summary.recommended_depth || 3;
      root.querySelector('#ontx-depth').value = state.maxDepth;
      root.querySelector('#ontx-depth-display').textContent = state.maxDepth;
      state.selectedId = dataset().root_id;
      syncButtons();
      renderAll();
    }}
    function setOntology(ontology) {{
      state.ontology = ontology;
      state.maxDepth = depthStore[datasetKey()] || dataset().summary.recommended_depth || 3;
      root.querySelector('#ontx-depth').value = state.maxDepth;
      root.querySelector('#ontx-depth-display').textContent = state.maxDepth;
      state.selectedId = dataset().root_id;
      renderAll();
    }}
    function setLayout(layout) {{
      state.layout = layout;
      syncButtons();
      renderGraph();
    }}
    function syncButtons() {{
      root.querySelectorAll('#ontx-source .ontx-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.source === state.source));
      root.querySelectorAll('#ontx-layout .ontx-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.layout === state.layout));
    }}
    function relevantAvailable(ds) {{
      return (ds.summary.sample_exact_count || 0) + (ds.summary.sample_aligned_count || 0);
    }}
    function currentExpanded() {{
      return expanded[datasetKey()];
    }}
    function collapseAll() {{
      expanded[datasetKey()] = new Set();
      renderGraph();
      renderInspector();
    }}
    function expandAll() {{
      const ds = dataset();
      expanded[datasetKey()] = new Set(ds.nodes.filter(n => n.kind === 'branch').map(n => n.id));
      renderGraph();
      renderInspector();
    }}
    function resetToDepth() {{
      const ds = dataset();
      expanded[datasetKey()] = new Set(ds.nodes.filter(n => n.kind === 'branch' && n.depth < state.maxDepth).map(n => n.id));
      renderGraph();
      renderInspector();
    }}
    function expandAncestors(id) {{
      const ds = dataset();
      const nodeMap = ds.nodeMap;
      let cur = nodeMap[id];
      while (cur && cur.parent) {{
        currentExpanded().add(cur.parent);
        cur = nodeMap[cur.parent];
      }}
    }}
    /* BM25-inspired lexical search */
    function bm25Score(tokens, node) {{
      if (!tokens.length) return 0;
      const k1 = 1.5, b = 0.75, AVG_LEN = 14;
      const docText = `${{node.label}} ${{node.path_label||''}} ${{node.name||''}}`.toLowerCase();
      const docTokens = docText.split(/[^a-z0-9]+/).filter(Boolean);
      const docLen = Math.max(docTokens.length, 1);
      let score = 0;
      for (const term of tokens) {{
        const tf = docTokens.reduce((a, t) => a + (t.startsWith(term) ? 1 : 0), 0);
        if (!tf) continue;
        const labelExact = node.label.toLowerCase().includes(term);
        const labelBoost = labelExact ? 2.5 : 1.0;
        const tf_n = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * docLen / AVG_LEN));
        score += tf_n * labelBoost;
      }}
      return score;
    }}
    function searchMatches(ds) {{
      const q = state.search.trim().toLowerCase();
      if (!q) return [];
      const tokens = q.split(/ +/).filter(t => t.length >= 2);
      if (!tokens.length) {{
        return ds.nodes.filter(n => n.label.toLowerCase().startsWith(q)).slice(0, 18);
      }}
      return ds.nodes
        .map(n => ({{ node: n, score: bm25Score(tokens, n) }}))
        .filter(r => r.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 18)
        .map(r => r.node);
    }}
    function allowRelevant(node) {{
      if (!state.relevantOnly) return true;
      return (node.sample_exact_subtree || 0) > 0 || (node.sample_aligned_subtree || 0) > 0;
    }}
    function collectVisible(ds) {{
      const nodeMap = ds.nodeMap;
      const visibleNodes = [];
      const visibleEdges = [];
      const visibleChildren = {{}};
      function walk(id) {{
        const node = nodeMap[id];
        if (!node) return;
        if (id !== ds.root_id && !allowRelevant(node)) return;
        visibleNodes.push(node);
        const childIds = node.children
          .filter(childId => nodeMap[childId] && allowRelevant(nodeMap[childId]));
        const open = id === ds.root_id || currentExpanded().has(id);
        const descend = open && node.depth < state.maxDepth;
        visibleChildren[id] = descend ? childIds : [];
        if (descend) {{
          childIds.forEach(childId => {{
            visibleEdges.push([id, childId]);
            walk(childId);
          }});
        }}
      }}
      walk(ds.root_id);
      return {{ visibleNodes, visibleEdges, visibleChildren }};
    }}
    function layoutTree(ds, visible) {{
      const order = {{}};
      const children = visible.visibleChildren;
      let leafIndex = 0;
      function place(id) {{
        const kids = children[id] || [];
        if (!kids.length) {{
          order[id] = leafIndex++;
          return order[id];
        }}
        const vals = kids.map(place);
        order[id] = vals.reduce((a, b) => a + b, 0) / Math.max(vals.length, 1);
        return order[id];
      }}
      place(ds.root_id);
      const maxDepth = Math.max(...visible.visibleNodes.map(n => n.depth), 0);
      if (state.layout === 'flow') {{
        const width = Math.max(860, 160 + maxDepth * 220 + 340);
        const height = Math.max(440, leafIndex * 54 + 120);
        const pos = {{}};
        visible.visibleNodes.forEach(node => {{
          pos[node.id] = {{
            x: 84 + node.depth * 220,
            y: 60 + (order[node.id] || 0) * 54,
          }};
        }});
        return {{ pos, width, height, centerX: width / 2, centerY: height / 2 }};
      }}
      if (state.layout === 'vertical') {{
        const width = Math.max(760, leafIndex * 60 + 160);
        const height = Math.max(520, 120 + maxDepth * 180 + 160);
        const pos = {{}};
        visible.visibleNodes.forEach(node => {{
          pos[node.id] = {{
            x: 80 + (order[node.id] || 0) * 60,
            y: 72 + node.depth * 180,
          }};
        }});
        return {{ pos, width, height, centerX: width / 2, centerY: height / 2 }};
      }}

      const maxVisibleDepth = Math.max(...visible.visibleNodes.map(n => n.depth), 1);
      const angleById = {{}};
      leafIndex = 0;
      function placeRadial(id) {{
        const kids = children[id] || [];
        if (!kids.length) {{
          const angle = ((leafIndex++) / Math.max(visible.visibleNodes.filter(n => !(children[n.id] || []).length).length, 1)) * Math.PI * 2;
          angleById[id] = angle;
          return angle;
        }}
        const vals = kids.map(placeRadial);
        angleById[id] = vals.reduce((a, b) => a + b, 0) / Math.max(vals.length, 1);
        return angleById[id];
      }}
      placeRadial(ds.root_id);
      const radiusStep = 128;
      const radiusMax = Math.max(2, maxVisibleDepth) * radiusStep;
      const width = Math.max(780, radiusMax * 2 + 280);
      const height = Math.max(780, radiusMax * 2 + 280);
      const centerX = width / 2;
      const centerY = height / 2;
      const pos = {{}};
      visible.visibleNodes.forEach(node => {{
        const angle = angleById[node.id] || 0;
        const radius = node.depth * radiusStep;
        pos[node.id] = {{
          x: centerX + Math.cos(angle - Math.PI / 2) * radius,
          y: centerY + Math.sin(angle - Math.PI / 2) * radius,
        }};
      }});
      return {{ pos, width, height, centerX, centerY }};
    }}
    function edgePath(p0, p1) {{
      if (state.layout === 'flow') {{
        const dx = (p1.x - p0.x) * 0.38;
        return `M ${{p0.x}} ${{p0.y}} C ${{p0.x + dx}} ${{p0.y}}, ${{p1.x - dx}} ${{p1.y}}, ${{p1.x}} ${{p1.y}}`;
      }}
      if (state.layout === 'vertical') {{
        const dy = (p1.y - p0.y) * 0.4;
        return `M ${{p0.x}} ${{p0.y}} C ${{p0.x}} ${{p0.y + dy}}, ${{p1.x}} ${{p1.y - dy}}, ${{p1.x}} ${{p1.y}}`;
      }}
      return `M ${{p0.x}} ${{p0.y}} L ${{p1.x}} ${{p1.y}}`;
    }}
    function nodeFill(node) {{
      const spec = activeSpec();
      const leaf = node.kind !== 'branch';
      const sat = leaf ? 74 : 56;
      const light = Math.max(30, (leaf ? 88 : 92) - node.depth * 6.2);
      return `hsl(${{spec.hue}}, ${{sat}}%, ${{light}}%)`;
    }}
    function nodeRadius(node, dense) {{
      if (node.kind === 'branch') {{
        return dense ? Math.min(10, 5 + Math.log2((node.leaf_count || 1) + 1) * 1.15) : Math.min(16, 7 + Math.log2((node.leaf_count || 1) + 1) * 1.3);
      }}
      return dense ? 3.6 : 5.4;
    }}
    function labelText(node, dense) {{
      if (state.labelMode === 'branches' && node.kind !== 'branch') return '';
      if (state.labelMode === 'full') return node.label;
      if (state.labelMode === 'compact') {{
        /* Always show at least tiny label on the right; suppress only in extreme density (>600 nodes) */
        if (dense && visible_count > 600 && node.kind !== 'branch') return '';
        return dense ? node.tiny : (node.kind === 'branch' ? node.short : node.tiny);
      }}
      return node.short;
    }}
    function labelAttrs(node, radius, position, layoutInfo) {{
      if (state.layout === 'flow') {{
        return {{ x: radius + 9, y: 4, anchor: 'start' }};
      }}
      if (state.layout === 'vertical') {{
        return {{ x: 0, y: radius + 16, anchor: 'middle' }};
      }}
      const toRight = position.x >= layoutInfo.centerX;
      return {{ x: toRight ? radius + 9 : -radius - 9, y: 4, anchor: toRight ? 'start' : 'end' }};
    }}
    function renderFocusCards() {{
      const envData = DATA.sources[state.source];
      const html = ['ATTACK', 'OPINION', 'PROFILE'].map(ontology => {{
        const ds = envData[ontology];
        const spec = COLORS[ontology];
        return `
          <div class="ontx-focus-item ${{ontology === state.ontology ? 'active' : ''}}" data-ontology="${{ontology}}">
            <div class="ontx-focus-top">
              <strong>${{ontology === 'ATTACK' ? 'Attack Space' : ontology === 'OPINION' ? 'Opinion Space' : 'Profile Space'}}</strong>
              <span class="ontx-focus-pill" style="background:${{spec.soft}};color:${{spec.accent}}">${{ds.summary.leaf_count}} leaves</span>
            </div>
            <div class="ontx-focus-meta">
              <span>${{ds.summary.branch_count}} branches</span>
              <span>depth ${{ds.summary.max_depth}}</span>
              <span>${{ds.summary.node_count}} total nodes</span>
            </div>
          </div>`;
      }}).join('');
      root.querySelector('#ontx-focus').innerHTML = html;
      root.querySelectorAll('.ontx-focus-item').forEach(el => el.addEventListener('click', () => setOntology(el.dataset.ontology)));
    }}
    function renderCompare() {{
      const current = state.ontology;
      const html = ['production', 'test'].map(env => {{
        const ds = DATA.sources[env][current];
        const active = env === state.source;
        const isRun = env === DATA.current_run_source;
        return `
          <div class="ontx-compare-card ${{active ? 'active' : ''}}">
            <div class="ontx-compare-top">
              <strong>${{env === 'production' ? 'Production ontology' : 'Test ontology'}}</strong>
              <div style="display:flex;gap:6px;flex-wrap:wrap">
                ${{active ? badgeHtml('Visible source', 'rgba(29,78,137,0.08)', '{PALETTE['blue']}') : ''}}
                ${{isRun ? badgeHtml('Run 8 source', 'rgba(231,111,81,0.12)', '{PALETTE['orange']}') : ''}}
              </div>
            </div>
            <div class="ontx-compare-grid">
              <div class="ontx-metric"><div class="k">Leaves</div><div class="v">${{ds.summary.leaf_count}}</div></div>
              <div class="ontx-metric"><div class="k">Branches</div><div class="v">${{ds.summary.branch_count}}</div></div>
              <div class="ontx-metric"><div class="k">Depth</div><div class="v">${{ds.summary.max_depth}}</div></div>
              <div class="ontx-metric"><div class="k">Annotated leaves</div><div class="v">${{ds.summary.metadata_leaf_count}}</div></div>
            </div>
          </div>`;
      }}).join('');
      root.querySelector('#ontx-compare').innerHTML = html;
    }}
    function renderStatus(ds, visible) {{
      const dense = visible.visibleNodes.length > 260;
      const runNote = DATA.current_run_source === state.source
        ? 'Current source matches the ontology used in run 8.'
        : 'Current source is a comparison surface; switch to Test to inspect the exact run 8 ontology.';
      const relevantCount = ds.summary.sample_exact_count + ds.summary.sample_aligned_count;
      root.querySelector('#ontx-status').innerHTML =
        `<strong>Visible structure:</strong> ${{visible.visibleNodes.length}} of ${{ds.summary.node_count}} nodes, depth ≤ ${{state.maxDepth}}, ${{state.layout}} layout<br>` +
        `<strong>Run alignment:</strong> ${{relevantCount}} highlighted leaves available for this ontology across exact or leaf-name matching. ${{runNote}}<br>` +
        `<strong>Readability:</strong> ${{dense ? 'Dense mode is active; leaf labels are softened. Use search or reduce depth for cleaner local inspection.' : 'Local labels are fully readable at the current depth.'}}`;
    }}
    function renderCanvasHeader(ds, visible) {{
      root.querySelector('#ontx-active-title').textContent =
        `${{state.source === 'production' ? 'Production' : 'Test'}} · ${{state.ontology === 'ATTACK' ? 'Attack Ontology' : state.ontology === 'OPINION' ? 'Opinion Ontology' : 'Profile Ontology'}}`;
      root.querySelector('#ontx-active-sub').textContent =
        `Mixed hierarchical state space with ${{ds.summary.leaf_count}} terminal leaves, ${{ds.summary.branch_count}} branching nodes, and a maximum depth of ${{ds.summary.max_depth}}.`;
      root.querySelector('#ontx-chipline').innerHTML =
        badgeHtml(`Zoom ${{Math.round(state.zoom * 100)}}%`, 'rgba(20,33,61,0.06)', '{PALETTE['muted']}') +
        badgeHtml(`Visible ${{visible.visibleNodes.length}} nodes`, 'rgba(29,78,137,0.08)', '{PALETTE['blue']}') +
        badgeHtml(`Depth ${{state.maxDepth}}`, activeSpec().soft, activeSpec().accent);
    }}
    function renderSearchResults(ds) {{
      const q = state.search.trim();
      let items = [];
      if (q) {{
        items = searchMatches(ds);
      }} else {{
        items = ds.nodes
          .filter(n => n.depth === 1 || n.sample_exact || n.sample_aligned)
          .slice(0, 12);
      }}
      if (!items.length) {{
        root.querySelector('#ontx-results').innerHTML = '<div class="ontx-note">No ontology paths match the current search or relevance filter.</div>';
        return;
      }}
      root.querySelector('#ontx-results').innerHTML = items.map(node => `
        <div class="ontx-result" data-node-id="${{node.id}}">
          <strong>${{escapeHtml(node.label)}}</strong>
          <span>${{escapeHtml(node.path_label || node.label)}}</span>
        </div>`).join('');
      root.querySelectorAll('.ontx-result').forEach(el => el.addEventListener('click', () => {{
        const id = el.dataset.nodeId;
        expandAncestors(id);
        state.selectedId = id;
        renderAll();
      }}));
    }}
    function renderHighlights(ds, visible) {{
      let items = ds.nodes.filter(n => n.kind !== 'branch' && (n.sample_exact || n.sample_aligned));
      if (state.relevantOnly) {{
        const visibleSet = new Set(visible.visibleNodes.map(n => n.id));
        items = items.filter(n => visibleSet.has(n.id));
      }}
      items = items
        .sort((a, b) => (Number(b.sample_exact) - Number(a.sample_exact)) || a.path_label.localeCompare(b.path_label))
        .slice(0, 9);
      if (!items.length) {{
        root.querySelector('#ontx-highlights').innerHTML =
          `<div class="ontx-note">${{state.ontology === 'PROFILE'
            ? 'The profile ontology is shown structurally; this run does not carry a leaf-level selected profile subset like the attack and opinion factorial leaves.'
            : 'No run-aligned leaf highlights are available for the current source. Switch source or disable the relevance-only filter to inspect the full ontology.'}}</div>`;
        return;
      }}
      root.querySelector('#ontx-highlights').innerHTML = items.map(node => `
        <div class="ontx-highlight-item">
          <strong>${{escapeHtml(node.label)}} ${{node.sample_exact ? '· exact run leaf' : '· leaf-name aligned'}}</strong>
          <span>${{escapeHtml(node.path_label)}}</span>
        </div>`).join('');
    }}
    function renderInspector() {{
      const ds = dataset();
      const node = ds.nodeMap[state.selectedId] || ds.nodeMap[ds.root_id];
      if (!node) return;
      const badges = [
        badgeHtml(node.kind === 'branch' ? 'Branch' : (node.kind === 'leaf_meta' ? 'Leaf + metadata' : 'Leaf'), activeSpec().soft, activeSpec().accent),
        badgeHtml(`Depth ${{node.depth}}`, 'rgba(20,33,61,0.06)', '{PALETTE['muted']}'),
      ];
      if (node.sample_exact) badges.push(badgeHtml('Exact run-aligned', 'rgba(240,192,64,0.18)', '{PALETTE['amber']}'));
      if (node.sample_aligned) badges.push(badgeHtml('Leaf-name aligned', 'rgba(200,155,60,0.12)', '{PALETTE['amber']}'));
      const children = (node.children || []).slice(0, 8).map(id => ds.nodeMap[id]).filter(Boolean);
      const metaHtml = node.metadata_preview && node.metadata_preview.length
        ? `<div class="ontx-meta-list">${{node.metadata_preview.map(([k,v]) => `
            <div class="ontx-meta-item">
              <strong>${{escapeHtml(k)}}</strong>
              <span>${{escapeHtml(v)}}</span>
            </div>`).join('')}}</div>`
        : `<div class="ontx-note">No leaf-level metadata stored on this node.</div>`;
      const childHtml = children.length
        ? `<div class="ontx-meta-list">${{children.map(child => `
            <div class="ontx-meta-item">
              <strong>${{escapeHtml(child.label)}}</strong>
              <span>${{child.leaf_count}} leaves below · ${{child.child_count}} direct children</span>
            </div>`).join('')}}</div>`
        : '';

      root.querySelector('#ontx-inspector').innerHTML = `
        <div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start;margin-bottom:10px">
          <div>
            <div style="font-weight:800;font-size:0.92rem;color:{PALETTE['ink']}">${{escapeHtml(node.label)}}</div>
            <div class="ontx-sub" style="margin:3px 0 0">${{node.parent ? 'Selected ontology node' : 'Synthetic ontology root used to organize the JSON hierarchy for visualization'}}</div>
          </div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end">${{badges.join('')}}</div>
        </div>
        <div class="ontx-path">${{escapeHtml(node.path_label || node.label)}}</div>
        <div class="ontx-kv" style="margin-top:10px">
          <div class="ontx-metric"><div class="k">Direct children</div><div class="v">${{node.child_count}}</div></div>
          <div class="ontx-metric"><div class="k">Leaves below</div><div class="v">${{node.leaf_count}}</div></div>
          <div class="ontx-metric"><div class="k">Subtree nodes</div><div class="v">${{node.subtree_node_count}}</div></div>
          <div class="ontx-metric"><div class="k">Metadata leaves below</div><div class="v">${{node.metadata_leaf_count}}</div></div>
        </div>
        <div style="margin-top:10px;font-weight:700;font-size:0.77rem;color:{PALETTE['navy']};margin-bottom:6px">Metadata preview</div>
        ${{metaHtml}}
        ${{children.length ? '<div style="margin-top:10px;font-weight:700;font-size:0.77rem;color:{PALETTE['navy']};margin-bottom:6px">Visible child branches</div>' + childHtml : ''}}
      `;
    }}
    let visible_count = 0;
    function renderGraph() {{
      const ds = dataset();
      const visible = collectVisible(ds);
      visible_count = visible.visibleNodes.length;
      renderStatus(ds, visible);
      renderCanvasHeader(ds, visible);
      const searchIds = new Set(searchMatches(ds).map(n => n.id));
      const layout = layoutTree(ds, visible);
      const dense = visible_count > 260;
      const svgWidth = layout.width * state.zoom;
      const svgHeight = layout.height * state.zoom;
      let svg = `<svg id="ontx-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${{layout.width}} ${{layout.height}}" width="${{svgWidth}}" height="${{svgHeight}}">`;
      svg += `<defs>
        <filter id="ontx-shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="rgba(20,33,61,0.12)"/>
        </filter>
      </defs>`;

      visible.visibleEdges.forEach(([a, b]) => {{
        const p0 = layout.pos[a];
        const p1 = layout.pos[b];
        const accent = activeSpec().edge;
        const boosted = searchIds.has(a) || searchIds.has(b);
        svg += `<path d="${{edgePath(p0, p1)}}" fill="none" stroke="${{boosted ? activeSpec().accent : accent}}" stroke-width="${{boosted ? 2.2 : 1.4}}" stroke-linecap="round"/>`;
      }});

      visible.visibleNodes.forEach(node => {{
        const pos = layout.pos[node.id];
        const selected = node.id === state.selectedId;
        const collapsed = node.kind === 'branch' && node.depth < state.maxDepth && !(currentExpanded().has(node.id) || node.id === ds.root_id);
        const radius = nodeRadius(node, dense);
        const fill = nodeFill(node);
        const stroke = selected ? '{PALETTE['gold']}' : activeSpec().accent;
        const label = labelText(node, dense);
        const attrs = label ? labelAttrs(node, radius, pos, layout) : null;
        const searchMatch = searchIds.has(node.id);
        svg += `<g class="ontx-node" data-node-id="${{node.id}}" style="cursor:pointer">`;
        if (state.highlightRun && (node.sample_exact || node.sample_aligned)) {{
          const ringColor = node.sample_exact ? '{PALETTE['gold']}' : '{PALETTE['amber']}';
          const dash = node.sample_exact ? '' : ' stroke-dasharray="4 3"';
          svg += `<circle cx="${{pos.x}}" cy="${{pos.y}}" r="${{radius + 4.3}}" fill="none" stroke="${{ringColor}}" stroke-width="2.2"${{dash}} opacity="0.98"/>`;
        }}
        if (state.showMetadata && node.kind === 'leaf_meta') {{
          svg += `<circle cx="${{pos.x}}" cy="${{pos.y}}" r="${{radius + 2.6}}" fill="none" stroke="{PALETTE['muted']}" stroke-width="1.2" stroke-dasharray="3 2" opacity="0.72"/>`;
        }}
        if (searchMatch) {{
          svg += `<circle cx="${{pos.x}}" cy="${{pos.y}}" r="${{radius + 6.6}}" fill="none" stroke="{PALETTE['sky']}" stroke-width="2" opacity="0.55"/>`;
        }}
        svg += `<circle cx="${{pos.x}}" cy="${{pos.y}}" r="${{radius}}" fill="${{fill}}" stroke="${{stroke}}" stroke-width="${{selected ? 2.6 : 1.4}}" filter="url(#ontx-shadow)"/>`;
        if (node.kind === 'branch') {{
          svg += `<text x="${{pos.x}}" y="${{pos.y + 3.2}}" text-anchor="middle" style="font:700 8px IBM Plex Sans, sans-serif;fill:${{collapsed ? '#ffffff' : '{PALETTE['ink']}'}}">${{collapsed ? '+' : '−'}}</text>`;
        }}
        if (label) {{
          svg += `<text x="${{pos.x + attrs.x}}" y="${{pos.y + attrs.y}}" text-anchor="${{attrs.anchor}}" style="font:600 ${{dense ? 8.6 : 9.4}}px IBM Plex Sans, sans-serif;fill:{PALETTE['ink']};paint-order:stroke;stroke:white;stroke-width:3">${{escapeHtml(label)}}</text>`;
        }}
        svg += `</g>`;
      }});
      svg += `</svg>`;
      root.querySelector('#ontx-canvas-wrap').innerHTML = svg;
      root.querySelectorAll('.ontx-node').forEach(el => el.addEventListener('click', ev => {{
        ev.stopPropagation();
        const id = el.dataset.nodeId;
        const node = ds.nodeMap[id];
        state.selectedId = id;
        if (node && node.kind === 'branch') {{
          if (currentExpanded().has(id)) {{
            currentExpanded().delete(id);
          }} else {{
            currentExpanded().add(id);
            /* Auto-extend maxDepth so the newly expanded node can show its children */
            if (node.depth >= state.maxDepth) {{
              state.maxDepth = node.depth + 1;
              depthStore[datasetKey()] = state.maxDepth;
              root.querySelector('#ontx-depth').value = state.maxDepth;
              root.querySelector('#ontx-depth-display').textContent = state.maxDepth;
            }}
          }}
        }}
        renderGraph();
        renderInspector();
      }}));
      renderHighlights(ds, visible);
    }}
    function renderAll() {{
      renderFocusCards();
      renderCompare();
      renderSearchResults(dataset());
      renderGraph();
      renderInspector();
    }}

    root.querySelectorAll('#ontx-source .ontx-btn').forEach(btn => btn.addEventListener('click', () => setSource(btn.dataset.source)));
    root.querySelectorAll('#ontx-layout .ontx-btn').forEach(btn => btn.addEventListener('click', () => setLayout(btn.dataset.layout)));
    root.querySelector('#ontx-depth').addEventListener('input', ev => {{
      state.maxDepth = parseInt(ev.target.value, 10);
      depthStore[datasetKey()] = state.maxDepth;
      root.querySelector('#ontx-depth-display').textContent = state.maxDepth;
      renderGraph();
      renderInspector();
    }});
    root.querySelector('#ontx-label-mode').addEventListener('change', ev => {{
      state.labelMode = ev.target.value;
      renderGraph();
    }});
    root.querySelector('#ontx-show-meta').addEventListener('change', ev => {{
      state.showMetadata = ev.target.checked;
      renderGraph();
    }});
    root.querySelector('#ontx-highlight-run').addEventListener('change', ev => {{
      state.highlightRun = ev.target.checked;
      renderGraph();
      renderInspector();
    }});
    root.querySelector('#ontx-relevant-only').addEventListener('change', ev => {{
      state.relevantOnly = ev.target.checked;
      renderAll();
    }});
    root.querySelector('#ontx-search').addEventListener('input', ev => {{
      state.search = ev.target.value;
      renderSearchResults(dataset());
      renderGraph();
    }});
    root.querySelector('#ontx-expand-all').addEventListener('click', expandAll);
    root.querySelector('#ontx-collapse-all').addEventListener('click', collapseAll);
    root.querySelector('#ontx-reset-depth').addEventListener('click', resetToDepth);
    root.querySelector('#ontx-zoom-in').addEventListener('click', () => {{
      state.zoom = Math.min(2.4, state.zoom + 0.15);
      renderGraph();
    }});
    root.querySelector('#ontx-zoom-out').addEventListener('click', () => {{
      state.zoom = Math.max(0.55, state.zoom - 0.15);
      renderGraph();
    }});
    root.querySelector('#ontx-zoom-reset').addEventListener('click', () => {{
      state.zoom = 1;
      renderGraph();
    }});

    /* ── Drag / pan on canvas-wrap ─────────────────────────────────────── */
    (function() {{
      const wrap = root.querySelector('#ontx-canvas-wrap');
      let drag = null;
      wrap.style.cursor = 'grab';
      wrap.addEventListener('mousedown', e => {{
        if (e.target.closest('.ontx-node')) return;
        drag = {{ sx: e.clientX + wrap.scrollLeft, sy: e.clientY + wrap.scrollTop }};
        wrap.style.cursor = 'grabbing';
        e.preventDefault();
      }});
      window.addEventListener('mousemove', e => {{
        if (!drag) return;
        wrap.scrollLeft = drag.sx - e.clientX;
        wrap.scrollTop  = drag.sy - e.clientY;
      }});
      window.addEventListener('mouseup', () => {{ drag = null; wrap.style.cursor = 'grab'; }});
      /* touch support */
      wrap.addEventListener('touchstart', e => {{
        if (e.touches.length !== 1 || e.target.closest('.ontx-node')) return;
        const t = e.touches[0];
        drag = {{ sx: t.clientX + wrap.scrollLeft, sy: t.clientY + wrap.scrollTop }};
      }}, {{passive: true}});
      wrap.addEventListener('touchmove', e => {{
        if (!drag || e.touches.length !== 1) return;
        const t = e.touches[0];
        wrap.scrollLeft = drag.sx - t.clientX;
        wrap.scrollTop  = drag.sy - t.clientY;
        e.preventDefault();
      }}, {{passive: false}});
      wrap.addEventListener('touchend', () => {{ drag = null; }});
    }})();

    initDatasets();
    syncButtons();
    renderAll();
  }})();
  </script>
</div>"""

def _fig_factorial_3d(long_df: pd.DataFrame) -> go.Figure:
    """Dual go.Surface: mean AE (RdBu_r) + ISD of AE (YlOrRd)."""
    atk_col = "attack_leaf" if "attack_leaf" in long_df.columns else "attack_leaf_label"
    op_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    ae_col = "adversarial_effectivity"
    for c in (atk_col, op_col, ae_col):
        if c not in long_df.columns:
            return go.Figure().add_annotation(text=f"Column '{c}' missing", showarrow=False)

    attacks  = sorted(long_df[atk_col].dropna().unique())
    opinions = sorted(long_df[op_col].dropna().unique())
    attack_labels = _unique_display_map(attacks)
    opinion_labels = _unique_display_map(opinions)

    def _matrix(func):
        return (
            long_df.groupby([atk_col, op_col])[ae_col].agg(func)
            .unstack(op_col).reindex(index=attacks, columns=opinions).fillna(0)
        )

    mean_mat = _matrix("mean")
    isd_mat  = _matrix("std")
    atk_l    = [_wrap_label(attack_labels[a], 18) for a in attacks]
    op_l     = [_wrap_label(opinion_labels[o], 18) for o in opinions]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Mean Adversarial Effectivity (AE)", "Inter-individual Variability (SD of AE)"],
        horizontal_spacing=0.12,
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
    atk_col = "attack_leaf" if "attack_leaf" in long_df.columns else "attack_leaf_label"
    op_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    ae_col = "adversarial_effectivity"
    if ae_col not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    attacks  = sorted(long_df[atk_col].dropna().unique())
    opinions = sorted(long_df[op_col].dropna().unique())
    attack_labels = _unique_display_map(attacks)
    opinion_labels = _unique_display_map(opinions)
    atk_l    = [_wrap_label(attack_labels[a], 18) for a in attacks]
    op_l     = [_wrap_label(opinion_labels[o], 18) for o in opinions]

    mean_m = (long_df.groupby([atk_col, op_col])[ae_col].mean()
              .unstack(op_col).reindex(index=attacks, columns=opinions).fillna(0))
    isd_m  = (long_df.groupby([atk_col, op_col])[ae_col].std()
              .unstack(op_col).reindex(index=attacks, columns=opinions).fillna(0))

    total_cells = len(attacks) * len(opinions)
    # Always side-by-side; let Plotly auto-manage the axis tick labels
    rows, cols = 1, 2
    fig = make_subplots(
        rows,
        cols,
        subplot_titles=["Mean AE (red = manipulation succeeded)", "Inter-individual SD of AE"],
        horizontal_spacing=0.20,
    )
    text_size = 11 if total_cells <= 25 else 9 if total_cells <= 49 else 0
    show_text = text_size > 0

    common = dict(
        hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>%{z:.1f}<extra></extra>",
        xgap=4,
        ygap=4,
    )
    mean_kwargs = {
        "z": mean_m.values,
        "x": op_l,
        "y": atk_l,
        "colorscale": "RdBu_r",
        "zmid": 0,
        "colorbar": dict(
            x=0.44, y=0.50, len=0.82,
            thickness=12, title="AE", title_side="right",
        ),
        **common,
    }
    isd_kwargs = {
        "z": isd_m.values,
        "x": op_l,
        "y": atk_l,
        "colorscale": "YlOrRd",
        "colorbar": dict(
            x=1.02, y=0.50, len=0.82,
            thickness=12, title="SD", title_side="right",
        ),
        **common,
    }
    if show_text:
        mean_kwargs.update(
            text=[[f"{v:.1f}" for v in row] for row in mean_m.values],
            texttemplate="%{text}",
            textfont=dict(size=text_size, color="white"),
        )
        isd_kwargs.update(
            text=[[f"{v:.1f}" for v in row] for row in isd_m.values],
            texttemplate="%{text}",
            textfont=dict(size=text_size, color="white"),
        )

    fig.add_trace(go.Heatmap(**mean_kwargs), row=1, col=1)
    fig.add_trace(go.Heatmap(**isd_kwargs), row=1, col=2)

    fig.update_xaxes(tickangle=-28, tickfont_size=9, automargin=True)
    fig.update_yaxes(tickfont_size=9, automargin=True)
    fig.update_annotations(font=dict(size=12.5))
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=470,
        margin=dict(l=160, r=90, t=68, b=130),
        title=dict(
            text="Factorial Heatmap — Mean AE & Inter-individual Moderation Strength",
            font_size=14,
        ),
    )
    return fig


def _fig_sem_network(
    sem_coeff_df: pd.DataFrame,
    long_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """3D hierarchical SEM map with profile ontology, attack scope, and opinion space."""
    df = sem_coeff_df[sem_coeff_df["op"] == "~"].copy()
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df = df.dropna(subset=["estimate"])
    if df.empty:
        return go.Figure().add_annotation(text="No SEM path data", showarrow=False)

    opinion_lookup: Dict[str, str] = {}
    opinion_group_lookup: Dict[str, str] = {}
    attack_stats = pd.DataFrame()
    if long_df is not None and not long_df.empty:
        opinion_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
        attack_col = "attack_leaf" if "attack_leaf" in long_df.columns else "attack_leaf_label"
        if opinion_col in long_df.columns:
            opinion_values = sorted(long_df[opinion_col].dropna().unique())
            opinion_display = _unique_display_map(opinion_values)
            for opinion_value in opinion_values:
                leaf_name = _leaf(opinion_value)
                opinion_lookup.setdefault(leaf_name, opinion_display[opinion_value])
                opinion_group_lookup.setdefault(leaf_name, _path_context(opinion_value, keep=1) or "Opinion Targets")
        if attack_col in long_df.columns:
            attack_stats = (
                long_df.groupby(attack_col, as_index=False)
                .agg(
                    mean_ae=("adversarial_effectivity", "mean"),
                    mean_abs=("abs_delta_score", "mean"),
                    sd_ae=("adversarial_effectivity", "std"),
                    n_rows=("scenario_id", "count"),
                )
            )

    df["rhs_label"] = df["rhs"].astype(str).map(lambda s: re.sub(r"\s+", " ", str(s)).strip())
    df["lhs_leaf"] = df["lhs"].astype(str).map(_pretty_indicator)
    df["lhs_label"] = df["lhs_leaf"].map(lambda leaf: opinion_lookup.get(leaf, leaf))
    df["mod_root"] = df["rhs_label"].map(lambda s: _infer_sem_moderator_groups(s)[0])
    df["mod_group"] = df["rhs_label"].map(lambda s: _infer_sem_moderator_groups(s)[1])
    df["ind_group"] = df["lhs_leaf"].map(lambda s: opinion_group_lookup.get(s, "Opinion Targets"))

    mod_rank = (
        df.groupby(["mod_root", "rhs_label"], as_index=False)
        .agg(abs_est=("estimate", lambda s: float(np.max(np.abs(s)))))
        .sort_values(["mod_root", "abs_est", "rhs_label"], ascending=[True, False, True])
    )
    ind_rank = (
        df.groupby(["ind_group", "lhs_label"], as_index=False)
        .agg(abs_est=("estimate", lambda s: float(np.max(np.abs(s)))))
        .sort_values(["ind_group", "abs_est", "lhs_label"], ascending=[True, False, True])
    )

    mod_groups = (
        mod_rank.groupby("mod_root", as_index=False)["abs_est"]
        .sum()
        .sort_values("abs_est", ascending=False)["mod_root"]
        .tolist()
    )
    ind_groups = (
        ind_rank.groupby("ind_group", as_index=False)["abs_est"]
        .sum()
        .sort_values("abs_est", ascending=False)["ind_group"]
        .tolist()
    )
    mod_group_members = {
        group: mod_rank.loc[mod_rank["mod_root"] == group, "rhs_label"].tolist()
        for group in mod_groups
    }
    ind_group_members = {
        group: ind_rank.loc[ind_rank["ind_group"] == group, "lhs_label"].tolist()
        for group in ind_groups
    }

    def _lane_layout(group_members: Dict[str, List[str]], group_order: List[str]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        n_groups = max(len(group_order), 1)
        z_vals = np.linspace((n_groups - 1) * 2.6 / 2, -(n_groups - 1) * 2.6 / 2, n_groups) if n_groups > 1 else np.array([0.0])
        group_pos: Dict[str, Tuple[float, float]] = {}
        leaf_pos: Dict[str, Tuple[float, float]] = {}
        for z_lane, group in zip(z_vals, group_order):
            labels = group_members.get(group, [])
            if len(labels) <= 1:
                y_vals = np.array([0.0] * max(len(labels), 1))
            else:
                y_vals = np.linspace((len(labels) - 1) * 1.15 / 2, -(len(labels) - 1) * 1.15 / 2, len(labels))
            group_pos[group] = (0.0, float(z_lane))
            for y_val, label in zip(y_vals, labels):
                leaf_pos[label] = (float(y_val), float(z_lane))
        return group_pos, leaf_pos

    mod_group_pos, mod_leaf_pos = _lane_layout(mod_group_members, mod_groups)
    ind_group_pos, ind_leaf_pos = _lane_layout(ind_group_members, ind_groups)

    if not attack_stats.empty:
        attack_stats = attack_stats.copy()
        attack_stats["attack_label"] = attack_stats[attack_stats.columns[0]].astype(str)
        attack_stats["attack_group"] = attack_stats["attack_label"].map(lambda s: _path_context(s, keep=1) or "Attack Scope")
        attack_order = (
            attack_stats.groupby("attack_group", as_index=False)["mean_abs"]
            .sum()
            .sort_values("mean_abs", ascending=False)["attack_group"]
            .tolist()
        )
        attack_group_members = {
            group: attack_stats.loc[attack_stats["attack_group"] == group]
            .sort_values("mean_abs", ascending=False)["attack_label"].tolist()
            for group in attack_order
        }
        attack_group_pos, attack_leaf_pos = _lane_layout(attack_group_members, attack_order)
    else:
        attack_order = []
        attack_group_pos = {}
        attack_leaf_pos = {}

    mod_order = [label for group in mod_groups for label in mod_group_members[group]]
    ind_order = [label for group in ind_groups for label in ind_group_members[group]]
    mod_codes = {label: f"M{i+1:02d}" for i, label in enumerate(mod_order)}
    ind_codes = {label: f"O{i+1:02d}" for i, label in enumerate(ind_order)}
    attack_codes = {label: f"A{i+1:02d}" for i, label in enumerate(sum([attack_group_members[g] for g in attack_order], []))}

    max_abs = float(df["estimate"].abs().max() or 1.0)
    group_edges = (
        df.groupby(["mod_root", "ind_group"], as_index=False)
        .agg(
            mean_est=("estimate", "mean"),
            mean_abs=("estimate", lambda s: float(np.mean(np.abs(s)))),
            min_p=("p_value", "min"),
            n_paths=("estimate", "count"),
        )
    )

    def _line_rgba(est: float, p_val: float | None, strong_alpha: float = 0.95) -> str:
        if p_val is None or pd.isna(p_val):
            alpha = 0.18
        elif p_val < 0.01:
            alpha = strong_alpha
        elif p_val < 0.05:
            alpha = 0.78
        elif p_val < 0.10:
            alpha = 0.48
        else:
            alpha = 0.18
        return f"rgba(29,78,137,{alpha})" if est >= 0 else f"rgba(192,57,43,{alpha})"

    x_mod_group, x_mod_leaf, x_center, x_ind_group, x_ind_leaf = 0.0, 1.0, 2.35, 3.7, 4.7
    z_all = [pos[1] for pos in mod_group_pos.values()] + [pos[1] for pos in ind_group_pos.values()] + [pos[1] for pos in attack_group_pos.values()]
    z_min = min(z_all) - 1.2 if z_all else -2.0
    z_max = max(z_all) + 1.2 if z_all else 2.0
    y_max_candidates = [abs(pos[0]) for pos in mod_leaf_pos.values()] + [abs(pos[0]) for pos in ind_leaf_pos.values()] + [abs(pos[0]) for pos in attack_leaf_pos.values()]
    y_limit = max(y_max_candidates + [1.8]) + 0.9

    traces: List[go.BaseTraceType] = []
    group_edge_count = 0
    leaf_edge_count = 0
    leaf_edge_meta: List[Dict[str, Any]] = []

    plane_y = np.array([[-y_limit, y_limit], [-y_limit, y_limit]])
    plane_z = np.array([[z_min, z_min], [z_max, z_max]])
    plane_x = np.full((2, 2), x_center)
    traces.append(go.Surface(
        x=plane_x,
        y=plane_y,
        z=plane_z,
        showscale=False,
        opacity=0.14,
        hoverinfo="skip",
        colorscale=[[0, "rgba(240,192,64,0.28)"], [1, "rgba(231,111,81,0.18)"]],
        name="AE corridor",
    ))

    for _, row in group_edges.iterrows():
        mod_group = str(row["mod_root"])
        ind_group = str(row["ind_group"])
        if mod_group not in mod_group_pos or ind_group not in ind_group_pos:
            continue
        y0, z0 = mod_group_pos[mod_group]
        y1, z1 = ind_group_pos[ind_group]
        est = float(row["mean_est"])
        p_val = float(row["min_p"]) if pd.notna(row["min_p"]) else None
        p_text = f"{p_val:.4f}" if p_val is not None else "n/a"
        traces.append(go.Scatter3d(
            x=[x_mod_group, 1.55, x_center, 3.05, x_ind_group],
            y=[y0, y0 * 0.5, (y0 + y1) / 2, y1 * 0.5, y1],
            z=[z0, z0, (z0 + z1) / 2, z1, z1],
            mode="lines",
            line=dict(color=_line_rgba(est, p_val, strong_alpha=0.88), width=max(5.0, row["mean_abs"] / max_abs * 12.0)),
            hovertemplate=(
                f"<b>{mod_group}</b> → <b>{ind_group}</b><br>"
                f"Mean β = {est:.3f}<br>"
                f"Mean |β| = {float(row['mean_abs']):.3f}<br>"
                f"Paths aggregated = {int(row['n_paths'])}<br>"
                f"Best p = {p_text}<extra>Group summary</extra>"
            ),
            showlegend=False,
            visible=True,
        ))
        group_edge_count += 1

    for _, row in df.iterrows():
        rhs = str(row["rhs_label"])
        lhs = str(row["lhs_label"])
        if rhs not in mod_leaf_pos or lhs not in ind_leaf_pos:
            continue
        y0, z0 = mod_leaf_pos[rhs]
        y1, z1 = ind_leaf_pos[lhs]
        est = float(row["estimate"])
        p_val = float(row["p_value"]) if pd.notna(row["p_value"]) else None
        p_text = f"{p_val:.4f}" if p_val is not None else "n/a"
        traces.append(go.Scatter3d(
            x=[x_mod_leaf, 1.75, x_center, 2.95, x_ind_leaf],
            y=[y0, y0 * 0.62, (y0 + y1) / 2, y1 * 0.62, y1],
            z=[z0, z0, (z0 + z1) / 2, z1, z1],
            mode="lines",
            line=dict(color=_line_rgba(est, p_val), width=max(3.0, abs(est) / max_abs * 9.0)),
            hovertemplate=(
                f"<b>{mod_codes.get(rhs, rhs)}</b> {rhs}<br>"
                f"→ <b>{ind_codes.get(lhs, lhs)}</b> {lhs}<br>"
                f"β = {est:.3f} {_p_stars(p_val)}<br>"
                f"p = {p_text}<extra>Leaf path</extra>"
            ),
            showlegend=False,
            visible=False,
        ))
        leaf_edge_meta.append({"p": p_val if p_val is not None else 1.0})
        leaf_edge_count += 1

    # group nodes
    traces.extend([
        go.Scatter3d(
            x=[x_mod_group] * len(mod_groups),
            y=[mod_group_pos[g][0] for g in mod_groups],
            z=[mod_group_pos[g][1] for g in mod_groups],
            mode="markers+text",
            marker=dict(size=16, color="#dbe8fb", symbol="square", line=dict(color=PALETTE["navy"], width=2)),
            text=[_clip_label(g, 22) for g in mod_groups],
            textposition="middle left",
            textfont=dict(size=10, color=PALETTE["navy"]),
            customdata=np.array(mod_groups, dtype=object),
            hovertemplate="<b>%{customdata}</b><extra>Profile family</extra>",
            showlegend=False,
            visible=True,
        ),
        go.Scatter3d(
            x=[x_ind_group] * len(ind_groups),
            y=[ind_group_pos[g][0] for g in ind_groups],
            z=[ind_group_pos[g][1] for g in ind_groups],
            mode="markers+text",
            marker=dict(size=16, color="#d8f2ef", symbol="square", line=dict(color=PALETTE["teal"], width=2)),
            text=[_clip_label(g, 22) for g in ind_groups],
            textposition="middle right",
            textfont=dict(size=10, color=PALETTE["teal"]),
            customdata=np.array(ind_groups, dtype=object),
            hovertemplate="<b>%{customdata}</b><extra>Opinion family</extra>",
            showlegend=False,
            visible=True,
        ),
    ])

    # attack scope nodes
    if not attack_stats.empty:
        attack_hover = []
        attack_sizes = []
        attack_text = []
        attack_labels_order = sum([attack_group_members[g] for g in attack_order], [])
        mean_abs_max = float(max(attack_stats["mean_abs"].max(), 0.01))
        for attack_label in attack_labels_order:
            row = attack_stats.loc[attack_stats["attack_label"] == attack_label].iloc[0]
            attack_hover.append([
                attack_codes[attack_label],
                attack_label,
                float(row["mean_ae"]),
                float(row["mean_abs"]),
                float(row["sd_ae"]) if pd.notna(row["sd_ae"]) else 0.0,
                int(row["n_rows"]),
            ])
            attack_sizes.append(10 + float(row["mean_abs"]) / mean_abs_max * 14)
            attack_text.append(attack_codes[attack_label] if len(attack_labels_order) > 5 else _clip_label(_leaf(attack_label), 16))
        traces.append(go.Scatter3d(
            x=[x_center] * len(attack_labels_order),
            y=[attack_leaf_pos[a][0] for a in attack_labels_order],
            z=[attack_leaf_pos[a][1] for a in attack_labels_order],
            mode="markers+text",
            marker=dict(
                size=attack_sizes,
                color=[v[3] for v in attack_hover],
                colorscale="YlOrRd",
                line=dict(color="white", width=1.6),
                opacity=0.92,
            ),
            text=attack_text,
            textposition="top center",
            textfont=dict(size=9, color=PALETTE["orange"]),
            customdata=np.array(attack_hover, dtype=object),
            hovertemplate=(
                "<b>%{customdata[0]}</b> %{customdata[1]}<br>"
                "Mean AE = %{customdata[2]:.2f}<br>"
                "Mean |Δ| = %{customdata[3]:.2f}<br>"
                "SD AE = %{customdata[4]:.2f}<br>"
                "Rows = %{customdata[5]}<extra>Attack scope context</extra>"
            ),
            showlegend=False,
            visible=True,
        ))

    traces.extend([
        go.Scatter3d(
            x=[x_mod_leaf] * len(mod_order),
            y=[mod_leaf_pos[label][0] for label in mod_order],
            z=[mod_leaf_pos[label][1] for label in mod_order],
            mode="markers+text",
            marker=dict(size=10, color=PALETTE["navy"], symbol="circle", line=dict(color="white", width=1.6)),
            text=[mod_codes[label] for label in mod_order],
            textposition="middle left",
            textfont=dict(size=9, color=PALETTE["ink"]),
            customdata=np.array([[mod_codes[label], label, _infer_sem_moderator_groups(label)[0]] for label in mod_order], dtype=object),
            hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>Family: %{customdata[2]}<extra>Moderator leaf</extra>",
            showlegend=False,
            visible=False,
        ),
        go.Scatter3d(
            x=[x_ind_leaf] * len(ind_order),
            y=[ind_leaf_pos[label][0] for label in ind_order],
            z=[ind_leaf_pos[label][1] for label in ind_order],
            mode="markers+text",
            marker=dict(size=10, color=PALETTE["teal"], symbol="diamond", line=dict(color="white", width=1.6)),
            text=[ind_codes[label] for label in ind_order],
            textposition="middle right",
            textfont=dict(size=9, color=PALETTE["ink"]),
            customdata=np.array([[ind_codes[label], label, next((g for g in ind_groups if label in ind_group_members[g]), "Opinion Targets")] for label in ind_order], dtype=object),
            hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>Family: %{customdata[2]}<extra>Opinion leaf</extra>",
            showlegend=False,
            visible=False,
        ),
    ])

    # trace visibility masks
    base_count = 1  # center plane
    group_edge_start = 1
    leaf_edge_start = group_edge_start + group_edge_count
    node_start = leaf_edge_start + leaf_edge_count
    total_traces = len(traces)

    attack_trace_count = 1 if not attack_stats.empty else 0
    always_visible_idx = {0, node_start, node_start + 1}
    if attack_trace_count:
        always_visible_idx.add(node_start + 2)
    mod_leaf_trace_idx = node_start + 2 + attack_trace_count
    ind_leaf_trace_idx = node_start + 3 + attack_trace_count

    def _mask(view: str) -> List[bool]:
        vis = [False] * total_traces
        for idx in always_visible_idx:
            if idx < total_traces:
                vis[idx] = True
        if view == "group":
            for idx in range(group_edge_start, leaf_edge_start):
                vis[idx] = True
        elif view == "leaf":
            vis[mod_leaf_trace_idx] = True
            vis[ind_leaf_trace_idx] = True
            for idx in range(leaf_edge_start, node_start):
                vis[idx] = True
        elif view == "sig":
            vis[mod_leaf_trace_idx] = True
            vis[ind_leaf_trace_idx] = True
            for idx, meta in enumerate(leaf_edge_meta, start=leaf_edge_start):
                vis[idx] = meta["p"] < 0.05
        elif view == "hsig":
            vis[mod_leaf_trace_idx] = True
            vis[ind_leaf_trace_idx] = True
            for idx, meta in enumerate(leaf_edge_meta, start=leaf_edge_start):
                vis[idx] = meta["p"] < 0.01
        return vis

    cameras = {
        "Perspective": dict(eye=dict(x=1.75, y=1.55, z=0.95)),
        "Profile Side": dict(eye=dict(x=0.15, y=2.3, z=0.55)),
        "Opinion Side": dict(eye=dict(x=-0.15, y=-2.3, z=0.55)),
        "Top Down": dict(eye=dict(x=0.0, y=0.15, z=2.65)),
    }

    fig = go.Figure(traces)
    fig.update_layout(
        paper_bgcolor="white",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=max(760, 90 * max(len(mod_groups), len(ind_groups)) + 320),
        title=dict(text="Hierarchical Structural Equation Model — 3D Moderation Space", font_size=14),
        margin=dict(l=20, r=20, t=92, b=110),
        scene=dict(
            xaxis=dict(visible=False, range=[-0.35, 5.05]),
            yaxis=dict(visible=False, range=[-y_limit, y_limit]),
            zaxis=dict(visible=False, range=[z_min, z_max]),
            aspectmode="manual",
            aspectratio=dict(x=2.4, y=1.6, z=1.1),
            bgcolor="white",
            camera=dict(eye=dict(x=1.75, y=1.55, z=0.95)),
        ),
        annotations=[
            dict(x=0.12, y=1.08, xref="paper", yref="paper", showarrow=False,
                 text="<b>PROFILE ONTOLOGY</b>", font=dict(size=11, color=PALETTE["navy"])),
            dict(x=0.50, y=1.08, xref="paper", yref="paper", showarrow=False,
                 text="<b>ATTACK-EFFECTIVITY SCOPE</b>", font=dict(size=11, color=PALETTE["orange"])),
            dict(x=0.88, y=1.08, xref="paper", yref="paper", showarrow=False,
                 text="<b>OPINION SPACE</b>", font=dict(size=11, color=PALETTE["teal"])),
            dict(x=0.50, y=-0.13, xref="paper", yref="paper", showarrow=False,
                 text=(
                     "Default view collapses to group summary for readability. Switch to leaf detail for M## / O## codes; hover nodes and paths for full labels and coefficients.<br>"
                     "Attack nodes in the middle summarize the manipulated attack space for the run; SEM paths themselves connect profile-side moderators to attack-effectivity indicators in opinion space."
                 ),
                 font=dict(size=8.6, color=PALETTE["muted"])),
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.24,
                xanchor="center",
                y=1.17,
                yanchor="top",
                buttons=[
                    dict(label="Group Summary", method="update", args=[{"visible": _mask("group")}]),
                    dict(label="Leaf Detail", method="update", args=[{"visible": _mask("leaf")}]),
                    dict(label="Significant", method="update", args=[{"visible": _mask("sig")}]),
                    dict(label="Highly Significant", method="update", args=[{"visible": _mask("hsig")}]),
                ],
                bgcolor=PALETTE["panel"],
                bordercolor=PALETTE["line"],
                font=dict(size=10),
                pad=dict(l=4, r=4, t=4, b=4),
            ),
            dict(
                type="buttons",
                direction="right",
                x=0.76,
                xanchor="center",
                y=1.17,
                yanchor="top",
                buttons=[
                    dict(label=name, method="relayout", args=[{"scene.camera": camera}])
                    for name, camera in cameras.items()
                ],
                bgcolor=PALETTE["panel"],
                bordercolor=PALETTE["line"],
                font=dict(size=10),
                pad=dict(l=4, r=4, t=4, b=4),
            ),
        ],
    )
    fig.update_traces(selector=dict(type="surface"), visible=True)
    fig.update_layout(showlegend=False)
    for idx, visible in enumerate(_mask("group")):
        fig.data[idx].visible = visible
    return fig


def _html_sem_network(
    sem_coeff_df: pd.DataFrame,
    long_df: Optional[pd.DataFrame] = None,
) -> str:
    """Custom interactive 3D hierarchical SEM network with explicit UI controls."""
    df = sem_coeff_df[sem_coeff_df["op"] == "~"].copy()
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df = df.dropna(subset=["estimate"])
    if df.empty:
        return "<p>No SEM path data available.</p>"

    opinion_lookup: Dict[str, str] = {}
    opinion_group_lookup: Dict[str, str] = {}
    attack_stats = pd.DataFrame()
    if long_df is not None and not long_df.empty:
        opinion_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
        attack_col = "attack_leaf" if "attack_leaf" in long_df.columns else "attack_leaf_label"
        if opinion_col in long_df.columns:
            opinion_values = sorted(long_df[opinion_col].dropna().unique())
            opinion_display = _unique_display_map(opinion_values)
            for opinion_value in opinion_values:
                leaf_name = _pretty_indicator(str(opinion_value))
                opinion_lookup.setdefault(leaf_name, opinion_display[opinion_value])
                opinion_group_lookup.setdefault(leaf_name, _path_context(opinion_value, keep=1) or "Opinion Targets")
        if attack_col in long_df.columns:
            attack_stats = (
                long_df.groupby(attack_col, as_index=False)
                .agg(
                    mean_ae=("adversarial_effectivity", "mean"),
                    mean_abs=("abs_delta_score", "mean"),
                    sd_ae=("adversarial_effectivity", "std"),
                    n_rows=("scenario_id", "count"),
                )
                .rename(columns={attack_col: "attack_label"})
            )

    df["rhs_label"] = df["rhs"].astype(str).map(lambda s: re.sub(r"\s+", " ", str(s)).strip())
    df["lhs_leaf"] = df["lhs"].astype(str).map(_pretty_indicator)
    df["lhs_label"] = df["lhs_leaf"].map(lambda leaf: opinion_lookup.get(leaf, leaf))
    df["mod_root"] = df["rhs_label"].map(lambda s: _infer_sem_moderator_groups(s)[0])
    df["ind_group"] = df["lhs_leaf"].map(lambda s: opinion_group_lookup.get(s, "Opinion Targets"))
    df["mod_role"] = df["mod_root"].map(
        lambda g: "control" if g == "Model Controls" else "demographic" if g == "Demographics" else "profile"
    )

    mod_rank = (
        df.groupby(["mod_root", "mod_role", "rhs_label"], as_index=False)
        .agg(abs_est=("estimate", lambda s: float(np.max(np.abs(s)))))
        .sort_values(["mod_root", "abs_est", "rhs_label"], ascending=[True, False, True])
    )
    ind_rank = (
        df.groupby(["ind_group", "lhs_label"], as_index=False)
        .agg(abs_est=("estimate", lambda s: float(np.max(np.abs(s)))))
        .sort_values(["ind_group", "abs_est", "lhs_label"], ascending=[True, False, True])
    )

    mod_groups = (
        mod_rank.groupby("mod_root", as_index=False)["abs_est"]
        .sum()
        .sort_values("abs_est", ascending=False)["mod_root"]
        .tolist()
    )
    ind_groups = (
        ind_rank.groupby("ind_group", as_index=False)["abs_est"]
        .sum()
        .sort_values("abs_est", ascending=False)["ind_group"]
        .tolist()
    )
    mod_group_members = {
        group: mod_rank.loc[mod_rank["mod_root"] == group, "rhs_label"].tolist()
        for group in mod_groups
    }
    ind_group_members = {
        group: ind_rank.loc[ind_rank["ind_group"] == group, "lhs_label"].tolist()
        for group in ind_groups
    }
    mod_role_lookup = {row["rhs_label"]: row["mod_role"] for _, row in mod_rank.iterrows()}

    def _lane_layout(group_members: Dict[str, List[str]], group_order: List[str]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        n_groups = max(len(group_order), 1)
        z_vals = np.linspace((n_groups - 1) * 2.6 / 2, -(n_groups - 1) * 2.6 / 2, n_groups) if n_groups > 1 else np.array([0.0])
        group_pos: Dict[str, Tuple[float, float]] = {}
        leaf_pos: Dict[str, Tuple[float, float]] = {}
        for z_lane, group in zip(z_vals, group_order):
            labels = group_members.get(group, [])
            if len(labels) <= 1:
                y_vals = np.array([0.0] * max(len(labels), 1))
            else:
                y_vals = np.linspace((len(labels) - 1) * 1.12 / 2, -(len(labels) - 1) * 1.12 / 2, len(labels))
            group_pos[group] = (0.0, float(z_lane))
            for y_val, label in zip(y_vals, labels):
                leaf_pos[label] = (float(y_val), float(z_lane))
        return group_pos, leaf_pos

    def _reorder_within_groups(
        frame: pd.DataFrame,
        src_col: str,
        src_groups: Dict[str, List[str]],
        tgt_col: str,
        tgt_groups: Dict[str, List[str]],
        n_iter: int = 4,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        src_members = {k: list(v) for k, v in src_groups.items()}
        tgt_members = {k: list(v) for k, v in tgt_groups.items()}
        for _ in range(n_iter):
            _, tgt_leaf = _lane_layout(tgt_members, list(tgt_members.keys()))
            src_scores = (
                frame.groupby(src_col)[[tgt_col, "estimate"]]
                .apply(lambda g: float(np.average(
                    [tgt_leaf[val][0] + tgt_leaf[val][1] * 0.15 for val in g[tgt_col]],
                    weights=np.abs(g["estimate"]).to_numpy(),
                )))
                .to_dict()
            )
            for group, labels in src_members.items():
                src_members[group] = sorted(labels, key=lambda label: (src_scores.get(label, 0.0), label))

            _, src_leaf = _lane_layout(src_members, list(src_members.keys()))
            tgt_scores = (
                frame.groupby(tgt_col)[[src_col, "estimate"]]
                .apply(lambda g: float(np.average(
                    [src_leaf[val][0] + src_leaf[val][1] * 0.15 for val in g[src_col]],
                    weights=np.abs(g["estimate"]).to_numpy(),
                )))
                .to_dict()
            )
            for group, labels in tgt_members.items():
                tgt_members[group] = sorted(labels, key=lambda label: (tgt_scores.get(label, 0.0), label))
        return src_members, tgt_members

    mod_group_members, ind_group_members = _reorder_within_groups(
        df,
        "rhs_label",
        mod_group_members,
        "lhs_label",
        ind_group_members,
    )

    mod_group_pos, mod_leaf_pos = _lane_layout(mod_group_members, mod_groups)
    ind_group_pos, ind_leaf_pos = _lane_layout(ind_group_members, ind_groups)

    attack_group_members: Dict[str, List[str]] = {}
    attack_group_pos: Dict[str, Tuple[float, float]] = {}
    attack_leaf_pos: Dict[str, Tuple[float, float]] = {}
    attack_order: List[str] = []
    if not attack_stats.empty:
        attack_stats["attack_group"] = attack_stats["attack_label"].map(lambda s: _path_context(s, keep=1) or "Attack Scope")
        attack_order = (
            attack_stats.groupby("attack_group", as_index=False)["mean_abs"]
            .sum()
            .sort_values("mean_abs", ascending=False)["attack_group"]
            .tolist()
        )
        attack_group_members = {
            group: attack_stats.loc[attack_stats["attack_group"] == group]
            .sort_values("mean_abs", ascending=False)["attack_label"].tolist()
            for group in attack_order
        }
        attack_group_pos, attack_leaf_pos = _lane_layout(attack_group_members, attack_order)

    mod_order = [label for group in mod_groups for label in mod_group_members[group]]
    ind_order = [label for group in ind_groups for label in ind_group_members[group]]
    attack_order_flat = [label for group in attack_order for label in attack_group_members[group]]

    mod_codes = {label: f"M{i+1:02d}" for i, label in enumerate(mod_order)}
    ind_codes = {label: f"O{i+1:02d}" for i, label in enumerate(ind_order)}
    attack_codes = {label: f"A{i+1:02d}" for i, label in enumerate(attack_order_flat)}

    x_mod_group, x_mod_leaf, x_attack_group, x_attack_leaf, x_ind_group, x_ind_leaf = 0.0, 1.0, 2.0, 2.45, 3.7, 4.7

    profile_group_nodes = [
        dict(
            id=f"pg::{group}",
            label=group,
            short=_clip_label(group, 24),
            x=x_mod_group,
            y=mod_group_pos[group][0],
            z=mod_group_pos[group][1],
            family=group,
            role="group",
        )
        for group in mod_groups
    ]
    profile_leaf_nodes = [
        dict(
            id=f"pl::{label}",
            code=mod_codes[label],
            label=label,
            short=_clip_label(label.replace("Big Five ", ""), 24),
            x=x_mod_leaf,
            y=mod_leaf_pos[label][0],
            z=mod_leaf_pos[label][1],
            family=next((g for g in mod_groups if label in mod_group_members[g]), "Other Moderators"),
            role=mod_role_lookup.get(label, "profile"),
        )
        for label in mod_order
    ]
    opinion_group_nodes = [
        dict(
            id=f"og::{group}",
            label=group,
            short=_clip_label(group, 24),
            x=x_ind_group,
            y=ind_group_pos[group][0],
            z=ind_group_pos[group][1],
            family=group,
            role="group",
        )
        for group in ind_groups
    ]
    opinion_leaf_nodes = [
        dict(
            id=f"ol::{label}",
            code=ind_codes[label],
            label=label,
            short=_clip_label(label, 24),
            x=x_ind_leaf,
            y=ind_leaf_pos[label][0],
            z=ind_leaf_pos[label][1],
            family=next((g for g in ind_groups if label in ind_group_members[g]), "Opinion Targets"),
            role="leaf",
        )
        for label in ind_order
    ]
    attack_group_nodes = [
        dict(
            id=f"ag::{group}",
            label=group,
            short=_clip_label(group, 20),
            x=x_attack_group,
            y=attack_group_pos[group][0],
            z=attack_group_pos[group][1],
            family=group,
            role="group",
        )
        for group in attack_order
    ]
    attack_leaf_nodes = []
    if not attack_stats.empty:
        mean_abs_max = float(max(attack_stats["mean_abs"].max(), 0.01))
        for attack_label in attack_order_flat:
            row = attack_stats.loc[attack_stats["attack_label"] == attack_label].iloc[0]
            attack_leaf_nodes.append(dict(
                id=f"al::{attack_label}",
                code=attack_codes[attack_label],
                label=attack_label,
                short=_clip_label(_leaf(attack_label), 22),
                x=x_attack_leaf,
                y=attack_leaf_pos[attack_label][0],
                z=attack_leaf_pos[attack_label][1],
                family=row["attack_group"],
                mean_ae=float(row["mean_ae"]),
                mean_abs=float(row["mean_abs"]),
                sd_ae=float(row["sd_ae"]) if pd.notna(row["sd_ae"]) else 0.0,
                n_rows=int(row["n_rows"]),
                size=float(10 + float(row["mean_abs"]) / mean_abs_max * 14),
            ))

    group_edges = []
    grouped = (
        df.groupby(["mod_root", "ind_group"], as_index=False)
        .agg(
            mean_est=("estimate", "mean"),
            mean_abs=("estimate", lambda s: float(np.mean(np.abs(s)))),
            min_p=("p_value", "min"),
            n_paths=("estimate", "count"),
        )
    )
    for _, row in grouped.iterrows():
        mod_group = str(row["mod_root"])
        ind_group = str(row["ind_group"])
        if mod_group not in mod_group_pos or ind_group not in ind_group_pos:
            continue
        y0, z0 = mod_group_pos[mod_group]
        y1, z1 = ind_group_pos[ind_group]
        group_edges.append(dict(
            id=f"ge::{mod_group}::{ind_group}",
            source_group=mod_group,
            target_group=ind_group,
            source_role="group",
            estimate=float(row["mean_est"]),
            abs_est=float(row["mean_abs"]),
            p=float(row["min_p"]) if pd.notna(row["min_p"]) else 1.0,
            n_paths=int(row["n_paths"]),
            x=[x_mod_group, 1.10, 2.15, 3.00, x_ind_group],
            y=[y0, y0 * 0.58, (y0 + y1) / 2, y1 * 0.58, y1],
            z=[z0, z0, (z0 + z1) / 2, z1, z1],
        ))

    leaf_edges = []
    for _, row in df.iterrows():
        rhs = str(row["rhs_label"])
        lhs = str(row["lhs_label"])
        if rhs not in mod_leaf_pos or lhs not in ind_leaf_pos:
            continue
        y0, z0 = mod_leaf_pos[rhs]
        y1, z1 = ind_leaf_pos[lhs]
        src_group = next((g for g in mod_groups if rhs in mod_group_members[g]), "Other Moderators")
        tgt_group = next((g for g in ind_groups if lhs in ind_group_members[g]), "Opinion Targets")
        leaf_edges.append(dict(
            id=f"le::{rhs}::{lhs}",
            source=rhs,
            target=lhs,
            source_code=mod_codes[rhs],
            target_code=ind_codes[lhs],
            source_group=src_group,
            target_group=tgt_group,
            source_role=mod_role_lookup.get(rhs, "profile"),
            estimate=float(row["estimate"]),
            abs_est=float(abs(row["estimate"])),
            p=float(row["p_value"]) if pd.notna(row["p_value"]) else 1.0,
            x=[x_mod_leaf, 1.75, 2.35, 2.95, x_ind_leaf],
            y=[y0, y0 * 0.62, (y0 + y1) / 2, y1 * 0.62, y1],
            z=[z0, z0, (z0 + z1) / 2, z1, z1],
        ))

    payload = {
        "profile_groups": mod_groups,
        "opinion_groups": ind_groups,
        "profile_group_nodes": profile_group_nodes,
        "profile_leaf_nodes": profile_leaf_nodes,
        "attack_group_nodes": attack_group_nodes,
        "attack_leaf_nodes": attack_leaf_nodes,
        "opinion_group_nodes": opinion_group_nodes,
        "opinion_leaf_nodes": opinion_leaf_nodes,
        "group_edges": group_edges,
        "leaf_edges": leaf_edges,
        "max_abs_beta": float(max(df["estimate"].abs().max(), 0.01)),
        "z_range": [
            float(min(
                [node["z"] for node in profile_group_nodes + opinion_group_nodes + attack_group_nodes] + [-1.8]
            ) - 1.2),
            float(max(
                [node["z"] for node in profile_group_nodes + opinion_group_nodes + attack_group_nodes] + [1.8]
            ) + 1.2),
        ],
        "y_limit": float(max(
            [abs(node["y"]) for node in profile_leaf_nodes + opinion_leaf_nodes + attack_leaf_nodes] + [1.8]
        ) + 0.95),
    }

    profile_filter_html = "".join(
        f"""<label class="semn-chip"><input type="checkbox" class="semn-profile-group" value="{group}" checked> <span>{group}</span></label>"""
        for group in mod_groups
    )
    opinion_filter_html = "".join(
        f"""<label class="semn-chip"><input type="checkbox" class="semn-opinion-group" value="{group}" checked> <span>{group}</span></label>"""
        for group in ind_groups
    )

    return f"""
<div id="semn-root">
  <style>
    #semn-root .semn-shell{{display:grid;grid-template-columns:minmax(290px,330px) minmax(0,1fr);gap:16px;align-items:start}}
    #semn-root .semn-card{{background:#f7faff;border:1px solid #dbe3ef;border-radius:12px;padding:12px 13px;box-shadow:0 3px 14px rgba(20,33,61,0.05)}}
    #semn-root .semn-card + .semn-card{{margin-top:10px}}
    #semn-root .semn-title{{font-weight:800;font-size:0.92rem;color:{PALETTE['navy']};margin-bottom:8px}}
    #semn-root .semn-sub{{font-size:0.75rem;color:{PALETTE['muted']};line-height:1.45;margin-bottom:8px}}
    #semn-root .semn-segment{{display:flex;flex-wrap:wrap;gap:6px}}
    #semn-root .semn-btn{{padding:6px 9px;border-radius:999px;border:1px solid #c8d7ec;background:#fff;color:{PALETTE['ink']};cursor:pointer;font-size:0.75rem;font-weight:700}}
    #semn-root .semn-btn.active{{background:{PALETTE['blue']};border-color:{PALETTE['blue']};color:#fff}}
    #semn-root .semn-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
    #semn-root .semn-grid.one{{grid-template-columns:1fr}}
    #semn-root .semn-row{{display:flex;justify-content:space-between;align-items:center;gap:8px}}
    #semn-root .semn-toggle{{display:flex;align-items:center;gap:7px;font-size:0.76rem;color:{PALETTE['ink']};font-weight:600}}
    #semn-root .semn-select{{width:100%;padding:7px 8px;border-radius:8px;border:1px solid #dbe3ef;background:#fff;color:{PALETTE['ink']};font-size:0.80rem}}
    #semn-root .semn-slider-wrap{{background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:9px 10px}}
    #semn-root .semn-slider-meta{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;font-size:0.76rem;color:{PALETTE['muted']};font-weight:700}}
    #semn-root input[type="range"]{{width:100%;accent-color:{PALETTE['blue']}}}
    #semn-root .semn-quick{{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}}
    #semn-root .semn-chip{{display:flex;align-items:center;gap:7px;padding:6px 8px;border-radius:9px;background:#fff;border:1px solid #dbe3ef;font-size:0.75rem;color:{PALETTE['ink']}}}
    #semn-root .semn-filter-list{{display:flex;flex-wrap:wrap;gap:6px;max-height:140px;overflow:auto;padding-right:3px}}
    #semn-root .semn-stage{{display:flex;flex-direction:column;gap:12px}}
    #semn-root .semn-banner{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;background:linear-gradient(135deg,#f8fbff 0%,#eef5ff 100%);border:1px solid #dbe3ef;border-radius:12px;padding:11px 13px}}
    #semn-root .semn-status{{font-size:0.79rem;color:{PALETTE['muted']};line-height:1.45}}
    #semn-root .semn-status strong{{color:{PALETTE['ink']}}}
    #semn-root .semn-legend{{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}}
    #semn-root .semn-legend-item{{font-size:0.73rem;color:{PALETTE['muted']};display:flex;align-items:center;gap:6px}}
    #semn-root .semn-swatch{{display:inline-block;width:28px;height:8px;border-radius:999px}}
    #semn-root .semn-swatch.pos{{background:linear-gradient(90deg,rgba(255,219,210,0.75),rgba(192,57,43,0.95))}}
    #semn-root .semn-swatch.neg{{background:linear-gradient(90deg,rgba(216,232,255,0.75),rgba(29,78,137,0.95))}}
    #semn-root .semn-swatch.sig{{background:linear-gradient(90deg,rgba(120,120,120,0.20),rgba(120,120,120,0.95))}}
    #semn-root #semn-plot{{background:#fff;border:1px solid #dbe3ef;border-radius:14px;min-height:660px}}
    #semn-root .semn-bottom{{display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:12px}}
    #semn-root .semn-panel{{background:#fff;border:1px solid #dbe3ef;border-radius:12px;padding:12px 13px}}
    #semn-root .semn-panel h4{{margin:0 0 8px;font-size:0.82rem;color:{PALETTE['navy']}}}
    #semn-root .semn-list{{display:flex;flex-direction:column;gap:7px}}
    #semn-root .semn-item{{padding:7px 8px;border-radius:10px;background:#f8fbff;border:1px solid #e0eaf8}}
    #semn-root .semn-item-top{{display:flex;justify-content:space-between;gap:8px;align-items:center;font-size:0.76rem}}
    #semn-root .semn-item-top strong{{color:{PALETTE['ink']}}}
    #semn-root .semn-badge{{display:inline-flex;align-items:center;padding:2px 6px;border-radius:999px;font-size:0.66rem;font-weight:800;letter-spacing:0.02em}}
    #semn-root .semn-badge.hsig{{background:rgba(231,111,81,0.14);color:{PALETTE['red']}}}
    #semn-root .semn-badge.sig{{background:rgba(29,78,137,0.14);color:{PALETTE['blue']}}}
    #semn-root .semn-badge.weak{{background:rgba(20,33,61,0.08);color:{PALETTE['muted']}}}
    #semn-root .semn-item-sub{{font-size:0.72rem;color:{PALETTE['muted']};line-height:1.4;margin-top:3px}}
    #semn-root .semn-map{{display:flex;flex-wrap:wrap;gap:7px}}
    #semn-root .semn-map span{{padding:4px 6px;border-radius:8px;background:#f7faff;border:1px solid #e0eaf8;font-size:0.72rem;color:{PALETTE['ink']}}}
    @media (max-width: 1120px) {{
      #semn-root .semn-shell{{grid-template-columns:1fr}}
      #semn-root .semn-bottom{{grid-template-columns:1fr}}
    }}
  </style>
  <div class="semn-shell">
    <div>
      <div class="semn-card">
        <div class="semn-title">View Presets</div>
        <div class="semn-sub">Start with a simple baseline summary of PROFILE-side moderation families, then progressively add leaf-level SEM paths and attack-context structure.</div>
        <div class="semn-segment" id="semn-view-mode">
          <button class="semn-btn active" data-view="overview">Baseline</button>
          <button class="semn-btn" data-view="leaf">Leaf Detail</button>
          <button class="semn-btn" data-view="all">All Layers</button>
        </div>
      </div>

      <div class="semn-card">
        <div class="semn-title">Path Threshold</div>
        <div class="semn-slider-wrap">
          <div class="semn-slider-meta"><span>Maximum p-value shown</span><span id="semn-p-display">0.10</span></div>
          <input type="range" id="semn-p-slider" min="0" max="100" value="50" step="1">
          <div class="semn-quick">
            <button class="semn-btn" data-p="0.01">0.01</button>
            <button class="semn-btn" data-p="0.05">0.05</button>
            <button class="semn-btn active" data-p="0.10">0.10</button>
            <button class="semn-btn" data-p="1.00">All</button>
          </div>
        </div>
      </div>

      <div class="semn-card">
        <div class="semn-title">Layer Controls</div>
        <div class="semn-sub">Presets set a good default; these switches let you explicitly choose which moderation layers stay visible.</div>
        <div class="semn-grid one">
          <label class="semn-toggle"><input type="checkbox" id="semn-show-group-edges" checked> Show family ribbons</label>
          <label class="semn-toggle"><input type="checkbox" id="semn-show-leaf-edges"> Show leaf-level paths</label>
          <label class="semn-toggle"><input type="checkbox" id="semn-show-group-nodes" checked> Show family nodes</label>
          <label class="semn-toggle"><input type="checkbox" id="semn-show-leaf-nodes"> Show leaf nodes</label>
        </div>
        <div class="semn-row" style="margin-top:10px">
          <label class="semn-toggle"><input type="checkbox" id="semn-include-controls"> Include model controls</label>
          <label class="semn-toggle"><input type="checkbox" id="semn-show-attack" checked> Show attack context</label>
        </div>
      </div>

      <div class="semn-card">
        <div class="semn-title">Path Filters</div>
        <div class="semn-grid">
          <div>
            <div class="semn-sub" style="margin-bottom:6px">Direction</div>
            <div class="semn-segment" id="semn-sign-mode">
              <button class="semn-btn active" data-sign="all">All</button>
              <button class="semn-btn" data-sign="positive">Positive</button>
              <button class="semn-btn" data-sign="negative">Negative</button>
            </div>
          </div>
          <div>
            <div class="semn-sub" style="margin-bottom:6px">Labels</div>
            <select id="semn-label-density" class="semn-select">
              <option value="minimal">Minimal</option>
              <option value="codes" selected>Codes</option>
              <option value="short">Short labels</option>
            </select>
          </div>
        </div>
      </div>

      <div class="semn-card">
        <div class="semn-title">Hierarchy Filters</div>
        <div class="semn-sub">Choose which PROFILE-side families and OPINION-space families remain in the 3D moderation view.</div>
        <div class="semn-sub" style="font-weight:700;color:{PALETTE['ink']};margin-bottom:6px">Profile families</div>
        <div class="semn-filter-list">{profile_filter_html}</div>
        <div class="semn-sub" style="font-weight:700;color:{PALETTE['ink']};margin:10px 0 6px">Opinion families</div>
        <div class="semn-filter-list">{opinion_filter_html}</div>
      </div>

      <div class="semn-card">
        <div class="semn-title">Camera</div>
        <div class="semn-segment" id="semn-camera-mode">
          <button class="semn-btn active" data-camera="perspective">Perspective</button>
          <button class="semn-btn" data-camera="profile">Profile Side</button>
          <button class="semn-btn" data-camera="opinion">Opinion Side</button>
          <button class="semn-btn" data-camera="top">Top Down</button>
        </div>
      </div>
    </div>

    <div class="semn-stage">
      <div class="semn-banner">
        <div class="semn-status" id="semn-status"></div>
        <div class="semn-legend">
          <div class="semn-legend-item"><span class="semn-swatch pos"></span><span>positive moderation of attack effectivity</span></div>
          <div class="semn-legend-item"><span class="semn-swatch neg"></span><span>negative moderation / resistance</span></div>
          <div class="semn-legend-item"><span class="semn-swatch sig"></span><span>stronger opacity = lower p-value</span></div>
          <div class="semn-legend-item"><span style="font-weight:800;color:{PALETTE['ink']}">|β|</span><span>thicker ribbons = stronger moderation</span></div>
        </div>
      </div>
      <div id="semn-plot"></div>
      <div class="semn-bottom">
        <div class="semn-panel">
          <h4>Interpretation</h4>
          <div id="semn-logic" class="semn-sub" style="margin:0"></div>
        </div>
        <div class="semn-panel">
          <h4>Highlighted Paths</h4>
          <div id="semn-focus" class="semn-list"></div>
        </div>
        <div class="semn-panel">
          <h4>Visible Node Map</h4>
          <div id="semn-map" class="semn-map"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
  (function(){{
    const DATA = {json.dumps(payload)};
    const root = document.getElementById('semn-root');
    const plotEl = root.querySelector('#semn-plot');

    const cameras = {{
      perspective: {{eye: {{x: 1.7, y: 1.55, z: 0.95}}}},
      profile:     {{eye: {{x: 0.18, y: 2.35, z: 0.55}}}},
      opinion:     {{eye: {{x: -0.18, y: -2.35, z: 0.55}}}},
      top:         {{eye: {{x: 0.0, y: 0.14, z: 2.7}}}},
    }};

    function sliderToP(val) {{
      return Math.pow(10, -2 + (parseFloat(val) / 50));
    }}
    function pToSlider(p) {{
      const clamped = Math.max(0.01, Math.min(1, p));
      return Math.round((Math.log10(clamped) + 2) * 50);
    }}
    function fmtP(p) {{
      if (p >= 0.995) return 'All';
      if (p < 0.02) return p.toFixed(3);
      if (p < 0.1) return p.toFixed(2);
      return p.toFixed(2);
    }}
    function colorForEdge(est, p, maxAbs) {{
      const t = Math.max(0.12, Math.min(1, Math.abs(est) / Math.max(maxAbs, 0.01)));
      const pos0 = [255, 224, 216], pos1 = [192, 57, 43];
      const neg0 = [218, 232, 255], neg1 = [29, 78, 137];
      const base = est >= 0 ? pos0 : neg0;
      const end  = est >= 0 ? pos1 : neg1;
      const r = Math.round(base[0] + (end[0] - base[0]) * t);
      const g = Math.round(base[1] + (end[1] - base[1]) * t);
      const b = Math.round(base[2] + (end[2] - base[2]) * t);
      let a = 0.16;
      if (p <= 0.01) a = 0.98;
      else if (p <= 0.05) a = 0.82;
      else if (p <= 0.10) a = 0.54;
      return `rgba(${{r}},${{g}},${{b}},${{a}})`;
    }}
    function activeButtonValue(containerId, attr) {{
      const btn = root.querySelector(`#${{containerId}} .semn-btn.active`);
      return btn ? btn.dataset[attr] : null;
    }}
    function selectedValues(selector) {{
      return Array.from(root.querySelectorAll(selector)).filter(el => el.checked).map(el => el.value);
    }}
    function labelText(node, density, mode) {{
      if (!node) return '';
      if (node.role === 'group') return node.short || node.label;
      if (density === 'minimal') return mode === 'leaf' || mode === 'all' ? (node.code || '') : '';
      if (density === 'codes') return node.code || '';
      return node.code ? `${{node.code}}  ${{node.short || node.label}}` : (node.short || node.label);
    }}
    function pBadge(p) {{
      if (p <= 0.01) return ['Highly significant', 'hsig'];
      if (p <= 0.05) return ['Significant', 'sig'];
      return ['Exploratory', 'weak'];
    }}
    function applyPreset(mode) {{
      const groupEdges = root.querySelector('#semn-show-group-edges');
      const leafEdges = root.querySelector('#semn-show-leaf-edges');
      const groupNodes = root.querySelector('#semn-show-group-nodes');
      const leafNodes = root.querySelector('#semn-show-leaf-nodes');
      if (!groupEdges || !leafEdges || !groupNodes || !leafNodes) return;
      if (mode === 'overview') {{
        groupEdges.checked = true;
        leafEdges.checked = false;
        groupNodes.checked = true;
        leafNodes.checked = false;
      }} else if (mode === 'leaf') {{
        groupEdges.checked = false;
        leafEdges.checked = true;
        groupNodes.checked = false;
        leafNodes.checked = true;
      }} else {{
        groupEdges.checked = true;
        leafEdges.checked = true;
        groupNodes.checked = true;
        leafNodes.checked = true;
      }}
    }}
    function syncQuickPButtons(pMax) {{
      root.querySelectorAll('.semn-btn[data-p]').forEach(el => el.classList.remove('active'));
      const matches = [
        ['0.01', 0.01],
        ['0.05', 0.05],
        ['0.10', 0.10],
        ['1.00', 1.00],
      ];
      const found = matches.find(([, v]) => Math.abs(v - pMax) < 1e-3);
      if (!found) return;
      const btn = root.querySelector(`.semn-btn[data-p="${{found[0]}}"]`);
      if (btn) btn.classList.add('active');
    }}
    function state() {{
      return {{
        mode: activeButtonValue('semn-view-mode', 'view') || 'overview',
        sign: activeButtonValue('semn-sign-mode', 'sign') || 'all',
        pMax: sliderToP(root.querySelector('#semn-p-slider').value),
        showGroupEdges: root.querySelector('#semn-show-group-edges').checked,
        showLeafEdges: root.querySelector('#semn-show-leaf-edges').checked,
        showGroupNodes: root.querySelector('#semn-show-group-nodes').checked,
        showLeafNodes: root.querySelector('#semn-show-leaf-nodes').checked,
        includeControls: root.querySelector('#semn-include-controls').checked,
        showAttack: root.querySelector('#semn-show-attack').checked,
        labelDensity: root.querySelector('#semn-label-density').value,
        camera: activeButtonValue('semn-camera-mode', 'camera') || 'perspective',
        profileGroups: selectedValues('.semn-profile-group'),
        opinionGroups: selectedValues('.semn-opinion-group'),
      }};
    }}
    function edgePass(edge, st) {{
      if (edge.p > st.pMax) return false;
      if (st.sign === 'positive' && edge.estimate <= 0) return false;
      if (st.sign === 'negative' && edge.estimate >= 0) return false;
      if (!st.profileGroups.includes(edge.source_group)) return false;
      if (!st.opinionGroups.includes(edge.target_group)) return false;
      if (!st.includeControls && edge.source_role === 'control') return false;
      return true;
    }}
    function nodePass(node, st) {{
      if (!node) return false;
      if (node.family && DATA.profile_groups.includes(node.family)) {{
        if (!st.profileGroups.includes(node.family)) return false;
        if (!st.includeControls && node.role === 'control') return false;
      }}
      if (node.family && DATA.opinion_groups.includes(node.family)) {{
        if (!st.opinionGroups.includes(node.family)) return false;
      }}
      return true;
    }}
    function makeLineTrace(edge, maxAbs, name) {{
      return {{
        type: 'scatter3d',
        mode: 'lines',
        x: edge.x,
        y: edge.y,
        z: edge.z,
        line: {{
          color: colorForEdge(edge.estimate, edge.p, maxAbs),
          width: Math.max(name === 'group' ? 5.5 : 3.0, (Math.abs(edge.estimate) / Math.max(maxAbs, 0.01)) * (name === 'group' ? 13.0 : 9.0)),
        }},
        hovertemplate: name === 'group'
          ? `<b>${{edge.source_group}}</b> → <b>${{edge.target_group}}</b><br>Mean β = ${{edge.estimate.toFixed(3)}}<br>Mean |β| = ${{edge.abs_est.toFixed(3)}}<br>Best p = ${{edge.p.toFixed(4)}}<br>Aggregated paths = ${{edge.n_paths}}<extra>Group summary</extra>`
          : `<b>${{edge.source_code}}</b> ${{edge.source}}<br>→ <b>${{edge.target_code}}</b> ${{edge.target}}<br>β = ${{edge.estimate.toFixed(3)}}<br>p = ${{edge.p.toFixed(4)}}<extra>Leaf path</extra>`,
        showlegend: false,
      }};
    }}
    function makeNodeTrace(nodes, marker, density, mode, hoverTitle) {{
      return {{
        type: 'scatter3d',
        mode: 'markers+text',
        x: nodes.map(n => n.x),
        y: nodes.map(n => n.y),
        z: nodes.map(n => n.z),
        text: nodes.map(n => labelText(n, density, mode)),
        textposition: marker.textposition,
        textfont: marker.textfont,
        marker: marker.marker,
        customdata: nodes.map(n => [n.code || '', n.label, n.family || '', n.role || '']),
        hovertemplate: `<b>%{{customdata[0]}}</b> %{{customdata[1]}}<br>Family: %{{customdata[2]}}<br>Type: %{{customdata[3]}}<extra>${{hoverTitle}}</extra>`,
        showlegend: false,
      }};
    }}
    function makeAttackTrace(nodes, density, mode) {{
      return {{
        type: 'scatter3d',
        mode: 'markers+text',
        x: nodes.map(n => n.x),
        y: nodes.map(n => n.y),
        z: nodes.map(n => n.z),
        text: nodes.map(n => density === 'minimal' ? n.code : (density === 'codes' ? n.code : `${{n.code}}  ${{n.short}}`)),
        textposition: 'top center',
        textfont: {{size: 9, color: '{PALETTE['orange']}'}},
        marker: {{
          size: nodes.map(n => n.size || 10),
          color: nodes.map(n => n.mean_ae),
          colorscale: 'RdBu_r',
          cmid: 0,
          line: {{color: 'white', width: 1.6}},
          opacity: 0.92,
        }},
        customdata: nodes.map(n => [n.code, n.label, n.mean_ae, n.mean_abs, n.sd_ae, n.n_rows]),
        hovertemplate: `<b>%{{customdata[0]}}</b> %{{customdata[1]}}<br>Mean AE = %{{customdata[2]:.2f}}<br>Mean |Δ| = %{{customdata[3]:.2f}}<br>SD AE = %{{customdata[4]:.2f}}<br>Rows = %{{customdata[5]}}<extra>Attack scope</extra>`,
        showlegend: false,
      }};
    }}
    function buildTraces(st) {{
      const traces = [];
      const visibleGroupEdges = DATA.group_edges.filter(edge => edgePass(edge, st));
      const visibleLeafEdges = DATA.leaf_edges.filter(edge => edgePass(edge, st));

      if (st.showAttack) {{
        traces.push({{
          type: 'surface',
          x: [[2.22, 2.22], [2.22, 2.22]],
          y: [[-DATA.y_limit, DATA.y_limit], [-DATA.y_limit, DATA.y_limit]],
          z: [[DATA.z_range[0], DATA.z_range[0]], [DATA.z_range[1], DATA.z_range[1]]],
          showscale: false,
          opacity: 0.12,
          hoverinfo: 'skip',
          colorscale: [[0, 'rgba(240,192,64,0.30)'], [1, 'rgba(231,111,81,0.16)']],
          name: 'Attack scope plane',
        }});
      }}

      if (st.showGroupEdges) visibleGroupEdges.forEach(edge => traces.push(makeLineTrace(edge, DATA.max_abs_beta, 'group')));
      if (st.showLeafEdges) visibleLeafEdges.forEach(edge => traces.push(makeLineTrace(edge, DATA.max_abs_beta, 'leaf')));

      const profileGroupNodes = DATA.profile_group_nodes.filter(n => nodePass(n, st));
      const opinionGroupNodes = DATA.opinion_group_nodes.filter(n => nodePass(n, st));
      const profileLeafNodes = DATA.profile_leaf_nodes.filter(n => nodePass(n, st));
      const opinionLeafNodes = DATA.opinion_leaf_nodes.filter(n => nodePass(n, st));
      const attackGroupNodes = DATA.attack_group_nodes;
      const attackLeafNodes = DATA.attack_leaf_nodes;

      if (st.showGroupNodes) {{
        traces.push(makeNodeTrace(profileGroupNodes, {{
          marker: {{size: 17, color: '#dbe8fb', symbol: 'square', line: {{color: '{PALETTE['navy']}', width: 2}}}},
          textposition: 'middle left',
          textfont: {{size: 10, color: '{PALETTE['navy']}'}},
        }}, 'short', st.mode, 'Profile family'));
        if (st.showAttack && attackGroupNodes.length) {{
          traces.push(makeNodeTrace(attackGroupNodes, {{
            marker: {{size: 15, color: '#f7e4bf', symbol: 'square', line: {{color: '{PALETTE['orange']}', width: 1.8}}}},
            textposition: 'middle center',
            textfont: {{size: 9.5, color: '{PALETTE['orange']}'}},
          }}, 'short', st.mode, 'Attack family'));
        }}
        traces.push(makeNodeTrace(opinionGroupNodes, {{
          marker: {{size: 17, color: '#d8f2ef', symbol: 'square', line: {{color: '{PALETTE['teal']}', width: 2}}}},
          textposition: 'middle right',
          textfont: {{size: 10, color: '{PALETTE['teal']}'}},
        }}, 'short', st.mode, 'Opinion family'));
      }}
      if (st.showLeafNodes) {{
        traces.push(makeNodeTrace(profileLeafNodes, {{
          marker: {{size: 10, color: '{PALETTE['navy']}', symbol: 'circle', line: {{color: 'white', width: 1.5}}}},
          textposition: 'middle left',
          textfont: {{size: 8.5, color: '{PALETTE['ink']}'}},
        }}, st.labelDensity, st.mode, 'Profile moderator'));
        if (st.showAttack && attackLeafNodes.length) traces.push(makeAttackTrace(attackLeafNodes, st.labelDensity, st.mode));
        traces.push(makeNodeTrace(opinionLeafNodes, {{
          marker: {{size: 10, color: '{PALETTE['teal']}', symbol: 'diamond', line: {{color: 'white', width: 1.5}}}},
          textposition: 'middle right',
          textfont: {{size: 8.5, color: '{PALETTE['ink']}'}},
        }}, st.labelDensity, st.mode, 'Opinion indicator'));
      }}
      return {{
        traces,
        visibleGroupEdges,
        visibleLeafEdges,
        profileGroupNodes,
        opinionGroupNodes,
        profileLeafNodes,
        opinionLeafNodes,
        attackGroupNodes,
        attackLeafNodes,
      }};
    }}
    function updateStatus(st, built) {{
      const modeText = st.mode === 'overview' ? 'group summary' : st.mode === 'leaf' ? 'leaf detail' : 'all layers';
      const pText = fmtP(st.pMax);
      const edgeCount = (st.showGroupEdges ? built.visibleGroupEdges.length : 0) + (st.showLeafEdges ? built.visibleLeafEdges.length : 0);
      const controlText = st.includeControls ? 'including model controls' : 'inter-individual differences only';
      const hsig = built.visibleLeafEdges.filter(edge => edge.p <= 0.01).length;
      const sig = built.visibleLeafEdges.filter(edge => edge.p > 0.01 && edge.p <= 0.05).length;
      const exploratory = built.visibleLeafEdges.filter(edge => edge.p > 0.05).length;
      const profileShown = st.showLeafNodes ? built.profileLeafNodes.length : built.profileGroupNodes.length;
      const opinionShown = st.showLeafNodes ? built.opinionLeafNodes.length : built.opinionGroupNodes.length;
      const attackShown = st.showAttack ? (st.showLeafNodes ? built.attackLeafNodes.length : built.attackGroupNodes.length) : 0;
      root.querySelector('#semn-status').innerHTML =
        `<strong>Preset:</strong> ${{modeText}}<br>` +
        `<strong>Visible paths:</strong> ${{edgeCount}} under p ≤ ${{pText}}, ${{st.sign}} sign filter, ${{controlText}}<br>` +
        `<strong>Coverage:</strong> ${{profileShown}} profile-side nodes, ${{opinionShown}} opinion-side nodes` +
        `${{st.showAttack ? `, ${{attackShown}} attack-context nodes` : ''}}<br>` +
        `<strong>Significance mix:</strong> ${{hsig}} highly significant, ${{sig}} significant, ${{exploratory}} exploratory leaf paths<br>` +
        `<strong>Logic:</strong> PROFILE-side moderators shape attacked opinion-shift indicators; attack nodes in the middle show manipulated context for the run, not direct SEM regressors.`;
      root.querySelector('#semn-p-display').textContent = fmtP(st.pMax);
    }}
    function updateLogic(st, built) {{
      const ribbonTxt = st.showGroupEdges ? `${{built.visibleGroupEdges.length}} family ribbons` : 'no family ribbons';
      const leafTxt = st.showLeafEdges ? `${{built.visibleLeafEdges.length}} leaf-level SEM paths` : 'no leaf-level paths';
      const txt = st.mode === 'overview'
        ? `Baseline keeps the moderation story readable first: which PROFILE-side families shape which parts of opinion space under cybermanipulation.`
        : st.mode === 'leaf'
        ? `Leaf detail exposes each SEM coefficient individually. M## codes are profile-side moderators; O## codes are opinion indicators; A## nodes summarize attack context for the run.`
        : `All layers overlays family structure, leaf-level coefficients, and attack context so you can compare the coarse moderation map against exact SEM paths.`;
      root.querySelector('#semn-logic').textContent = `${{txt}} Current selection shows ${{ribbonTxt}} and ${{leafTxt}}.`;
    }}
    function updateFocus(st, visibleGroupEdges, visibleLeafEdges) {{
      const items = []
        .concat(st.showGroupEdges ? visibleGroupEdges.map(item => Object.assign({{_kind: 'group'}}, item)) : [])
        .concat(st.showLeafEdges ? visibleLeafEdges.map(item => Object.assign({{_kind: 'leaf'}}, item)) : [])
        .sort((a,b) => (a.p - b.p) || (Math.abs(b.estimate) - Math.abs(a.estimate)))
        .slice(0, 8);
      const wrap = root.querySelector('#semn-focus');
      if (!items.length) {{
        wrap.innerHTML = `<div class="semn-sub" style="margin:0">No paths remain under the current filters.</div>`;
        return;
      }}
      wrap.innerHTML = items.map(item => {{
        const badge = pBadge(item.p);
        const title = item._kind === 'group'
          ? `${{item.source_group}} → ${{item.target_group}}`
          : `${{item.source_code}} → ${{item.target_code}}`;
        const sub = item._kind === 'group'
          ? `${{item.n_paths}} constituent paths · mean β = ${{item.estimate.toFixed(3)}} · best p = ${{item.p.toFixed(4)}}`
          : `${{item.source}} → ${{item.target}} · β = ${{item.estimate.toFixed(3)}} · p = ${{item.p.toFixed(4)}}`;
        return `
          <div class="semn-item">
            <div class="semn-item-top">
              <strong>${{title}}</strong>
              <span class="semn-badge ${{badge[1]}}">${{badge[0]}}</span>
            </div>
            <div class="semn-item-sub">${{sub}}</div>
          </div>`;
      }}).join('');
    }}
    function updateNodeMap(st, profileLeafNodes, opinionLeafNodes, attackLeafNodes) {{
      const nodes = [];
      if (profileLeafNodes.length) nodes.push('<span><strong>Profile moderators</strong></span>');
      profileLeafNodes.forEach(n => nodes.push(`<span><strong>${{n.code}}</strong> ${{n.short}}</span>`));
      if (opinionLeafNodes.length) nodes.push('<span><strong>Opinion indicators</strong></span>');
      opinionLeafNodes.forEach(n => nodes.push(`<span><strong>${{n.code}}</strong> ${{n.short}}</span>`));
      if (st.showAttack && attackLeafNodes.length) nodes.push('<span><strong>Attack context</strong></span>');
      if (st.showAttack) attackLeafNodes.forEach(n => nodes.push(`<span><strong>${{n.code}}</strong> ${{n.short}}</span>`));
      root.querySelector('#semn-map').innerHTML = nodes.join('');
    }}
    function render() {{
      const st = state();
      const built = buildTraces(st);
      const layout = {{
        paper_bgcolor: 'white',
        margin: {{l: 0, r: 0, t: 16, b: 0}},
        font: {{family: 'IBM Plex Sans, Avenir Next, Segoe UI, sans-serif'}},
        scene: {{
          xaxis: {{visible: false, range: [-0.35, 5.05]}},
          yaxis: {{visible: false, range: [-DATA.y_limit, DATA.y_limit]}},
          zaxis: {{visible: false, range: DATA.z_range}},
          aspectmode: 'manual',
          aspectratio: {{x: 2.45, y: 1.55, z: 1.1}},
          bgcolor: 'white',
          camera: cameras[st.camera],
        }},
        showlegend: false,
      }};
      Plotly.react(plotEl, built.traces, layout, {{displayModeBar: false, responsive: true}});
      syncQuickPButtons(st.pMax);
      updateStatus(st, built);
      updateLogic(st, built);
      updateFocus(st, built.visibleGroupEdges, built.visibleLeafEdges);
      updateNodeMap(st, built.profileLeafNodes, built.opinionLeafNodes, built.attackLeafNodes);
    }}

    root.querySelectorAll('.semn-btn[data-view]').forEach(btn => btn.addEventListener('click', () => {{
      root.querySelectorAll('#semn-view-mode .semn-btn').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      applyPreset(btn.dataset.view);
      render();
    }}));
    root.querySelectorAll('.semn-btn[data-sign]').forEach(btn => btn.addEventListener('click', () => {{
      root.querySelectorAll('#semn-sign-mode .semn-btn').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      render();
    }}));
    root.querySelectorAll('.semn-btn[data-camera]').forEach(btn => btn.addEventListener('click', () => {{
      root.querySelectorAll('#semn-camera-mode .semn-btn').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      render();
    }}));
    root.querySelector('#semn-p-slider').addEventListener('input', render);
    root.querySelectorAll('.semn-btn[data-p]').forEach(btn => btn.addEventListener('click', () => {{
      root.querySelectorAll('.semn-btn[data-p]').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      root.querySelector('#semn-p-slider').value = pToSlider(parseFloat(btn.dataset.p));
      render();
    }}));
    root.querySelectorAll('#semn-show-group-edges,#semn-show-leaf-edges,#semn-show-group-nodes,#semn-show-leaf-nodes,#semn-include-controls,#semn-show-attack,#semn-label-density,.semn-profile-group,.semn-opinion-group')
      .forEach(el => el.addEventListener('change', render));

    const parentPanel = root.closest('.tab-panel');
    if (parentPanel) {{
      const obs = new MutationObserver(() => {{
        if (parentPanel.classList.contains('active')) {{
          setTimeout(() => Plotly.Plots.resize(plotEl), 40);
        }}
      }});
      obs.observe(parentPanel, {{attributes: true, attributeFilter: ['class']}});
    }}
    window.addEventListener('resize', () => Plotly.Plots.resize(plotEl));
    applyPreset('overview');
    render();
  }})();
  </script>
</div>"""


def _fig_sem_heatmap(
    sem_coeff_df: pd.DataFrame,
    exploratory_df: pd.DataFrame,
    long_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    df = sem_coeff_df[sem_coeff_df["op"] == "~"].copy()
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["p_value"]  = pd.to_numeric(df["p_value"],  errors="coerce")
    indicators = df["lhs"].unique().tolist()
    hm = df.pivot_table(index="rhs", columns="lhs", values="estimate", aggfunc="mean")

    if not exploratory_df.empty and "normalized_weight_pct" in exploratory_df.columns:
        order = exploratory_df.sort_values("normalized_weight_pct", ascending=False)["moderator_label"].tolist()
        hm = hm.reindex([r for r in order if r in hm.index])

    hm = hm[[c for c in indicators if c in hm.columns]]

    indicator_labels: Dict[str, str] = {col: _pretty_indicator(str(col)) for col in hm.columns}
    if long_df is not None and not long_df.empty:
        opinion_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
        if opinion_col in long_df.columns:
            opinion_values = sorted(long_df[opinion_col].dropna().unique())
            opinion_display = _unique_display_map(opinion_values)
            for value in opinion_values:
                indicator_labels[_pretty_indicator(str(value))] = opinion_display[value]
    col_labels = [_wrap_label(indicator_labels.get(_pretty_indicator(str(c)), _pretty_indicator(str(c))), 18) for c in hm.columns]

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
    op_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    ae_col, abs_col = "adversarial_effectivity", "abs_delta_score"
    if op_col not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    opinions = sorted(long_df[op_col].dropna().unique())
    opinion_labels = _unique_display_map(opinions)
    colors   = px.colors.qualitative.Bold[:max(len(opinions), 4)]

    fig = make_subplots(1, 2,
                        subplot_titles=["Adversarial Effectivity (AE)",
                                        "Absolute Opinion Shift |Δ|"],
                        horizontal_spacing=0.09)

    for i, (op, clr) in enumerate(zip(opinions, colors)):
        sub = long_df[long_df[op_col] == op]
        lbl = opinion_labels[op]

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


def _fig_raw_attack_comparison(long_df: pd.DataFrame) -> go.Figure:
    """Box + strip: AE and |Δ| grouped by attack vector, color = opinion leaf."""
    atk_col = "attack_leaf" if "attack_leaf" in long_df.columns else "attack_leaf_label"
    op_col  = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    ae_col, abs_col = "adversarial_effectivity", "abs_delta_score"
    for c in (atk_col, op_col, ae_col):
        if c not in long_df.columns:
            return go.Figure().add_annotation(text=f"Column '{c}' missing", showarrow=False)

    attacks  = sorted(long_df[atk_col].dropna().unique())
    opinions = sorted(long_df[op_col].dropna().unique())
    atk_labels = _unique_display_map(attacks)
    op_labels  = _unique_display_map(opinions)
    palette    = px.colors.qualitative.Bold

    fig = make_subplots(1, 2,
        subplot_titles=["Adversarial Effectivity by Attack Vector",
                        "Absolute Opinion Shift |Δ| by Attack Vector"],
        horizontal_spacing=0.14)

    for oi, op in enumerate(opinions):
        sub = long_df[long_df[op_col] == op]
        clr = palette[oi % len(palette)]
        lbl = op_labels[op]
        for ci, ycol in enumerate([ae_col, abs_col], 1):
            if ycol not in sub.columns:
                continue
            x_vals = [_wrap_label(atk_labels[a], 22) for a in attacks]
            y_boxes = [sub[sub[atk_col] == a][ycol].dropna().values for a in attacks]
            for xi, (xv, yv) in enumerate(zip(x_vals, y_boxes)):
                if len(yv) == 0:
                    continue
                fig.add_trace(go.Box(
                    x=[xv] * len(yv), y=yv,
                    name=lbl, legendgroup=lbl,
                    showlegend=(ci == 1 and xi == 0),
                    marker_color=clr, line_color=clr,
                    boxmean="sd", whiskerwidth=0.5,
                    width=0.14, opacity=0.78,
                    hovertemplate=f"<b>{lbl}</b><br>%{{x}}<br>{ycol}: %{{y:.1f}}<extra></extra>",
                ), row=1, col=ci)

    # mean AE per attack vector as diamond markers
    atk_means = long_df.groupby(atk_col)[ae_col].mean()
    fig.add_trace(go.Scatter(
        x=[_wrap_label(atk_labels[a], 22) for a in attacks],
        y=[atk_means.get(a, 0) for a in attacks],
        mode="markers", marker=dict(symbol="diamond", size=11,
            color=PALETTE["navy"], line=dict(color="white", width=1.2)),
        name="Overall mean AE", showlegend=True,
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1, row=1, col=1)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=540, boxmode="group",
        title=dict(text="Raw Outcome Distributions by Attack Vector", font_size=14),
        margin=dict(l=65, r=30, t=55, b=110),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font_size=9),
    )
    fig.update_xaxes(tickangle=-20, tickfont_size=9)
    return fig


def _fig_raw_score_scatter(long_df: pd.DataFrame) -> go.Figure:
    """Baseline vs. post scatter with regression trend and AE density marginals."""
    if not {"pre_score", "post_score"}.issubset(long_df.columns):
        # Try alternate column names
        pre_col  = next((c for c in long_df.columns if "baseline" in c.lower() or "pre_score" in c.lower()), None)
        post_col = next((c for c in long_df.columns if "post_score" in c.lower() or "attacked" in c.lower()), None)
        if not pre_col or not post_col:
            return go.Figure().add_annotation(text="Pre/post score columns unavailable", showarrow=False)
    else:
        pre_col, post_col = "pre_score", "post_score"

    ae_col = "adversarial_effectivity"
    op_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    df = long_df[[pre_col, post_col, op_col]].dropna()
    if ae_col in long_df.columns:
        df = long_df[[pre_col, post_col, op_col, ae_col]].dropna()

    opinions = sorted(df[op_col].dropna().unique())
    op_labels = _unique_display_map(opinions)
    palette   = px.colors.qualitative.Bold

    fig = go.Figure()
    # Identity line
    rng = [float(df[pre_col].min() - 2), float(df[pre_col].max() + 2)]
    fig.add_trace(go.Scatter(
        x=rng, y=rng, mode="lines",
        line=dict(color="#aab4c8", width=1.5, dash="dot"),
        name="No change (y = x)", showlegend=True,
    ))

    for oi, op in enumerate(opinions):
        sub = df[df[op_col] == op]
        clr = palette[oi % len(palette)]
        lbl = op_labels[op]
        color_vals = sub[ae_col].values if ae_col in sub.columns else None

        scatter_kw = dict(
            x=sub[pre_col].values, y=sub[post_col].values,
            mode="markers",
            marker=dict(size=5, opacity=0.55, line=dict(color="white", width=0.3),
                        color=(color_vals if color_vals is not None else clr),
                        colorscale="RdBu_r" if color_vals is not None else None,
                        cmid=0 if color_vals is not None else None,
                        showscale=False),
            name=lbl, legendgroup=lbl, showlegend=True,
            hovertemplate=f"<b>{lbl}</b><br>Pre: %{{x:.1f}}<br>Post: %{{y:.1f}}<extra></extra>",
        )
        fig.add_trace(go.Scatter(**scatter_kw))

        # Regression line per opinion
        if len(sub) >= 4:
            x_np = sub[pre_col].values
            y_np = sub[post_col].values
            try:
                m, b_intercept = np.polyfit(x_np, y_np, 1)
                xs = np.array([x_np.min(), x_np.max()])
                fig.add_trace(go.Scatter(
                    x=xs, y=m * xs + b_intercept, mode="lines",
                    line=dict(color=clr, width=1.8, dash="solid"),
                    name=f"{lbl} trend", legendgroup=lbl, showlegend=False,
                ))
            except Exception:
                pass

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=540,
        xaxis_title="Pre-attack Opinion Score",
        yaxis_title="Post-attack Opinion Score",
        title=dict(text="Pre vs. Post Attack Scores — Colour = AE direction", font_size=14),
        margin=dict(l=70, r=30, t=55, b=65),
        legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center", font_size=9),
    )
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
    if "estimate" in df.columns:
        df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    if "mean_abs_estimate" in df.columns:
        df["mean_abs_estimate"] = pd.to_numeric(df["mean_abs_estimate"], errors="coerce")
    df_pos = df[df["normalized_weight_pct"] > 0].copy()
    if df_pos.empty:
        return go.Figure().add_annotation(text="All weights zero", showarrow=False)

    path_rows: List[Tuple[List[str], float]] = []
    for _, row in df_pos.iterrows():
        label = str(row.get("moderator_label") or row.get("term") or "Unnamed moderator")
        ontology_group = None if pd.isna(row.get("ontology_group")) else str(row.get("ontology_group"))
        path_rows.append((_moderator_hierarchy(label, ontology_group), float(row["normalized_weight_pct"])))

    ids, labels, parents, values, paths = _build_tree_nodes(path_rows)

    top_n = min(18, len(df_pos))
    df_top = df_pos.sort_values("normalized_weight_pct", ascending=False).head(top_n).iloc[::-1]
    estimate_col = "estimate" if "estimate" in df_top.columns else "weighted_mean_estimate"
    if estimate_col in df_top.columns:
        df_top[estimate_col] = pd.to_numeric(df_top[estimate_col], errors="coerce").fillna(0)
    else:
        df_top[estimate_col] = 0.0
    df_top["leaf_label"] = [
        _clip_label(_moderator_hierarchy(
            str(row.get("moderator_label") or row.get("term") or "Unnamed moderator"),
            None if pd.isna(row.get("ontology_group")) else str(row.get("ontology_group")),
        )[-1], 34)
        for _, row in df_top.iterrows()
    ]

    fig = make_subplots(
        1,
        2,
        column_widths=[0.40, 0.60],
        horizontal_spacing=0.20,
        subplot_titles=["Ontology hierarchy", f"Top leaf moderators ({top_n} of {len(df_pos)})"],
        specs=[[{"type": "treemap"}, {"type": "xy"}]],
    )

    fig.add_trace(go.Treemap(
        ids=ids,
        labels=[_wrap_label(label, 16) for label in labels],
        parents=parents,
        values=values,
        branchvalues="total",
        textinfo="label+value",
        marker=dict(
            colors=values,
            colorscale="Blues",
            line=dict(width=1.6, color="white"),
        ),
        tiling=dict(pad=4),
        customdata=np.array(paths, dtype=object),
        hovertemplate="%{customdata}<br>Weight: %{value:.1f}%<extra></extra>",
        root_color="#eef4ff",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df_top["normalized_weight_pct"],
        y=df_top["leaf_label"],
        orientation="h",
        marker=dict(
            color=df_top[estimate_col],
            colorscale="RdBu_r",
            cmid=0,
            line=dict(color="white", width=0.6),
            colorbar=dict(title="Signed β", thickness=12),
        ),
        text=[f"{v:.1f}%" for v in df_top["normalized_weight_pct"]],
        textposition="outside",
        cliponaxis=False,
        customdata=np.column_stack([
            df_top["moderator_label"].astype(str),
            df_top[estimate_col].astype(float),
            df_top["normalized_weight_pct"].astype(float),
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Weight: %{customdata[2]:.1f}%<br>"
            "Signed β: %{customdata[1]:.3f}<extra></extra>"
        ),
    ), row=1, col=2)

    fig.update_xaxes(
        title="Normalized importance share (%)",
        row=1,
        col=2,
        automargin=True,
    )
    fig.update_yaxes(
        tickfont=dict(size=9.2),
        row=1,
        col=2,
        automargin=True,
    )
    fig.update_annotations(font=dict(size=12.5))
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=max(620, 30 * top_n + 220),
        showlegend=False,
        title=dict(text="Hierarchical Feature Importance — Conditional Susceptibility Model", font_size=14),
        margin=dict(l=50, r=110, t=78, b=54),
        bargap=0.22,
    )
    return fig


def _fig_profile_heatmap(long_df: pd.DataFrame, profile_index_df: pd.DataFrame) -> go.Figure:
    op_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    ae_col = "adversarial_effectivity"
    if op_col not in long_df.columns or ae_col not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    matrix = long_df.pivot_table(index="profile_id", columns=op_col,
                                  values=ae_col, aggfunc="mean")
    if not profile_index_df.empty and "susceptibility_index_pct" in profile_index_df.columns:
        order = (profile_index_df.sort_values("susceptibility_index_pct", ascending=False)
                 ["profile_id"].tolist())
        matrix = matrix.reindex([p for p in order if p in matrix.index])

    opinion_labels = _unique_display_map(list(matrix.columns))
    col_labels = [_wrap_label(opinion_labels[c], 18) for c in matrix.columns]
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
    opinion_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    has_leaf = opinion_col in long_df.columns
    opinion_text = None
    if has_leaf:
        vals = long_df[opinion_col].astype(str).tolist()
        display_map = _unique_display_map(sorted(set(vals)))
        opinion_text = [display_map.get(v, _leaf(v)) for v in vals]

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
        text=opinion_text if has_leaf else None,
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

    attack_values = sorted(task_coeff_df["attack_leaf"].dropna().unique())
    opinion_values = sorted(task_coeff_df["opinion_leaf"].dropna().unique())
    attack_labels = _unique_display_map(attack_values)
    opinion_labels = _unique_display_map(opinion_values)

    tasks_json: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (ak, ok), grp in task_coeff_df.groupby(["attack_leaf", "opinion_leaf"]):
        tasks_json.setdefault(str(ak), {})[str(ok)] = dict(
            zip(grp["term"].tolist(), grp["estimate"].astype(float).tolist())
        )

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
const ATTACKS={json.dumps(attack_values)};
const OPINIONS={json.dumps(opinion_values)};
const ATTACK_LABELS={json.dumps(attack_labels)};
const OPINION_LABELS={json.dumps(opinion_labels)};
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
  const v=vals();
  let h='<table style="border-collapse:collapse;font-size:0.82rem">';
  h+='<tr><th style="padding:5px 8px;border-bottom:2px solid #dbe3ef;text-align:left;font-size:0.76rem">Attack \\ Opinion</th>';
  OPINIONS.forEach(o=>h+=`<th title="${{o}}" style="padding:5px 7px;border-bottom:2px solid #dbe3ef;font-size:0.76rem;min-width:80px">${{OPINION_LABELS[o]}}</th>`);
  h+='</tr>';
  ATTACKS.forEach(a=>{{
    h+=`<tr><td title="${{a}}" style="padding:5px 8px;font-weight:600;border-right:1px solid #dbe3ef;font-size:0.78rem;white-space:nowrap">${{ATTACK_LABELS[a]}}</td>`;
    OPINIONS.forEach(o=>{{
      const c=(TASKS[a]||{{}})[o]||{{}};
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

    feat_cols = sorted([
        c for c in long_df.columns
        if (c.startswith("profile_cont_") or c.startswith("profile_cat__"))
        and c != "profile_cont_heuristic_shift_sensitivity_proxy"
    ])

    profile_feats_df = long_df.groupby("profile_id")[feat_cols].first().reset_index()
    feat_means = {c: float(profile_feats_df[c].mean()) for c in feat_cols}

    profiles_json: Dict[str, Dict[str, float]] = {}
    for _, row in profile_feats_df.iterrows():
        profiles_json[str(row["profile_id"])] = {
            c: float(row[c]) for c in feat_cols if not pd.isna(row[c])
        }

    all_attacks = sorted(task_coeff_df["attack_leaf"].dropna().unique())
    all_opinions = sorted(task_coeff_df["opinion_leaf"].dropna().unique())
    attack_labels = _unique_display_map(all_attacks)
    opinion_labels = _unique_display_map(all_opinions)
    attack_context = {value: _path_context(value, keep=1) for value in all_attacks}
    opinion_context = {value: _path_context(value, keep=1) for value in all_opinions}

    tasks_json: Dict[str, Dict[str, Dict[str, float]]] = {}
    weights_json: Dict[str, Dict[str, float]] = {}
    for (ak, ok), grp in task_coeff_df.groupby(["attack_leaf", "opinion_leaf"]):
        tasks_json.setdefault(str(ak), {})[str(ok)] = dict(
            zip(grp["term"].tolist(), grp["estimate"].astype(float).tolist())
        )
        weights_json.setdefault(str(ak), {})[str(ok)] = 1.0

    if not task_summary_df.empty:
        for _, row in task_summary_df.iterrows():
            ak = str(row["attack_leaf"])
            ok = str(row["opinion_leaf"])
            weights_json.setdefault(ak, {})[ok] = float(row.get("reliability_weight", 1.0))

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
<div id="cse-root" style="display:grid;grid-template-columns:minmax(295px,330px) minmax(0,1fr);grid-template-rows:auto;gap:16px;align-items:start">

<!-- ══ LEFT: profile builder ══ -->
<div style="display:flex;flex-direction:column;gap:10px">

  <div style="background:#f0f5ff;border-radius:10px;padding:12px 14px">
    <div style="font-weight:700;font-size:0.92rem;color:{PALETTE['navy']};margin-bottom:10px">
      👤 Profile Configuration
    </div>
    <div style="font-size:0.76rem;line-height:1.45;color:{PALETTE['muted']};margin:-2px 0 10px">
      Build a synthetic profile manually or load a random observed profile. The score updates against the currently selected conditional task scope.
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
      <button onclick="cse_reset()" style="flex:1;padding:7px;background:{PALETTE['blue']};color:#fff;border:none;border-radius:7px;cursor:pointer;font-size:0.80rem;font-weight:600">Reset to mean</button>
      <button onclick="cse_random()" style="flex:1;padding:7px;background:{PALETTE['teal']};color:#fff;border:none;border-radius:7px;cursor:pointer;font-size:0.80rem;font-weight:600">Load random observed</button>
    </div>
  </div>

  <!-- Task selector -->
  <div style="background:#f0f5ff;border-radius:10px;padding:12px 14px">
    <div style="font-weight:700;font-size:0.92rem;color:{PALETTE['navy']};margin-bottom:8px">
      🎯 Conditional Task Scope
    </div>
    <div style="font-size:0.76rem;line-height:1.45;color:{PALETTE['muted']};margin:-1px 0 10px">
      Only the selected attack × opinion tasks contribute to the Conditional Susceptibility Score.
    </div>
    <div id="cse-task-summary" style="display:inline-flex;align-items:center;gap:8px;padding:5px 9px;border-radius:999px;background:rgba(29,78,137,0.08);color:{PALETTE['blue']};font-size:0.74rem;font-weight:700;margin-bottom:12px"></div>

    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
      <div style="font-size:0.78rem;font-weight:700;color:{PALETTE['muted']}">Attack vectors <span id="cse-atk-count"></span></div>
      <div style="display:flex;gap:5px">
        <button type="button" onclick="cse_all_atk(true)" style="padding:4px 8px;border-radius:999px;border:1px solid #bdd0ea;background:#fff;color:{PALETTE['blue']};cursor:pointer;font-size:0.72rem;font-weight:700">Select all</button>
        <button type="button" onclick="cse_all_atk(false)" style="padding:4px 8px;border-radius:999px;border:1px solid #ead2cc;background:#fff;color:{PALETTE['orange']};cursor:pointer;font-size:0.72rem;font-weight:700">Clear</button>
      </div>
    </div>
    <div id="cse-atk-checks" style="display:flex;flex-direction:column;gap:5px;margin-bottom:12px;max-height:160px;overflow:auto;padding-right:3px">
      {''.join(f"""<label style="font-size:0.82rem;display:flex;align-items:flex-start;gap:8px;cursor:pointer;padding:6px 7px;border-radius:8px;background:rgba(255,255,255,0.55)">
        <input type="checkbox" checked id="cse-atk-{i}" onchange="cse_update()" style="accent-color:{PALETTE['blue']};margin-top:2px">
        <span style="display:flex;flex-direction:column;gap:1px">
          <span title="{atk}" style="font-weight:600;color:{PALETTE['ink']}">{attack_labels[atk]}</span>
          <span style="font-size:0.71rem;color:{PALETTE['muted']}">{attack_context[atk] or "Attack family"}</span>
        </span></label>""" for i, atk in enumerate(all_attacks))}
    </div>

    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
      <div style="font-size:0.78rem;font-weight:700;color:{PALETTE['muted']}">Opinion targets <span id="cse-op-count"></span></div>
      <div style="display:flex;gap:5px">
        <button type="button" onclick="cse_all_op(true)" style="padding:4px 8px;border-radius:999px;border:1px solid #bdd0ea;background:#fff;color:{PALETTE['blue']};cursor:pointer;font-size:0.72rem;font-weight:700">Select all</button>
        <button type="button" onclick="cse_all_op(false)" style="padding:4px 8px;border-radius:999px;border:1px solid #ead2cc;background:#fff;color:{PALETTE['orange']};cursor:pointer;font-size:0.72rem;font-weight:700">Clear</button>
      </div>
    </div>
    <div id="cse-op-checks" style="display:flex;flex-direction:column;gap:5px;max-height:170px;overflow:auto;padding-right:3px">
      {''.join(f"""<label style="font-size:0.82rem;display:flex;align-items:flex-start;gap:8px;cursor:pointer;padding:6px 7px;border-radius:8px;background:rgba(255,255,255,0.55)">
        <input type="checkbox" checked id="cse-op-{i}" onchange="cse_update()" style="accent-color:{PALETTE['blue']};margin-top:2px">
        <span style="display:flex;flex-direction:column;gap:1px">
          <span title="{op}" style="font-weight:600;color:{PALETTE['ink']}">{opinion_labels[op]}</span>
          <span style="font-size:0.71rem;color:{PALETTE['muted']}">{opinion_context[op] or "Opinion family"}</span>
        </span></label>""" for i, op in enumerate(all_opinions))}
    </div>

    <div style="font-size:0.72rem;line-height:1.45;color:{PALETTE['muted']};margin-top:8px">
      This selector defines the condition under which the score is estimated. Unselected tasks are excluded rather than down-weighted.
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
    <div id="cse-grid" style="overflow:auto"></div>
  </div>

  <!-- Gauge + Radar side by side -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">

    <div style="background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:14px">
      <div style="font-weight:700;font-size:0.88rem;color:{PALETTE['navy']};margin-bottom:8px">
        🎯 Conditional Susceptibility Score
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
const ATTACK_LABELS = {json.dumps(attack_labels)};
const OPINION_LABELS = {json.dumps(opinion_labels)};
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
  const pairs = [];
  atks.forEach(a => {{
    ops.forEach(o => {{
      if ((TASKS[a]||{{}})[o]) pairs.push([a,o]);
    }});
  }});
  return pairs;
}}

function predictAE(pf, attackKey, opinionKey) {{
  const c = ((TASKS[attackKey] || {{}})[opinionKey]) || {{}};
  let ae = 0;
  Object.entries(c).forEach(([t,e]) => ae += e * (pf[t] ?? 0));
  return ae;
}}

function computeScore(pf, selectedPairs) {{
  if (selectedPairs.length === 0) return {{ae_map:{{}}, raw:0, pct:50, dist:[]}};
  const ae_map = {{}};
  let wsum = 0, wtot = 0;
  selectedPairs.forEach(([a,o]) => {{
    const ae = predictAE(pf, a, o);
    if (!ae_map[a]) ae_map[a] = {{}};
    ae_map[a][o] = ae;
    const w = ((WEIGHTS[a] || {{}})[o]) || 1;
    wsum += ae * w; wtot += w;
  }});
  const raw = wtot > 0 ? wsum / wtot : 0;

  // distribution: re-score all 100 original profiles on selected tasks
  const dist = Object.values(PROFILES).map(pfOrig => {{
    let ws=0, wt=0;
    selectedPairs.forEach(([a,o]) => {{
      const w = ((WEIGHTS[a] || {{}})[o]) || 1;
      ws += predictAE(pfOrig, a, o) * w;
      wt += w;
    }});
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

function renderTaskScope(selectedPairs) {{
  const selAtks = ALL_ATTACKS.filter((_,i) => document.getElementById('cse-atk-'+i)?.checked);
  const selOps  = ALL_OPINIONS.filter((_,i) => document.getElementById('cse-op-'+i)?.checked);
  document.getElementById('cse-atk-count').textContent = `(${{
    selAtks.length
  }}/${{ALL_ATTACKS.length}})`;
  document.getElementById('cse-op-count').textContent = `(${{
    selOps.length
  }}/${{ALL_OPINIONS.length}})`;
  document.getElementById('cse-task-summary').textContent =
    `Conditional score uses ${{selectedPairs.length}} / ${{Object.values(TASKS).reduce((n, obj) => n + Object.keys(obj).length, 0)}} configured tasks`;
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
  selOps.forEach(o=>h+=`<th title="${{o}}" style="padding:6px 8px;border-bottom:2px solid #dbe3ef;font-size:0.75rem;color:{PALETTE['muted']};min-width:90px">${{OPINION_LABELS[o]}}</th>`);
  h+='</tr>';
  selAtks.forEach(a=>{{
    h+=`<tr><td title="${{a}}" style="padding:7px 10px;font-weight:600;border-right:1px solid #dbe3ef;white-space:nowrap;font-size:0.80rem;color:{PALETTE['ink']};min-width:180px">${{ATTACK_LABELS[a]}}</td>`;
    selOps.forEach(o=>{{
      const ae = ((ae_map[a] || {{}})[o]) ?? 0;
      const bg=aeColor(ae), tc=Math.abs(ae)>25?'#fff':'{PALETTE['ink']}';
      const lbl = ae>5?'↑ succ':ae<-5?'↓ back':'≈ neut';
      h+=`<td title="${{a}} | ${{o}}" style="text-align:center;padding:8px 6px;background:${{bg}};color:${{tc}};border:2px solid rgba(255,255,255,0.35);border-radius:5px;font-weight:700;font-size:0.92rem">
        ${{ae.toFixed(1)}}<br><span style="font-size:0.65rem;opacity:0.85">${{lbl}}</span></td>`;
    }});
    h+='</tr>';
  }});
  document.getElementById('cse-grid').innerHTML=h+'</table>';
}}

function renderGauge(pct, raw, nProfiles) {{
  const gc = pct<33?'{PALETTE['teal']}':pct<67?'{PALETTE['amber']}':'{PALETTE['red']}';
  const label = pct>=75?'High conditional susceptibility':pct>=50?'Moderately high':pct>=25?'Moderately low':'Low conditional susceptibility';
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
    `Conditional score ranks at the ${{pct}}th percentile vs ${{nProfiles}} original profiles under the selected task scope`;
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

function renderContrib(pf, selectedPairs) {{
  const contribs = {{}};
  if (!selectedPairs.length) {{
    document.getElementById('cse-contrib').innerHTML =
      '<div style="font-size:0.80rem;color:{PALETTE["muted"]}">Select at least one attack and one opinion target to see feature contributions.</div>';
    return;
  }}
  selectedPairs.forEach(([a,o]) => {{
    const c = ((TASKS[a] || {{}})[o]) || {{}};
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
  const pairs = getSelectedTasks();
  const {{ae_map, raw, pct, dist}} = computeScore(pf, pairs);
  renderTaskScope(pairs);
  renderGrid(ae_map);
  renderGauge(pct, raw, dist.length);
  renderRadar(pf);
  renderContrib(pf, pairs);
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
        ("🗂 Ontologies",        ["Ontology Explorer"]),
        ("📡 Factorial Space",   ["Factorial 3D Surface", "Factorial Heat + Contour"]),
        ("🧠 SEM Analysis",      ["SEM Network", "SEM Heatmap"]),
        ("🔬 Estimation",        ["Conditional Susceptibility Estimator", "Perturbation Explorer"]),
        ("👤 Profiles",          ["Susceptibility Map", "Profile Heatmap"]),
        ("📊 Moderators",        ["Moderator Forest", "Hierarchical Importance"]),
        ("📈 Raw Data",          ["Violin Distributions", "Attack Comparison", "Pre vs. Post Scatter", "Baseline vs Post"]),
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
    stage_outputs_root = s05.parent

    def _load(p: Path) -> pd.DataFrame:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    profile_df       = _load(s05 / "profile_level_effectivity.csv")
    profile_index_df = _load(s06 / "profile_susceptibility_index.csv")
    exploratory_df   = _load(s06 / "exploratory_moderator_comparison.csv")
    weight_df        = _load(s06 / "moderator_weight_table.csv")
    task_coeff_df    = _load(s06 / "conditional_susceptibility_task_coefficients.csv")
    task_summary_df  = _load(s06 / "conditional_susceptibility_task_summary.csv")
    ontology_catalog_path = stage_outputs_root / "01_create_scenarios" / "ontology_leaf_catalog.json"
    ontology_catalog = (
        json.loads(ontology_catalog_path.read_text(encoding="utf-8"))
        if ontology_catalog_path.exists() else {}
    )
    ontology_payload = _load_dashboard_ontology_payload(ontology_catalog)

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

    def _add_html(title: str, html: str, fname: Optional[str] = None) -> None:
        if fname:
            visual_files.append(_save_html_block(html, figures_dir / fname, title))
        figure_divs.append((title, html))

    _add_html("Ontology Explorer", _html_ontology_explorer(ontology_payload), "ontology_explorer.html")
    _add_fig("Factorial 3D Surface",    _fig_factorial_3d(long_df),          "factorial_3d.html")
    _add_fig("Factorial Heat + Contour", _fig_factorial_2d(long_df),          "factorial_2d.html")

    if not sem_coeff_df.empty:
        _add_html("SEM Network", _html_sem_network(sem_coeff_df, long_df),               "sem_network.html")
        _add_fig("SEM Heatmap",  _fig_sem_heatmap(sem_coeff_df, exploratory_df, long_df), "sem_heatmap.html")

    if not task_coeff_df.empty:
        _add_html("Conditional Susceptibility Estimator",
                  _html_cs_estimator(task_coeff_df, task_summary_df, long_df),
                  "conditional_susceptibility_estimator.html")
        _add_html("Perturbation Explorer",
                  _html_perturbation_explorer(task_coeff_df, long_df),
                  "perturbation_explorer.html")

    _add_fig("Violin Distributions", _fig_violin(long_df),               "violin.html")
    _add_fig("Attack Comparison",    _fig_raw_attack_comparison(long_df), "attack_comparison.html")
    _add_fig("Pre vs. Post Scatter", _fig_raw_score_scatter(long_df),    "pre_post_scatter.html")

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
        "Ontology Explorer: the dashboard defaults to the production ATTACK / OPINION / PROFILE ontologies, with a source switch to the test ontologies used in run 8.",
        "Adversarial effectivity (AE = Δ × d_k): <b>positive = manipulation succeeded</b>, negative = backfire or resistance.",
        "The 3D surface shows mean AE and inter-individual SD across the full 4×4 attack–opinion factorial.",
        "SEM Network: start with the Baseline preset, then use layer toggles, p-value thresholding, and hierarchy filters to reveal leaf-level moderation paths and attack context.",
        "Conditional Susceptibility Estimator: configure any profile, choose any conditional attack × opinion scope, and inspect the predicted AE grid plus score rank.",
        "The Conditional Susceptibility Score is re-computed on the fly for the selected task subset versus the 100 original profiles.",
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
