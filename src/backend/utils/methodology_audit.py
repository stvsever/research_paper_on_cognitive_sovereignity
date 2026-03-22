from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.backend.utils.schemas import SemFitResult


def build_assumption_register(df: pd.DataFrame, sem_result: SemFitResult) -> List[Dict[str, Any]]:
    n = len(df)
    attack_ratio = float(df["attack_present"].mean()) if "attack_present" in df.columns and n else 0.0
    score_unique = int(df["baseline_score"].nunique()) if "baseline_score" in df.columns else 0

    mean_realism = None
    if "attack_realism_score" in df.columns:
        attack_realism = df.loc[df["attack_present"] == 1, "attack_realism_score"].dropna()
        if len(attack_realism) > 0:
            mean_realism = float(attack_realism.mean())

    assumptions = [
        {
            "assumption": "Pilot sample size is sufficient for stable moderation estimates.",
            "status": "risk" if n < 80 else "ok",
            "evidence": {"n_rows": n},
            "mitigation": "Treat estimates as exploratory and scale scenario count in later runs.",
        },
        {
            "assumption": "Treatment-control balance is adequate.",
            "status": "ok" if 0.35 <= attack_ratio <= 0.65 else "risk",
            "evidence": {"attack_ratio": attack_ratio},
            "mitigation": "Use explicit stratified assignment in scenario generator.",
        },
        {
            "assumption": "Opinion measurement has enough numeric resolution.",
            "status": "ok" if score_unique >= max(6, int(0.4 * n)) else "risk",
            "evidence": {"unique_baseline_scores": score_unique},
            "mitigation": "Prompt enforces high-resolution scores in [-1000,1000].",
        },
        {
            "assumption": "Attack text realism is adequate.",
            "status": "ok" if mean_realism is not None and mean_realism >= 0.7 else "risk",
            "evidence": {"mean_attack_realism_score": mean_realism},
            "mitigation": "Self-supervised realism reviewer + rewrite loop in Stage 03.",
        },
        {
            "assumption": "SEM converges with acceptable fit.",
            "status": "ok" if sem_result.converged else "risk",
            "evidence": {"sem_converged": sem_result.converged, "warnings": sem_result.warnings},
            "mitigation": "Use robust OLS supplement and inspect diagnostics.",
        },
        {
            "assumption": "LLM outputs are reproducible and auditable.",
            "status": "ok",
            "evidence": {"raw_llm_logging": True},
            "mitigation": "Persist all prompts/outputs in provenance/raw_llm.",
        },
    ]
    return assumptions


def build_peer_review_critique_notes(df: pd.DataFrame, sem_result: SemFitResult) -> List[Dict[str, str]]:
    notes = [
        {
            "critique": "Synthetic LLM agents may not represent human cognition or causal response behavior.",
            "implemented_change": "Added transparent assumption register and explicit exploratory caveat in reports.",
        },
        {
            "critique": "Adversarial content may be unrealistic or too generic.",
            "implemented_change": "Added realism review agent with rewrite loop and heuristic checks.",
        },
        {
            "critique": "Model dependence on a single LLM could bias outputs.",
            "implemented_change": "Model is CLI-configurable and run metadata captures exact model for replication.",
        },
        {
            "critique": "Insufficient sample size inflates uncertainty and may destabilize SEM.",
            "implemented_change": "Included robust OLS complement and explicit pilot warnings; ready for scale-up runs.",
        },
    ]

    if len(df) < 20:
        notes.append(
            {
                "critique": "Pilot sample is too small for publication-grade inferential claims.",
                "implemented_change": "Outputs marked as pilot; pipeline prepared for larger scenario counts.",
            }
        )

    if sem_result.warnings:
        notes.append(
            {
                "critique": "SEM warnings indicate possible misfit or numerical instability.",
                "implemented_change": "Warnings surfaced in reports and fit diagnostics exported for scrutiny.",
            }
        )

    return notes


def render_methodology_audit_text(
    assumptions: List[Dict[str, Any]],
    critiques: List[Dict[str, str]],
) -> str:
    lines: List[str] = [
        "Methodology Audit and Peer-Review Risk Register",
        "===============================================",
        "",
        "Assumption Register",
        "-------------------",
    ]

    for idx, item in enumerate(assumptions, start=1):
        lines.append(f"{idx}. Assumption: {item['assumption']}")
        lines.append(f"   Status: {item['status']}")
        lines.append(f"   Evidence: {item['evidence']}")
        lines.append(f"   Mitigation: {item['mitigation']}")
        lines.append("")

    lines.append("Peer-Review Critique Mitigations")
    lines.append("-------------------------------")
    for idx, item in enumerate(critiques, start=1):
        lines.append(f"{idx}. Potential critique: {item['critique']}")
        lines.append(f"   Implemented change: {item['implemented_change']}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
