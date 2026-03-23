from __future__ import annotations

"""
Technical overview
------------------
This module builds pseudoprofiles from the hierarchical PROFILE ontology. It
handles mixed variable types:
- categorical selections, such as sex
- continuous trait values, such as age or Big Five facet percentiles

The deterministic generator creates a coherent baseline profile from ontology
leaf structure alone. Optional LLM-based profile generation can then modify
that baseline, but the deterministic seed remains the structural fallback and
anchor.

Why this module matters:
- it is the entry point that converts ontology structure into profile vectors
- it preserves the distinction between categorical and continuous profile data
- it provides consistent profile IDs and selected leaf metadata for downstream
  scenario generation and moderation analysis

The legacy `heuristic_shift_sensitivity_proxy` and `resilience_index` are still
attached here because other realism layers use them for boundedness checks.
They are not intended to be the final susceptibility estimator used in the
paper-facing analysis pipeline.
"""

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from src.backend.utils.ontology_utils import flatten_leaf_paths
from src.backend.utils.scenario_realism import (
    compute_resilience_index,
    compute_shift_sensitivity_proxy,
)
from src.backend.utils.schemas import ProfileConfiguration


@dataclass
class ProfileSamplingResult:
    profile: ProfileConfiguration
    sampling_mode_used: str


def _normalize_token(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _extract_big_five_structure(profile_leaf_paths: List[str]) -> Dict[str, List[str]]:
    structure: Dict[str, List[str]] = {}
    for path in profile_leaf_paths:
        parts = [p.strip() for p in path.split(">")]
        if len(parts) < 5:
            continue
        if "personality" not in parts[1].lower() and "personality" not in " ".join(parts).lower():
            continue
        if "big_five" not in " ".join(parts).lower():
            continue
        trait = parts[-2]
        facet = parts[-1]
        structure.setdefault(trait, []).append(facet)
    return structure


def _extract_sex_options(profile_leaf_paths: List[str]) -> List[str]:
    values: List[str] = []
    for path in profile_leaf_paths:
        parts = [p.strip() for p in path.split(">")]
        if len(parts) >= 3 and parts[-2].lower() == "sex":
            values.append(parts[-1])
    return values or ["Male", "Female", "Other"]


def deterministic_profile(
    profile_leaf_paths: List[str],
    profile_id: str,
    seed: int,
) -> ProfileConfiguration:
    rng = random.Random(seed)

    big_five_structure = _extract_big_five_structure(profile_leaf_paths)
    sex_options = _extract_sex_options(profile_leaf_paths)

    categorical_attributes: Dict[str, str] = {
        "sex": rng.choice(sex_options),
    }

    continuous_attributes: Dict[str, float] = {}
    age_years = max(18, min(85, int(rng.gauss(42, 14))))
    continuous_attributes["age_years"] = float(age_years)

    for trait, facets in sorted(big_five_structure.items()):
        trait_key = _normalize_token(trait)
        trait_anchor = max(0.0, min(100.0, rng.betavariate(2.1, 2.1) * 100.0))
        facet_values: List[float] = []
        for facet in sorted(facets):
            facet_key = _normalize_token(facet)
            facet_val = max(0.0, min(100.0, trait_anchor + rng.gauss(0.0, 8.5)))
            facet_values.append(facet_val)
            continuous_attributes[f"big_five_{trait_key}_{facet_key}_pct"] = round(facet_val, 3)
        if facet_values:
            continuous_attributes[f"big_five_{trait_key}_mean_pct"] = round(
                sum(facet_values) / len(facet_values), 3
            )

    tmp_profile = ProfileConfiguration(
        profile_id=profile_id,
        categorical_attributes=categorical_attributes,
        continuous_attributes=continuous_attributes,
        selected_leaf_nodes=[],
    )
    # Legacy realism helpers only. They support prompt-boundedness checks but are not the
    # analysis-facing susceptibility construct, which is now computed post hoc conditional
    # on the configured attack/opinion target set in Stage 06.
    continuous_attributes["heuristic_shift_sensitivity_proxy"] = compute_shift_sensitivity_proxy(tmp_profile)
    continuous_attributes["resilience_index"] = compute_resilience_index(tmp_profile)

    # TODO: add realism heuristics to reject implausible high-order combinations
    # (e.g., conflicting profiles that may be semantically inconsistent at scale).

    selected_leaf_nodes = [
        path
        for path in profile_leaf_paths
        if "> Sex >" in path or "> Personality >" in path or "> Age >" in path
    ]

    return ProfileConfiguration(
        profile_id=profile_id,
        categorical_attributes=categorical_attributes,
        continuous_attributes=continuous_attributes,
        selected_leaf_nodes=selected_leaf_nodes,
        metadata={"generation": "deterministic"},
    )


def sample_profile(
    profile_tree: Dict[str, dict],
    profile_id: str,
    seed: int,
    generation_mode: str,
    llm_generator: Optional[
        Callable[[str, int, List[str], ProfileConfiguration], Optional[ProfileConfiguration]]
    ] = None,
) -> ProfileSamplingResult:
    profile_leaf_paths = flatten_leaf_paths(profile_tree)

    deterministic = deterministic_profile(
        profile_leaf_paths=profile_leaf_paths,
        profile_id=profile_id,
        seed=seed,
    )

    mode = generation_mode.lower()

    if mode == "deterministic":
        return ProfileSamplingResult(profile=deterministic, sampling_mode_used="deterministic")

    if llm_generator is None:
        return ProfileSamplingResult(
            profile=deterministic,
            sampling_mode_used="deterministic_fallback_no_llm_generator",
        )

    llm_result = llm_generator(profile_id, seed, profile_leaf_paths, deterministic)
    if llm_result is None:
        return ProfileSamplingResult(
            profile=deterministic,
            sampling_mode_used=f"{mode}_fallback_deterministic",
        )

    merged_continuous = dict(deterministic.continuous_attributes)
    merged_continuous.update(llm_result.continuous_attributes)

    merged_categorical = dict(deterministic.categorical_attributes)
    merged_categorical.update(llm_result.categorical_attributes)

    merged = ProfileConfiguration(
        profile_id=profile_id,
        categorical_attributes=merged_categorical,
        continuous_attributes=merged_continuous,
        selected_leaf_nodes=deterministic.selected_leaf_nodes,
        metadata={
            "generation": mode,
            "deterministic_seed": seed,
            "llm_profile_adjusted": True,
        },
    )
    # Legacy realism helpers only. They are excluded from the conditional susceptibility
    # index fit used for downstream moderation reporting.
    merged.continuous_attributes["heuristic_shift_sensitivity_proxy"] = compute_shift_sensitivity_proxy(merged)
    merged.continuous_attributes["resilience_index"] = compute_resilience_index(merged)
    return ProfileSamplingResult(profile=merged, sampling_mode_used=mode)
