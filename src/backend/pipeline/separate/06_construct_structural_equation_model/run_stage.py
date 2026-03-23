from __future__ import annotations

"""
Technical overview
------------------
Stage 06 is the main modeling stage. It takes the attacked long table and the
profile-level repeated-outcome panel from Stage 05 and produces three linked
outputs:

1. a repeated-outcome path SEM over attacked opinion-shift indicators
2. robust OLS / bootstrap summaries for profile-level attacked effectivity
3. a post hoc conditional susceptibility artifact and ranking

The SEM side answers:
    which profile features are associated with larger attacked shifts on each
    repeated opinion leaf?

The conditional susceptibility side answers:
    given the configured attack-leaf set and opinion-leaf set, which profiles
    are predicted to be more susceptible overall?

This distinction matters. The SEM/path estimates remain leaf-specific, while
the conditional susceptibility index aggregates fitted task-level profile
effects into a reusable target-conditional profile score.

This module therefore sits at the center of the research design:
- it preserves repeated attacked outcomes instead of collapsing too early
- it separates descriptive profile ranking from path-level moderation output
- it prepares reusable fitted artifacts for future runs and later nonlinear
  model replacements
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from semopy import Model, calc_stats
from semopy.inspector import inspect as sem_inspect

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.conditional_susceptibility import (
    build_conditional_weight_table,
    fit_conditional_susceptibility_index,
)
from src.backend.utils.data_utils import infer_analysis_mode, zscore_series
from src.backend.utils.io import abs_path, ensure_dir, stage_manifest_path, write_json, write_text
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.methodology_audit import (
    build_assumption_register,
    build_peer_review_critique_notes,
    render_methodology_audit_text,
)
from src.backend.utils.schemas import SemCoefficient, SemFitResult, StageArtifactManifest, StageConfig

LOGGER = logging.getLogger(__name__)


class Stage06Config(StageConfig):
    primary_moderator: str = "posthoc_profile_susceptibility_index"
    bootstrap_samples: int = 500


CORE_CONTINUOUS_MODERATORS = [
    "profile_cont_age_years",
    "profile_cont_big_five_neuroticism_mean_pct",
    "profile_cont_big_five_openness_to_experience_mean_pct",
    "profile_cont_big_five_conscientiousness_mean_pct",
]
EXPLORATORY_CONTINUOUS_MODERATORS = [
    "profile_cont_big_five_extraversion_mean_pct",
    "profile_cont_big_five_agreeableness_mean_pct",
]
SEX_COLUMNS = [
    "profile_cat__profile_cat_sex_Female",
    "profile_cat__profile_cat_sex_Other",
]
CONTROL_COLUMNS = [
    "mean_baseline_abs_score_z",
    "mean_exposure_quality_score_z",
]


def _pretty_moderator_label(column_name: str) -> str:
    label = column_name
    for prefix in ["profile_cont_", "profile_cat__profile_cat_", "profile_cat__", "profile_cat_"]:
        if label.startswith(prefix):
            label = label[len(prefix) :]
    label = label.replace("_z", "")
    label = label.replace("__", " ")
    label = label.replace("_", " ").strip()
    return " ".join(part.capitalize() if part.lower() != "pct" else "%" for part in label.split())


def _normalize_fit_indices(raw: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "Value" in value:
            inner = value["Value"]
            flat[key] = float(inner) if hasattr(inner, "__float__") else inner
        else:
            flat[key] = float(value) if hasattr(value, "__float__") else value
    for bounded_key in ["CFI", "TLI", "GFI", "AGFI", "NFI"]:
        if bounded_key in flat and flat[bounded_key] is not None:
            flat[bounded_key] = max(0.0, min(1.0, float(flat[bounded_key])))
    if "RMSEA" in flat and flat["RMSEA"] is not None:
        flat["RMSEA"] = max(0.0, float(flat["RMSEA"]))
    return flat


def _indicator_columns(profile_df: pd.DataFrame) -> List[str]:
    return sorted(
        column
        for column in profile_df.columns
        if column.startswith("abs_delta_indicator__") and not column.endswith("_z")
    )


def _safe_optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _available(columns: Sequence[str], df: pd.DataFrame) -> List[str]:
    return [column for column in columns if column in df.columns and df[column].nunique(dropna=True) > 1]


def _ensure_standardized_columns(df: pd.DataFrame, continuous_columns: Sequence[str]) -> pd.DataFrame:
    work = df.copy()
    for column in continuous_columns:
        if column in work.columns and work[column].nunique(dropna=True) > 1:
            work[f"{column}_z"] = zscore_series(work[column].astype(float))
    return work


def _core_structural_terms(df: pd.DataFrame) -> List[str]:
    terms: List[str] = []
    terms.extend(_available([f"{column}_z" for column in CORE_CONTINUOUS_MODERATORS], df))
    terms.extend(_available(SEX_COLUMNS, df))
    terms.extend(_available(CONTROL_COLUMNS, df))
    return terms


def _all_profile_terms(df: pd.DataFrame) -> List[str]:
    terms: List[str] = []
    terms.extend(_available([f"{column}_z" for column in CORE_CONTINUOUS_MODERATORS], df))
    terms.extend(_available([f"{column}_z" for column in EXPLORATORY_CONTINUOUS_MODERATORS], df))
    terms.extend(_available(SEX_COLUMNS, df))
    return terms


def _build_formula(target: str, terms: Sequence[str]) -> str:
    rhs = " + ".join(terms) if terms else "1"
    return f"{target} ~ {rhs}"


def _fit_sem(
    profile_df: pd.DataFrame,
    indicator_columns: List[str],
    structural_terms: List[str],
) -> Tuple[SemFitResult, pd.DataFrame]:
    warnings: List[str] = []
    if len(indicator_columns) < 3:
        return (
            SemFitResult(
                model_name="profile_panel_path_sem",
                model_formula="",
                converged=False,
                n_obs=len(profile_df),
                fit_indices={},
                coefficients=[],
                warnings=["Need at least three abs-delta indicators for the profile-level SEM."],
            ),
            pd.DataFrame(),
        )

    regression_blocks = [
        _build_formula(indicator, structural_terms)
        for indicator in indicator_columns
    ]
    covariance_blocks: List[str] = []
    for idx, left in enumerate(indicator_columns):
        for right in indicator_columns[idx + 1 :]:
            covariance_blocks.append(f"{left} ~~ {right}")
    model_formula = "\n".join([*regression_blocks, *covariance_blocks])
    model = Model(model_formula)

    try:
        model.fit(profile_df)
        converged = True
    except Exception as exc:
        warnings.append(f"semopy fit failed: {exc}")
        converged = False

    coefficients: List[SemCoefficient] = []
    fit_indices: Dict[str, Any] = {}
    factor_scores = pd.DataFrame(index=profile_df.index)
    if converged:
        est = sem_inspect(model)
        for _, row in est.iterrows():
            coefficients.append(
                SemCoefficient(
                    lhs=str(row.get("lval", "")),
                    op=str(row.get("op", "")),
                    rhs=_pretty_moderator_label(str(row.get("rval", ""))),
                    estimate=float(row.get("Estimate", 0.0)),
                    std_error=_safe_optional_float(row.get("Std. Err")),
                    z_value=_safe_optional_float(row.get("z-value")),
                    p_value=_safe_optional_float(row.get("p-value")),
                )
            )
        stats = calc_stats(model)
        if hasattr(stats, "to_dict"):
            fit_indices = _normalize_fit_indices(stats.to_dict())

    return (
        SemFitResult(
            model_name="profile_panel_path_sem",
            model_formula=model_formula,
            converged=converged,
            n_obs=len(profile_df),
            fit_indices=fit_indices,
            coefficients=coefficients,
            warnings=warnings,
        ),
        factor_scores,
    )


def _ridge_fit_matrix(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    xtx = x.T @ x
    xty = x.T @ y
    return np.linalg.pinv(xtx + alpha * penalty) @ xty


def _kfold_indices(n_obs: int, seed: int, n_splits: int = 5) -> List[np.ndarray]:
    n_splits = max(2, min(n_splits, n_obs))
    indices = np.arange(n_obs)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return [fold for fold in np.array_split(indices, n_splits) if len(fold) > 0]


def _cross_validated_ridge(
    df: pd.DataFrame,
    outcome: str,
    predictor_terms: Sequence[str],
    seed: int,
    alpha_grid: Sequence[float] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.Series]:
    alpha_grid = list(alpha_grid or np.logspace(-3, 3, 25))
    design_terms = ["Intercept", *predictor_terms]
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            *[df[term].astype(float).to_numpy() for term in predictor_terms],
        ]
    )
    y = df[outcome].astype(float).to_numpy()
    folds = _kfold_indices(len(df), seed=seed, n_splits=5)

    best_alpha = alpha_grid[0]
    best_cv_mse = float("inf")
    for alpha in alpha_grid:
        fold_mses: List[float] = []
        for fold in folds:
            mask = np.ones(len(df), dtype=bool)
            mask[fold] = False
            beta = _ridge_fit_matrix(x[mask], y[mask], alpha)
            preds = x[fold] @ beta
            fold_mses.append(float(np.mean((y[fold] - preds) ** 2)))
        cv_mse = float(np.mean(fold_mses))
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_alpha = alpha

    beta = _ridge_fit_matrix(x, y, best_alpha)
    coeff_df = pd.DataFrame(
        {
            "outcome": outcome,
            "term": design_terms,
            "estimate": beta,
            "alpha": best_alpha,
            "cv_mse": best_cv_mse,
        }
    )
    predictions = pd.Series(x @ beta, index=df.index, name=f"predicted__{outcome}")
    model_meta = {
        "outcome": outcome,
        "alpha": float(best_alpha),
        "cv_mse": float(best_cv_mse),
    }
    return coeff_df, model_meta, predictions


def _fit_ridge_path_models(
    profile_df: pd.DataFrame,
    indicator_columns: Sequence[str],
    predictor_terms: Sequence[str],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coeff_frames: List[pd.DataFrame] = []
    meta_rows: List[Dict[str, float]] = []
    for offset, outcome in enumerate(indicator_columns):
        coeff_df, meta, _ = _cross_validated_ridge(
            df=profile_df,
            outcome=outcome,
            predictor_terms=predictor_terms,
            seed=seed + offset,
        )
        coeff_frames.append(coeff_df)
        meta_rows.append(meta)
    return pd.concat(coeff_frames, ignore_index=True), pd.DataFrame(meta_rows)


def _moderator_group(term: str) -> str:
    if "age_years" in term:
        return "Demographics: Age"
    if "sex_" in term:
        return "Demographics: Sex"
    if "neuroticism" in term:
        return "Personality: Neuroticism"
    if "openness" in term:
        return "Personality: Openness"
    if "conscientiousness" in term:
        return "Personality: Conscientiousness"
    if "extraversion" in term:
        return "Personality: Extraversion"
    if "agreeableness" in term:
        return "Personality: Agreeableness"
    return "Other"


def _build_moderator_weight_table(
    profile_df: pd.DataFrame,
    ridge_params: pd.DataFrame,
    moderator_terms: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for term in moderator_terms:
        if term not in profile_df.columns:
            continue
        match = ridge_params.loc[ridge_params["term"] == term]
        if match.empty:
            continue
        estimate = float(match["estimate"].mean())
        mean_abs_estimate = float(match["estimate"].abs().mean())
        term_sd = float(profile_df[term].astype(float).std(ddof=0))
        importance = mean_abs_estimate * term_sd
        rows.append(
            {
                "term": term,
                "moderator_label": _pretty_moderator_label(term),
                "ontology_group": _moderator_group(term),
                "estimate": estimate,
                "mean_abs_estimate": mean_abs_estimate,
                "term_sd": term_sd,
                "importance": importance,
                "direction": "higher_effectivity" if estimate >= 0 else "lower_effectivity",
                "n_outcomes": int(match["outcome"].nunique()) if "outcome" in match.columns else 1,
            }
        )
    weight_df = pd.DataFrame(rows)
    if weight_df.empty:
        return weight_df
    denom = float(weight_df["importance"].sum()) or 1.0
    weight_df["normalized_weight_pct"] = (weight_df["importance"] / denom) * 100.0
    return weight_df.sort_values(["normalized_weight_pct", "moderator_label"], ascending=[False, True]).reset_index(drop=True)


def _bootstrap_ols(
    df: pd.DataFrame,
    formula: str,
    terms: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: List[Dict[str, float]] = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(df), size=len(df))
        sample = df.iloc[sample_idx].copy()
        try:
            model = smf.ols(formula, data=sample).fit()
        except Exception:
            continue
        records.append({term: float(model.params.get(term, np.nan)) for term in terms})

    if not records:
        return pd.DataFrame(
            {
                "term": list(terms),
                "bootstrap_mean": np.nan,
                "bootstrap_std": np.nan,
                "conf_low": np.nan,
                "conf_high": np.nan,
                "n_bootstrap_success": 0,
            }
        )

    boot_df = pd.DataFrame(records)
    summary_rows: List[Dict[str, object]] = []
    for term in terms:
        values = boot_df[term].dropna()
        summary_rows.append(
            {
                "term": term,
                "bootstrap_mean": float(values.mean()) if len(values) else np.nan,
                "bootstrap_std": float(values.std(ddof=0)) if len(values) else np.nan,
                "conf_low": float(values.quantile(0.025)) if len(values) else np.nan,
                "conf_high": float(values.quantile(0.975)) if len(values) else np.nan,
                "n_bootstrap_success": int(len(values)),
            }
        )
    return pd.DataFrame(summary_rows)


def _fit_exploratory_models(
    profile_df: pd.DataFrame,
    multivariate_params: pd.DataFrame,
    control_terms: List[str],
    candidate_terms: List[str],
) -> pd.DataFrame:
    multivariate_lookup = {row["term"]: row for row in multivariate_params.to_dict(orient="records")}
    rows: List[Dict[str, object]] = []

    for term in candidate_terms:
        formula = _build_formula("mean_abs_delta_score", [term, *control_terms])
        try:
            result = smf.ols(formula, data=profile_df).fit(cov_type="HC3")
        except Exception as exc:
            LOGGER.warning("Exploratory model failed for %s: %s", term, exc)
            continue
        conf = result.conf_int()
        multi = multivariate_lookup.get(term, {})
        rows.append(
            {
                "moderator_column": term,
                "moderator_label": _pretty_moderator_label(term),
                "multivariate_estimate": float(multi.get("estimate", np.nan)),
                "multivariate_std_error": float(multi.get("std_error", np.nan)),
                "multivariate_p_value": float(multi.get("p_value", np.nan)),
                "multivariate_conf_low": float(multi.get("conf_low", np.nan)),
                "multivariate_conf_high": float(multi.get("conf_high", np.nan)),
                "univariate_estimate": float(result.params.get(term, np.nan)),
                "univariate_std_error": float(result.bse.get(term, np.nan)),
                "univariate_p_value": float(result.pvalues.get(term, np.nan)),
                "univariate_conf_low": float(conf.loc[term, 0]),
                "univariate_conf_high": float(conf.loc[term, 1]),
                "role": "core" if term in _available([f"{column}_z" for column in CORE_CONTINUOUS_MODERATORS], profile_df) or term in _available(SEX_COLUMNS, profile_df) else "exploratory",
            }
        )
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    return comparison.sort_values(["role", "multivariate_p_value", "moderator_label"]).reset_index(drop=True)


def _compute_profile_susceptibility_outputs(
    profile_df: pd.DataFrame,
    ridge_params: pd.DataFrame,
    moderator_terms: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lookup = (
        ridge_params.groupby("term", as_index=False)["estimate"]
        .mean()
        .set_index("term")["estimate"]
        .to_dict()
    )
    intercept = float(lookup.get("Intercept", 0.0))
    scores = pd.Series(np.zeros(len(profile_df)), index=profile_df.index, dtype=float)
    contributions: Dict[str, pd.Series] = {}
    group_contributions: Dict[str, pd.Series] = {}
    breakdown_rows: List[Dict[str, object]] = []

    for term in moderator_terms:
        if term not in profile_df.columns:
            continue
        beta = float(lookup.get(term, 0.0))
        contribution = profile_df[term].astype(float) * beta
        scores = scores + contribution
        contributions[term] = contribution
        group = _moderator_group(term)
        if group not in group_contributions:
            group_contributions[group] = pd.Series(np.zeros(len(profile_df)), index=profile_df.index, dtype=float)
        group_contributions[group] = group_contributions[group] + contribution

    result = profile_df[["profile_id", "mean_abs_delta_score", "mean_signed_delta_score"]].copy()
    if "latent_attack_effectivity_factor_score" in profile_df.columns:
        result["latent_attack_effectivity_factor_score"] = profile_df["latent_attack_effectivity_factor_score"]
        result["latent_attack_effectivity_factor_score_z"] = zscore_series(profile_df["latent_attack_effectivity_factor_score"])
    result["profile_moderator_linear_score"] = scores
    result["profile_moderator_linear_score_z"] = zscore_series(scores)
    result["predicted_mean_abs_delta_from_moderators"] = intercept + scores
    result["susceptibility_index_pct"] = scores.rank(method="average", pct=True) * 100.0
    result["observed_effectivity_pct"] = result["mean_abs_delta_score"].rank(method="average", pct=True) * 100.0

    for term, contribution in contributions.items():
        result[f"contribution__{term}"] = contribution

    for group, contribution in group_contributions.items():
        slug = (
            group.lower()
            .replace(": ", "__")
            .replace(" ", "_")
            .replace("-", "_")
        )
        result[f"group_contribution__{slug}"] = contribution

    for row in result.to_dict(orient="records"):
        for term in contributions:
            breakdown_rows.append(
                {
                    "profile_id": row["profile_id"],
                    "component_type": "term",
                    "component_name": _pretty_moderator_label(term),
                    "component_key": term,
                    "ontology_group": _moderator_group(term),
                    "contribution": row.get(f"contribution__{term}", np.nan),
                    "susceptibility_index_pct": row["susceptibility_index_pct"],
                }
            )
        for group in group_contributions:
            slug = (
                group.lower()
                .replace(": ", "__")
                .replace(" ", "_")
                .replace("-", "_")
            )
            breakdown_rows.append(
                {
                    "profile_id": row["profile_id"],
                    "component_type": "group",
                    "component_name": group,
                    "component_key": slug,
                    "ontology_group": group,
                    "contribution": row.get(f"group_contribution__{slug}", np.nan),
                    "susceptibility_index_pct": row["susceptibility_index_pct"],
                }
            )

    result = result.sort_values("susceptibility_index_pct", ascending=False).reset_index(drop=True)
    breakdown_df = pd.DataFrame(breakdown_rows)
    return result, breakdown_df


def _safe_ols_summary(ols_model) -> str:
    try:
        return str(ols_model.summary())
    except Exception as exc:
        fallback_lines = [
            "statsmodels summary unavailable",
            f"reason: {exc}",
            "",
            "parameter estimates",
            ols_model.params.to_string(),
            "",
            "p-values",
            ols_model.pvalues.to_string(),
        ]
        return "\n".join(fallback_lines)


def _render_report(
    long_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    sem_result: SemFitResult,
    multivariate_formula: str,
    ols_summary: str,
    ols_table: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    exploratory_table: pd.DataFrame,
    profile_index_df: pd.DataFrame,
    weight_table: pd.DataFrame,
    task_summary_df: pd.DataFrame,
    run_id: str,
) -> str:
    fit_cfi = sem_result.fit_indices.get("CFI")
    fit_rmsea = sem_result.fit_indices.get("RMSEA")
    fit_line = (
        f"CFI={fit_cfi:.3f}, RMSEA={fit_rmsea:.3f}"
        if fit_cfi is not None and fit_rmsea is not None
        else "fit indices unavailable"
    )

    realism_text = "n/a"
    if "attack_realism_score" in long_df.columns:
        realism_vals = long_df["attack_realism_score"].dropna()
        if len(realism_vals) > 0:
            realism_text = f"{float(realism_vals.mean()):.3f}"

    plausibility_text = "n/a"
    if "post_plausibility_score" in long_df.columns:
        plausibility_vals = long_df["post_plausibility_score"].dropna()
        if len(plausibility_vals) > 0:
            plausibility_text = f"{float(plausibility_vals.mean()):.3f}"

    indicator_columns = _indicator_columns(profile_df)
    top_multivariate = exploratory_table.sort_values("multivariate_p_value").head(6) if not exploratory_table.empty else pd.DataFrame()
    top_weights = weight_table.head(6) if not weight_table.empty else pd.DataFrame()
    attack_leaves = sorted({str(value) for value in long_df["attack_leaf"].dropna().unique().tolist()}) if "attack_leaf" in long_df.columns else []
    opinion_leaves = sorted({str(value) for value in long_df["opinion_leaf"].dropna().unique().tolist()}) if "opinion_leaf" in long_df.columns else []
    task_meta_lines = []
    if not task_summary_df.empty:
        for row in task_summary_df.to_dict(orient="records"):
            task_meta_lines.append(
                f"{row['attack_leaf']} | {row['opinion_leaf']}: alpha={float(row['alpha']):.4f}, cv_mse={float(row['cv_mse']):.4f}, weight={float(row['reliability_weight']):.4f}"
            )

    bootstrap_lookup = {row["term"]: row for row in bootstrap_table.to_dict(orient="records")}

    lines = [
        f"Moderation Report - {run_id}",
        "=========================",
        "",
        f"Profiles analyzed: {len(profile_df)}",
        f"Attacked opinion scenarios analyzed: {len(long_df)}",
        f"Repeated opinion indicators: {len(indicator_columns)}",
        f"Mean absolute delta: {float(long_df['abs_delta_score'].mean()):.3f}",
        f"Mean signed delta: {float(long_df['delta_score'].mean()):.3f}",
        f"Mean attack realism score: {realism_text}",
        f"Mean post-exposure plausibility score: {plausibility_text}",
        "",
        "Profile-Level Path SEM",
        "----------------------",
        f"Converged: {sem_result.converged}",
        f"Fit indices: {fit_line}",
        f"Warnings: {', '.join(sem_result.warnings) if sem_result.warnings else 'none'}",
        f"Formula: {sem_result.model_formula}",
        "",
        "Regularized Susceptibility Model",
        "--------------------------------",
        "The empirical susceptibility index is conditional on the modeled attack-leaf set and opinion-leaf set. It is derived from task-specific cross-validated ridge models fit separately to each (attack leaf, opinion leaf) target, then aggregated back to the profile level using reliability weights.",
        f"Attack target set: {', '.join(attack_leaves) if attack_leaves else 'n/a'}",
        f"Opinion target set: {', '.join(opinion_leaves) if opinion_leaves else 'n/a'}",
    ]

    if task_meta_lines:
        lines.extend(task_meta_lines)

    lines.extend(
        [
            "",
        "Primary Multivariate Profile Model",
        "----------------------------------",
        f"Outcome: mean_abs_delta_score",
        f"Formula: {multivariate_formula}",
        ]
    )

    for row in ols_table.to_dict(orient="records"):
        boot = bootstrap_lookup.get(row["term"], {})
        lines.append(
            f"{row['term']}: est={row['estimate']:.4f}, p={row['p_value']:.6f}, boot95=[{boot.get('conf_low', np.nan):.4f}, {boot.get('conf_high', np.nan):.4f}]"
        )

    if not top_multivariate.empty:
        lines.extend(["", "Moderator Highlights", "-------------------"])
        for row in top_multivariate.to_dict(orient="records"):
            lines.append(
                f"{row['moderator_label']}: mean univariate b={row['univariate_estimate']:.4f}, p={row['univariate_p_value']:.6f}; ridge mean b={row.get('ridge_mean_estimate', np.nan):.4f}, weight_pct={row.get('normalized_weight_pct', np.nan):.2f}"
            )

    if not profile_index_df.empty:
        top_profiles = profile_index_df.head(5)
        lines.extend(["", "Empirical Profile Susceptibility Index", "------------------------------------"])
        lines.append("The post hoc susceptibility index is the percentile-ranked profile-only linear predictor under the configured attack/opinion target set.")
        for row in top_profiles.to_dict(orient="records"):
            lines.append(
                f"{row['profile_id']}: susceptibility_index_pct={row['susceptibility_index_pct']:.2f}, mean_abs_delta={row['mean_abs_delta_score']:.2f}"
            )

    if not top_weights.empty:
        lines.extend(["", "Moderator Weight Decomposition", "------------------------------"])
        for row in top_weights.to_dict(orient="records"):
            lines.append(
                f"{row['moderator_label']} [{row['ontology_group']}]: est={row['estimate']:.4f}, normalized_weight_pct={row['normalized_weight_pct']:.2f}"
            )

    lines.extend(
        [
            "",
            "OLS Supplement",
            "--------------",
            ols_summary,
            "",
            "Caveat",
            "------",
            "This attacked-only pilot estimates heterogeneity in attacked opinion movement. It does not estimate a no-attack counterfactual effect, and the post hoc susceptibility index is descriptive because it is derived from the fitted profile moderation model rather than observed independently.",
        ]
    )
    return "\n".join(lines)


def run_stage(input_path: str, output_dir: str, config: Stage06Config) -> StageArtifactManifest:
    ensure_dir(output_dir)
    long_df = pd.read_csv(input_path)
    analysis_mode = infer_analysis_mode(long_df)
    if analysis_mode != "treated_only":
        raise RuntimeError("Run 6 SEM stage is designed for attacked-only profile-panel data.")

    stage05_dir = Path(input_path).resolve().parent
    profile_summary_path = stage05_dir / "profile_level_effectivity.csv"
    profile_wide_path = stage05_dir / "profile_sem_wide.csv"
    if not profile_summary_path.exists() or not profile_wide_path.exists():
        raise RuntimeError("Stage 06 requires profile_level_effectivity.csv and profile_sem_wide.csv from Stage 05.")

    profile_summary_df = pd.read_csv(profile_summary_path)
    profile_df = pd.read_csv(profile_wide_path)
    profile_df = _ensure_standardized_columns(
        profile_df,
        [*CORE_CONTINUOUS_MODERATORS, *EXPLORATORY_CONTINUOUS_MODERATORS],
    )

    indicator_columns = _indicator_columns(profile_df)
    structural_terms = _core_structural_terms(profile_df)
    control_terms = _available(CONTROL_COLUMNS, profile_df)
    profile_terms = _all_profile_terms(profile_df)

    sem_result, factor_scores = _fit_sem(profile_df, indicator_columns=indicator_columns, structural_terms=structural_terms)

    multivariate_terms = [*profile_terms, *control_terms]
    multivariate_formula = _build_formula("mean_abs_delta_score", multivariate_terms)
    ols_model = smf.ols(multivariate_formula, data=profile_df).fit(cov_type="HC3")
    conf_int = ols_model.conf_int()
    ols_table = pd.DataFrame(
        {
            "term": ols_model.params.index,
            "estimate": ols_model.params.values,
            "std_error": ols_model.bse.values,
            "p_value": ols_model.pvalues.values,
            "conf_low": conf_int[0].values,
            "conf_high": conf_int[1].values,
        }
    )

    bootstrap_table = _bootstrap_ols(
        df=profile_df,
        formula=multivariate_formula,
        terms=list(ols_model.params.index),
        n_bootstrap=config.bootstrap_samples,
        seed=config.seed,
    )
    exploratory_table = _fit_exploratory_models(
        profile_df=profile_df,
        multivariate_params=ols_table,
        control_terms=control_terms,
        candidate_terms=profile_terms,
    )
    conditional_fit = fit_conditional_susceptibility_index(
        long_df=long_df,
        outcome_metric="abs_delta_score",
        feature_columns=profile_terms,
        excluded_feature_columns=[
            "profile_cont_heuristic_shift_sensitivity_proxy",
            "profile_cont_resilience_index",
        ],
        seed=config.seed,
    )
    weight_table = build_conditional_weight_table(conditional_fit.artifact)
    weight_table["moderator_label"] = weight_table["term"].map(_pretty_moderator_label)
    weight_table["ontology_group"] = weight_table["term"].map(_moderator_group)
    weight_table["direction"] = np.where(
        weight_table["weighted_mean_estimate"] >= 0,
        "higher_effectivity",
        "lower_effectivity",
    )
    weight_table["estimate"] = weight_table["weighted_mean_estimate"]
    weight_table["mean_abs_estimate"] = weight_table["weighted_mean_abs_estimate"]

    profile_index_df = conditional_fit.profile_scores.merge(
        profile_df[["profile_id", "mean_abs_delta_score", "mean_signed_delta_score"]],
        on="profile_id",
        how="left",
    )
    profile_index_df["observed_effectivity_pct"] = (
        profile_index_df["mean_abs_delta_score"].rank(method="average", pct=True) * 100.0
    )
    contribution_breakdown_df = conditional_fit.contribution_breakdown
    ridge_summary_lookup = (
        conditional_fit.task_coefficients.loc[conditional_fit.task_coefficients["term"] != "Intercept"]
        .groupby("term", as_index=False)
        .agg(
            ridge_mean_estimate=("estimate", "mean"),
            ridge_mean_abs_estimate=("estimate", lambda s: float(np.mean(np.abs(s)))),
        )
    )
    exploratory_table = exploratory_table.merge(
        ridge_summary_lookup,
        left_on="moderator_column",
        right_on="term",
        how="left",
    ).drop(columns=["term"], errors="ignore")
    exploratory_table = exploratory_table.merge(
        weight_table[["term", "normalized_weight_pct"]],
        left_on="moderator_column",
        right_on="term",
        how="left",
    ).drop(columns=["term"], errors="ignore")

    out = Path(output_dir)
    spec_txt = out / "sem_model_spec.txt"
    profile_formula_txt = out / "profile_multivariate_model_spec.txt"
    sem_json = out / "sem_result.json"
    sem_coeff_csv = out / "sem_coefficients.csv"
    sem_fit_json = out / "sem_fit_indices.json"
    ols_txt = out / "ols_robust_summary.txt"
    ols_params_csv = out / "ols_robust_params.csv"
    bootstrap_csv = out / "bootstrap_primary_params.csv"
    exploratory_csv = out / "exploratory_moderator_comparison.csv"
    weight_table_csv = out / "moderator_weight_table.csv"
    profile_index_csv = out / "profile_susceptibility_index.csv"
    contribution_breakdown_csv = out / "profile_susceptibility_breakdown.csv"
    latent_scores_csv = out / "latent_attack_effectivity_scores.csv"
    profile_summary_copy_csv = out / "profile_level_effectivity.csv"
    profile_wide_copy_csv = out / "profile_sem_wide.csv"
    ridge_coeff_csv = out / "conditional_susceptibility_task_coefficients.csv"
    ridge_summary_csv = out / "conditional_susceptibility_task_summary.csv"
    conditional_artifact_json = out / "conditional_susceptibility_artifact.json"
    report_txt = out / "moderation_report.txt"
    assumptions_json = out / "assumption_register.json"
    critiques_json = out / "peer_review_critiques.json"
    methodology_txt = out / "methodology_audit.txt"

    write_text(spec_txt, sem_result.model_formula)
    write_text(profile_formula_txt, multivariate_formula)
    write_json(sem_json, sem_result.model_dump())
    pd.DataFrame([coeff.model_dump() for coeff in sem_result.coefficients]).to_csv(sem_coeff_csv, index=False)
    write_json(sem_fit_json, sem_result.fit_indices)
    ols_summary_text = _safe_ols_summary(ols_model)
    write_text(ols_txt, ols_summary_text)
    ols_table.to_csv(ols_params_csv, index=False)
    bootstrap_table.to_csv(bootstrap_csv, index=False)
    exploratory_table.to_csv(exploratory_csv, index=False)
    weight_table.to_csv(weight_table_csv, index=False)
    profile_index_df.to_csv(profile_index_csv, index=False)
    contribution_breakdown_df.to_csv(contribution_breakdown_csv, index=False)
    conditional_fit.task_coefficients.to_csv(ridge_coeff_csv, index=False)
    conditional_fit.task_summary.to_csv(ridge_summary_csv, index=False)
    write_json(conditional_artifact_json, conditional_fit.artifact.model_dump())
    if "latent_attack_effectivity_factor_score" in profile_df.columns:
        profile_df[["profile_id", "latent_attack_effectivity_factor_score"]].to_csv(latent_scores_csv, index=False)
    profile_summary_df.to_csv(profile_summary_copy_csv, index=False)
    profile_df.to_csv(profile_wide_copy_csv, index=False)

    write_text(
        report_txt,
        _render_report(
            long_df=long_df,
            profile_df=profile_df,
            sem_result=sem_result,
            multivariate_formula=multivariate_formula,
            ols_summary=ols_summary_text,
            ols_table=ols_table,
            bootstrap_table=bootstrap_table,
            exploratory_table=exploratory_table,
            profile_index_df=profile_index_df,
            weight_table=weight_table,
            task_summary_df=conditional_fit.task_summary,
            run_id=config.run_id,
        ),
    )

    assumptions = build_assumption_register(long_df, sem_result)
    critiques = build_peer_review_critique_notes(long_df, sem_result)
    write_json(assumptions_json, assumptions)
    write_json(critiques_json, critiques)
    write_text(methodology_txt, render_methodology_audit_text(assumptions, critiques))

    output_files = [
        abs_path(spec_txt),
        abs_path(profile_formula_txt),
        abs_path(sem_json),
        abs_path(sem_coeff_csv),
        abs_path(sem_fit_json),
        abs_path(ols_txt),
        abs_path(ols_params_csv),
        abs_path(bootstrap_csv),
        abs_path(exploratory_csv),
        abs_path(weight_table_csv),
        abs_path(profile_index_csv),
        abs_path(contribution_breakdown_csv),
        abs_path(profile_summary_copy_csv),
        abs_path(profile_wide_copy_csv),
        abs_path(ridge_coeff_csv),
        abs_path(ridge_summary_csv),
        abs_path(conditional_artifact_json),
        abs_path(report_txt),
        abs_path(assumptions_json),
        abs_path(critiques_json),
        abs_path(methodology_txt),
    ]
    if latent_scores_csv.exists():
        output_files.append(abs_path(latent_scores_csv))

    manifest = StageArtifactManifest(
        stage_id="06",
        stage_name="construct_structural_equation_model",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(report_txt),
        output_files=output_files,
        record_count=len(profile_df),
        metadata={
            "sem_converged": sem_result.converged,
            "n_coefficients": len(sem_result.coefficients),
            "primary_moderator": config.primary_moderator,
            "bootstrap_samples": config.bootstrap_samples,
            "analysis_mode": analysis_mode,
            "indicator_columns": indicator_columns,
            "structural_terms": structural_terms,
            "multivariate_formula": multivariate_formula,
            "n_profile_moderators": len(profile_terms),
            "conditional_susceptibility_attack_leaves": conditional_fit.artifact.attack_leaves,
            "conditional_susceptibility_opinion_leaves": conditional_fit.artifact.opinion_leaves,
            "conditional_susceptibility_tasks": [task.task_key for task in conditional_fit.artifact.task_models],
        },
    )

    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 06 - SEM construction")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-moderator", default="posthoc_profile_susceptibility_index")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)

    config = Stage06Config(
        stage_name="construct_structural_equation_model",
        run_id=args.run_id,
        seed=args.seed,
        primary_moderator=args.primary_moderator,
        bootstrap_samples=args.bootstrap_samples,
    )

    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 06 completed: %s profiles", manifest.record_count)


if __name__ == "__main__":
    main()
