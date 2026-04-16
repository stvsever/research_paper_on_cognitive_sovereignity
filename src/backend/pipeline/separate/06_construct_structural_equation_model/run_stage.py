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
    HierarchicalDecomposition,
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
    """Prefer adversarially-aligned indicators (run_7+); fall back to abs_delta."""
    adversarial = sorted(
        column
        for column in profile_df.columns
        if column.startswith("adversarial_delta_indicator__") and not column.endswith("_z")
    )
    if adversarial:
        return adversarial
    return sorted(
        column
        for column in profile_df.columns
        if column.startswith("abs_delta_indicator__") and not column.endswith("_z")
    )


def _primary_outcome_column(profile_df: pd.DataFrame) -> str:
    """Return the profile-level aggregate outcome column for OLS/SEM."""
    if "mean_adversarial_effectivity" in profile_df.columns and profile_df["mean_adversarial_effectivity"].notna().any():
        return "mean_adversarial_effectivity"
    return "mean_abs_delta_score"


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


def _compute_icc(long_df: pd.DataFrame, outcome: str = "abs_delta_score",
                 cluster_col: str = "profile_id") -> Dict[str, float]:
    """Compute ICC(1) for a nested outcome.

    ICC(1) = σ²_between / (σ²_between + σ²_within)
    Estimated via one-way random-effects ANOVA decomposition.
    """
    groups = [g[outcome].values for _, g in long_df.groupby(cluster_col) if len(g) > 0]
    if len(groups) < 2:
        return {"icc1": np.nan, "sigma2_between": np.nan, "sigma2_within": np.nan, "n_clusters": 0}

    k = len(groups)
    ns = np.array([len(g) for g in groups])
    N = ns.sum()
    grand_mean = long_df[outcome].mean()

    ss_between = float(sum(n * (g.mean() - grand_mean) ** 2 for g, n in zip(groups, ns)))
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups))

    ms_between = ss_between / max(1, k - 1)
    ms_within = ss_within / max(1, N - k)

    n0 = (N - (ns ** 2).sum() / N) / max(1, k - 1)
    sigma2_between = max(0.0, (ms_between - ms_within) / max(1e-12, n0))
    sigma2_within = ms_within
    icc1 = sigma2_between / (sigma2_between + sigma2_within) if (sigma2_between + sigma2_within) > 0 else 0.0

    return {
        "icc1": round(icc1, 4),
        "sigma2_between": round(sigma2_between, 4),
        "sigma2_within": round(sigma2_within, 4),
        "n_clusters": k,
        "mean_cluster_size": round(float(ns.mean()), 2),
    }


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
    t = term.lower()
    # Demographics
    if "chronological_age" in t or "age_years" in t:
        return "Demographics: Age"
    if "sex_" in t or "profile_cat_sex" in t:
        return "Demographics: Sex"
    if "education_level" in t or ("education" in t and "level" in t):
        return "Demographics: Education"
    if "news_diet" in t:
        return "Demographics: News Diet"
    # Big Five
    if "neuroticism" in t:
        return "Big Five: Neuroticism"
    if "openness_to_experience" in t or ("openness" in t and "experience" in t):
        return "Big Five: Openness"
    if "conscientiousness" in t:
        return "Big Five: Conscientiousness"
    if "extraversion" in t:
        return "Big Five: Extraversion"
    if "agreeableness" in t:
        return "Big Five: Agreeableness"
    # Dual Process Inventory (run_9 ontology)
    if "dual_process" in t:
        return "Dual Process"
    # Digital Literacy Inventory (run_9 ontology)
    if "digital_literacy" in t:
        return "Digital Literacy"
    # Political Engagement Inventory (run_9 ontology)
    if "political_engagement_inventory_institutional_trust" in t:
        return "Political Engagement: Institutional Trust"
    if "political_engagement_inventory_ideological_identity" in t:
        return "Political Engagement: Ideology"
    if "political_engagement_inventory_political_interest" in t:
        return "Political Engagement: Interest"
    if "political_engagement_inventory_collective_efficacy" in t:
        return "Political Engagement: Efficacy"
    if "political_engagement_inventory" in t:
        return "Political Engagement"
    # Political Psychology (run_10+ ontology)
    if "political_psychology_institutional_trust" in t or "institutional_trust" in t:
        return "Political Psychology: Institutional Trust"
    if "political_psychology_ideological_positioning" in t or "ideological_positioning" in t:
        return "Political Psychology: Ideology"
    if "political_psychology_political_engagement" in t:
        return "Political Psychology: Engagement"
    if "political_psychology" in t:
        return "Political Psychology"
    # Socioeconomic Status (run_10+ ontology)
    if "socioeconomic_status_employment_type" in t or "employed_" in t or "unemployed" in t or "retired" in t or "employment_type" in t:
        return "Socioeconomic Status: Employment"
    if (
        "socioeconomic_status_economic_standing" in t
        or "household_income" in t
        or "economic_anxiety" in t
        or "financial_security" in t
        or "upward_mobility" in t
        or "subjective_class" in t
        or "economic_standing" in t
    ):
        return "Socioeconomic Status: Economic"
    if "socioeconomic" in t:
        return "Socioeconomic Status"
    # Social Context (run_10+ ontology)
    if (
        "social_context_online_behavior" in t
        or "social_media_hours" in t
        or "echo_chamber" in t
        or "online_political_discussion" in t
        or "platform_primary_type" in t
        or ("platform" in t and "dominant" in t)
    ):
        return "Social Context: Online Behavior"
    if (
        "social_context_social_capital" in t
        or "interpersonal_trust" in t
        or "social_network_diversity" in t
        or "community_belonging" in t
        or "social_isolation" in t
    ):
        return "Social Context: Social Capital"
    if "social_context" in t:
        return "Social Context"
    return "Other"


def _dynamic_profile_terms(df: pd.DataFrame) -> List[str]:
    """Return all profile feature columns with variance, excluding synthetic proxies."""
    _exclude = {
        "profile_cont_heuristic_shift_sensitivity_proxy",
        "profile_cont_resilience_index",
    }
    terms: List[str] = []
    for col in sorted(df.columns):
        if col in _exclude:
            continue
        if not (col.startswith("profile_cont_") or col.startswith("profile_cat__")):
            continue
        if col.endswith("_z"):  # skip pre-standardised duplicates
            continue
        if df[col].nunique(dropna=True) > 1:
            terms.append(col)
    return terms


def _fit_elastic_net(
    df: pd.DataFrame,
    outcome: str,
    feature_terms: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    """Cross-validated Elastic Net on all profile features.

    Addresses the fundamental weakness of the aggregate OLS: using only 8 hardcoded
    Big Five predictors when 100+ profile features are available. ElasticNetCV handles
    high-dimensional small-n by regularisation, selecting informative features and
    shrinking noise to zero. CV-R² is reported as an honest fit estimate.
    """
    from sklearn.linear_model import ElasticNetCV  # type: ignore
    from sklearn.model_selection import KFold, cross_val_score  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    x_raw = df[list(feature_terms)].astype(float).fillna(0.0).to_numpy()
    y = df[outcome].astype(float).to_numpy()

    col_std = np.std(x_raw, axis=0)
    valid_mask = col_std > 1e-8
    terms_used = [t for t, v in zip(feature_terms, valid_mask) if v]
    x = x_raw[:, valid_mask]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    cv_inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        cv=cv_inner,
        random_state=seed,
        max_iter=20000,
        alphas=50,
        n_jobs=-1,
    )
    enet.fit(x_scaled, y)

    y_hat = enet.predict(x_scaled)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_train = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    cv_outer = KFold(n_splits=5, shuffle=True, random_state=seed + 1)
    cv_r2_scores = cross_val_score(
        ElasticNetCV(
            l1_ratio=float(enet.l1_ratio_),
            cv=5,
            random_state=seed,
            max_iter=20000,
            alphas=50,
        ),
        x_scaled,
        y,
        cv=cv_outer,
        scoring="r2",
        n_jobs=-1,
    )
    cv_r2 = float(np.mean(cv_r2_scores))
    cv_r2_std = float(np.std(cv_r2_scores))

    coeff_df = pd.DataFrame(
        {
            "term": terms_used,
            "label": [_pretty_moderator_label(t) for t in terms_used],
            "ontology_group": [_moderator_group(t) for t in terms_used],
            "elastic_net_estimate": enet.coef_,
        }
    )
    selected_df = (
        coeff_df[coeff_df["elastic_net_estimate"].abs() > 1e-8]
        .copy()
        .sort_values("elastic_net_estimate", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )

    return {
        "r2_train": r2_train,
        "cv_r2": cv_r2,
        "cv_r2_std": cv_r2_std,
        "alpha": float(enet.alpha_),
        "l1_ratio": float(enet.l1_ratio_),
        "n_features_total": len(terms_used),
        "n_features_selected": int((np.abs(enet.coef_) > 1e-8).sum()),
        "coeff_df": coeff_df,
        "selected_df": selected_df,
    }


def _fit_random_forest(
    df: pd.DataFrame,
    outcome: str,
    feature_terms: Sequence[str],
    seed: int,
    n_estimators: int = 500,
) -> Dict[str, Any]:
    """Random Forest regression for non-linear moderation detection.

    OOB R² provides an honest fit estimate without a separate test set.
    Permutation importance (n_repeats=50) is preferred over MDI because it
    accounts for correlated features and is invariant to feature scale.
    """
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.inspection import permutation_importance as sk_perm_importance  # type: ignore

    x_raw = df[list(feature_terms)].astype(float).fillna(0.0).to_numpy()
    y = df[outcome].astype(float).to_numpy()

    col_std = np.std(x_raw, axis=0)
    valid_mask = col_std > 1e-8
    terms_used = [t for t, v in zip(feature_terms, valid_mask) if v]
    x = x_raw[:, valid_mask]

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=3,
        oob_score=True,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(x, y)

    perm = sk_perm_importance(rf, x, y, n_repeats=50, random_state=seed, n_jobs=-1)

    importance_df = (
        pd.DataFrame(
            {
                "term": terms_used,
                "label": [_pretty_moderator_label(t) for t in terms_used],
                "ontology_group": [_moderator_group(t) for t in terms_used],
                "permutation_importance_mean": perm.importances_mean,
                "permutation_importance_std": perm.importances_std,
                "mdi_importance": rf.feature_importances_,
            }
        )
        .sort_values("permutation_importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "oob_r2": float(rf.oob_score_),
        "n_estimators": n_estimators,
        "n_features": len(terms_used),
        "importance_df": importance_df,
    }


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
    cluster_col: str = "profile_id",
) -> pd.DataFrame:
    """Cluster bootstrap OLS: resamples at the profile level to respect nesting.

    When *cluster_col* is present in *df*, entire clusters are resampled
    (preserving within-cluster dependence).  Falls back to IID pairs
    bootstrap when the column is absent.
    """
    rng = np.random.default_rng(seed)
    use_cluster = cluster_col in df.columns
    if use_cluster:
        cluster_ids = df[cluster_col].unique()
    records: List[Dict[str, float]] = []
    for _ in range(n_bootstrap):
        if use_cluster:
            sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            sample = pd.concat(
                [df[df[cluster_col] == cid] for cid in sampled], ignore_index=True
            )
        else:
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
    outcome: str = "mean_abs_delta_score",
) -> pd.DataFrame:
    multivariate_lookup = {row["term"]: row for row in multivariate_params.to_dict(orient="records")}
    rows: List[Dict[str, object]] = []

    for term in candidate_terms:
        formula = _build_formula(outcome, [term, *control_terms])
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

    # Benjamini-Hochberg FDR correction for multiple comparisons
    for p_col, q_col in [
        ("univariate_p_value", "univariate_q_value"),
        ("multivariate_p_value", "multivariate_q_value"),
    ]:
        if p_col in comparison.columns:
            pvals = comparison[p_col].values
            valid = ~np.isnan(pvals)
            q = np.full_like(pvals, np.nan)
            if valid.sum() > 0:
                ranked = np.argsort(np.argsort(pvals[valid])) + 1
                m = valid.sum()
                q[valid] = np.minimum(1.0, pvals[valid] * m / ranked)
                # Enforce monotonicity (step-up)
                sorted_idx = np.argsort(pvals[valid])[::-1]
                q_sorted = q[valid][sorted_idx]
                for i in range(1, len(q_sorted)):
                    q_sorted[i] = min(q_sorted[i], q_sorted[i - 1])
                q[valid] = q_sorted[np.argsort(sorted_idx)]
            comparison[q_col] = q

    # Cohen's d effect size (standardised mean difference proxy from z-scored predictors)
    for prefix in ("univariate", "multivariate"):
        est_col = f"{prefix}_estimate"
        se_col = f"{prefix}_std_error"
        if est_col in comparison.columns and se_col in comparison.columns:
            comparison[f"{prefix}_cohens_d"] = (
                comparison[est_col] / comparison[se_col].replace(0, np.nan)
            ).round(4)

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
    primary_outcome: str = "mean_abs_delta_score",
    hierarchical_decomposition: Any = None,
    enet_result: Dict[str, Any] | None = None,
    rf_result: Dict[str, Any] | None = None,
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

    adv_text = "n/a"
    if "adversarial_effectivity" in long_df.columns:
        adv_vals = long_df["adversarial_effectivity"].dropna()
        if len(adv_vals) > 0:
            pos_pct = float((adv_vals > 0).mean() * 100.0)
            adv_text = f"{float(adv_vals.mean()):.3f} (positive={pos_pct:.1f}%)"

    lines = [
        f"Moderation Report - {run_id}",
        "=========================",
        "",
        f"Profiles analyzed: {len(profile_df)}",
        f"Attacked opinion scenarios analyzed: {len(long_df)}",
        f"Repeated opinion indicators: {len(indicator_columns)}",
        f"Primary effectivity outcome: {primary_outcome}",
        f"Mean adversarial effectivity: {adv_text}",
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
            f"Outcome: {primary_outcome}",
            f"Formula: {multivariate_formula}",
        ]
    )

    if hierarchical_decomposition is not None:
        lines.extend(["", "Hierarchical Variance Decomposition (Conditional Susceptibility)", "----------------------------------------------------------------"])
        lines.append(f"Full model CV-R\u00b2: {hierarchical_decomposition.full_model_cv_r2:.4f}")
        sorted_groups = sorted(
            hierarchical_decomposition.group_marginal_r2.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        for group, mar_r2 in sorted_groups:
            rel_pct = hierarchical_decomposition.group_relative_importance_pct.get(group, 0.0)
            lines.append(f"  {group}: marginal_R\u00b2={mar_r2:.4f}, relative_importance={rel_pct:.1f}%")

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
        lines.extend(["", "Empirical Profile Susceptibility Index", "--------------------------------------"])
        lines.append(f"The post hoc susceptibility index is the percentile-ranked profile-only linear predictor under the configured attack/opinion target set. Primary outcome: {primary_outcome}.")
        for row in top_profiles.to_dict(orient="records"):
            adv_val = row.get("mean_adversarial_effectivity")
            adv_str = f", mean_adversarial_eff={adv_val:.2f}" if adv_val is not None else ""
            lines.append(
                f"{row['profile_id']}: susceptibility_index_pct={row['susceptibility_index_pct']:.2f}, mean_abs_delta={row.get('mean_abs_delta_score', float('nan')):.2f}{adv_str}"
            )

    if not top_weights.empty:
        lines.extend(["", "Moderator Weight Decomposition", "------------------------------"])
        for row in top_weights.to_dict(orient="records"):
            lines.append(
                f"{row['moderator_label']} [{row['ontology_group']}]: est={row['estimate']:.4f}, normalized_weight_pct={row['normalized_weight_pct']:.2f}"
            )

    # Elastic Net results
    if enet_result is not None:
        lines.extend(
            [
                "",
                "Elastic Net Moderation Model (Full Feature Set)",
                "-----------------------------------------------",
                f"Features used: {enet_result.get('n_features_total', 'n/a')} (all available profile_cont_ / profile_cat__ columns)",
                f"Features selected (|coef|>0): {enet_result.get('n_features_selected', 'n/a')}",
                f"Best alpha: {enet_result.get('alpha', float('nan')):.5f}",
                f"Best l1_ratio: {enet_result.get('l1_ratio', float('nan')):.2f}  (1.0=LASSO, 0.0=Ridge)",
                f"Train R²: {enet_result.get('r2_train', float('nan')):.4f}",
                f"CV-R² (nested 5-fold): {enet_result.get('cv_r2', float('nan')):.4f} ± {enet_result.get('cv_r2_std', float('nan')):.4f}",
                "Top selected features:",
            ]
        )
        sel = enet_result.get("selected_df", pd.DataFrame())
        for row in sel.head(10).to_dict(orient="records"):
            lines.append(
                f"  {row.get('label', row.get('term'))}: coef={row['elastic_net_estimate']:.4f}  [{row.get('ontology_group', '')}]"
            )

    # Random Forest results
    if rf_result is not None:
        lines.extend(
            [
                "",
                "Random Forest Moderation Model (Non-linear, Full Feature Set)",
                "--------------------------------------------------------------",
                f"OOB R²: {rf_result.get('oob_r2', float('nan')):.4f}  (unbiased held-out estimate)",
                f"Trees: {rf_result.get('n_estimators', 'n/a')}, Features: {rf_result.get('n_features', 'n/a')}",
                "Top features by permutation importance:",
            ]
        )
        imp = rf_result.get("importance_df", pd.DataFrame())
        for row in imp.head(10).to_dict(orient="records"):
            lines.append(
                f"  {row.get('label', row.get('term'))}: perm_imp={row['permutation_importance_mean']:.4f} ± {row['permutation_importance_std']:.4f}  [{row.get('ontology_group', '')}]"
            )

    lines.extend(
        [
            "",
            "OLS Supplement (Conventional Benchmark — Big Five + Age + Sex Only)",
            "---------------------------------------------------------------------",
            "NOTE: The following OLS uses only the hardcoded Big Five domain means, age, and sex.",
            "It serves as a conventional benchmark. The Elastic Net and Random Forest above use",
            "the full feature set and are the primary moderation estimators.",
            "",
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
        raise RuntimeError("SEM stage is designed for attacked-only profile-panel data (attack_ratio=1.0).")

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
    primary_outcome = _primary_outcome_column(profile_df)

    sem_result, factor_scores = _fit_sem(profile_df, indicator_columns=indicator_columns, structural_terms=structural_terms)

    multivariate_terms = [*profile_terms, *control_terms]
    multivariate_formula = _build_formula(primary_outcome, multivariate_terms)
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
        outcome=primary_outcome,
    )
    # Use all available profile features (means + facets + demographics) for the
    # conditional susceptibility index — richer than SEM/OLS which use means only.
    # feature_columns=None triggers _default_feature_columns which picks all profile_cont_/profile_cat__ columns.
    conditional_outcome = "adversarial_effectivity" if "adversarial_effectivity" in long_df.columns and long_df["adversarial_effectivity"].notna().any() else "abs_delta_score"
    conditional_fit = fit_conditional_susceptibility_index(
        long_df=long_df,
        outcome_metric=conditional_outcome,
        feature_columns=None,  # use all available profile features
        excluded_feature_columns=[
            "profile_cont_heuristic_shift_sensitivity_proxy",
            "profile_cont_resilience_index",
        ],
        seed=config.seed,
        compute_hierarchy=True,
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

    merge_cols = ["profile_id", "mean_abs_delta_score", "mean_signed_delta_score"]
    if "mean_adversarial_effectivity" in profile_df.columns:
        merge_cols.append("mean_adversarial_effectivity")
    profile_index_df = conditional_fit.profile_scores.merge(
        profile_df[merge_cols],
        on="profile_id",
        how="left",
    )
    obs_col = "mean_adversarial_effectivity" if "mean_adversarial_effectivity" in profile_index_df.columns else "mean_abs_delta_score"
    profile_index_df["observed_effectivity_pct"] = (
        profile_index_df[obs_col].rank(method="average", pct=True) * 100.0
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

    # ------------------------------------------------------------------
    # Elastic Net + Random Forest on the full feature set
    # The aggregate OLS uses only 8 hardcoded Big Five / demographics
    # predictors.  The EN and RF models use every available profile
    # feature (~100 columns), which includes theoretically motivated
    # predictors (e.g. institutional trust, ideological identity, digital
    # literacy) that the OLS silently omits.
    # ------------------------------------------------------------------
    all_feature_terms = _dynamic_profile_terms(profile_df)
    LOGGER.info("Fitting Elastic Net on %d profile features ...", len(all_feature_terms))
    enet_result = _fit_elastic_net(
        df=profile_df,
        outcome=primary_outcome,
        feature_terms=all_feature_terms,
        seed=config.seed,
    )
    LOGGER.info(
        "Elastic Net: CV-R²=%.3f (±%.3f), α=%.4f, l1_ratio=%.2f, %d/%d features selected",
        enet_result["cv_r2"], enet_result["cv_r2_std"],
        enet_result["alpha"], enet_result["l1_ratio"],
        enet_result["n_features_selected"], enet_result["n_features_total"],
    )

    LOGGER.info("Fitting Random Forest on %d profile features ...", len(all_feature_terms))
    rf_result = _fit_random_forest(
        df=profile_df,
        outcome=primary_outcome,
        feature_terms=all_feature_terms,
        seed=config.seed,
    )
    LOGGER.info("Random Forest OOB R²=%.3f", rf_result["oob_r2"])

    # Merge EN and RF estimates into exploratory_table so the dashboard
    # forest plot can show the full-feature-set model alongside OLS.
    enet_lookup = enet_result["coeff_df"].set_index("term")["elastic_net_estimate"].to_dict()
    rf_lookup = rf_result["importance_df"].set_index("term")["permutation_importance_mean"].to_dict()
    if not exploratory_table.empty:
        exploratory_table["elastic_net_estimate"] = exploratory_table["moderator_column"].map(enet_lookup).astype(float)
        exploratory_table["rf_permutation_importance"] = exploratory_table["moderator_column"].map(rf_lookup).astype(float)

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
    enet_coeff_csv = out / "elastic_net_coefficients.csv"
    enet_selected_csv = out / "elastic_net_selected.csv"
    enet_summary_json = out / "elastic_net_summary.json"
    rf_importance_csv = out / "rf_feature_importance.csv"
    rf_summary_json = out / "rf_summary.json"

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
    if conditional_fit.hierarchical_decomposition is not None:
        hier = conditional_fit.hierarchical_decomposition
        write_json(
            out / "conditional_susceptibility_hierarchical_decomposition.json",
            {
                "full_model_cv_r2": hier.full_model_cv_r2,
                "group_marginal_r2": hier.group_marginal_r2,
                "group_relative_importance_pct": hier.group_relative_importance_pct,
                "task_group_r2": {k: v for k, v in hier.task_group_r2.items()},
                "notes": [
                    "marginal_r2 = full model CV-R2 minus CV-R2 of model with that group removed (leave-one-group-out).",
                    "Positive marginal_r2 means removing the group reduces predictive accuracy.",
                    "relative_importance_pct = |marginal_r2| / sum(|marginal_r2|) * 100.",
                ],
            },
        )
    if "latent_attack_effectivity_factor_score" in profile_df.columns:
        profile_df[["profile_id", "latent_attack_effectivity_factor_score"]].to_csv(latent_scores_csv, index=False)
    profile_summary_df.to_csv(profile_summary_copy_csv, index=False)
    profile_df.to_csv(profile_wide_copy_csv, index=False)
    enet_result["coeff_df"].to_csv(enet_coeff_csv, index=False)
    enet_result["selected_df"].to_csv(enet_selected_csv, index=False)
    rf_result["importance_df"].to_csv(rf_importance_csv, index=False)
    write_json(enet_summary_json, {
        "r2_train": enet_result["r2_train"],
        "cv_r2": enet_result["cv_r2"],
        "cv_r2_std": enet_result["cv_r2_std"],
        "alpha": enet_result["alpha"],
        "l1_ratio": enet_result["l1_ratio"],
        "n_features_total": enet_result["n_features_total"],
        "n_features_selected": enet_result["n_features_selected"],
        "note": (
            "CV-R² from nested 5-fold cross-validation (outer CV evaluates a fixed l1_ratio "
            "chosen by inner CV). Estimates on standardised features (StandardScaler). "
            "Addresses OLS limitation of using only hardcoded Big Five / demographic predictors."
        ),
    })
    write_json(rf_summary_json, {
        "oob_r2": rf_result["oob_r2"],
        "n_estimators": rf_result["n_estimators"],
        "n_features": rf_result["n_features"],
        "note": "OOB R² is an unbiased estimate of held-out accuracy (each tree scored on samples it never saw during training).",
    })

    # ICC computation for hierarchical nesting diagnostics
    icc_results: Dict[str, object] = {}
    for icc_outcome in ["abs_delta_score", "delta_score", "adversarial_effectivity"]:
        if icc_outcome in long_df.columns and long_df[icc_outcome].notna().sum() > 0:
            icc_results[icc_outcome] = _compute_icc(long_df, outcome=icc_outcome)
    if icc_results:
        write_json(out / "intraclass_correlation.json", icc_results)
        LOGGER.info("ICC results: %s", {k: v.get("icc1") for k, v in icc_results.items()})

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
            primary_outcome=primary_outcome,
            hierarchical_decomposition=conditional_fit.hierarchical_decomposition,
            enet_result=enet_result,
            rf_result=rf_result,
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
        abs_path(enet_coeff_csv),
        abs_path(enet_selected_csv),
        abs_path(enet_summary_json),
        abs_path(rf_importance_csv),
        abs_path(rf_summary_json),
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
            "elastic_net_cv_r2": enet_result["cv_r2"],
            "elastic_net_n_selected": enet_result["n_features_selected"],
            "elastic_net_n_total": enet_result["n_features_total"],
            "rf_oob_r2": rf_result["oob_r2"],
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
