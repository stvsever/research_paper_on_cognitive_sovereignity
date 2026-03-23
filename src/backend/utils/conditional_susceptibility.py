from __future__ import annotations

"""
Technical overview
------------------
This module implements the analysis-facing conditional susceptibility index used
after a run has completed. The key design choice is that susceptibility is not
treated as a fixed trait known before simulation. Instead, it is estimated
post hoc conditional on the exact target set that was modeled:

    target set = {(attack_leaf, opinion_leaf)}

For each attack-opinion task, the module fits a regularized profile-only ridge
model on observed attacked effectivity, currently using absolute post-minus-
baseline opinion shift as the default outcome. Each task-specific model
produces a fitted mapping:

    profile features -> predicted attacked effectivity for that task

Those task-level predictions are then aggregated back to the profile level
using reliability weights derived from sample size and cross-validated error.
The result is a reusable artifact that can later score new pseudoprofiles under
the same modeled attack/opinion configuration.

Why this module exists:
- to keep susceptibility estimation separate from prompt-time heuristics
- to make the estimator conditional on the configured ATTACK and OPINION leaves
- to provide a stable artifact interface so later nonlinear models can replace
  ridge fits without changing downstream scoring APIs

Key public entry points:
- fit_conditional_susceptibility_index(...)
- score_profiles_with_conditional_artifact(...)
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.backend.utils.schemas import (
    ConditionalSusceptibilityArtifact,
    ConditionalSusceptibilityTaskModel,
)


LEGACY_EXCLUDED_FEATURE_COLUMNS = {
    "profile_cont_heuristic_shift_sensitivity_proxy",
    "profile_cont_resilience_index",
}


@dataclass
class ConditionalSusceptibilityFitResult:
    artifact: ConditionalSusceptibilityArtifact
    task_coefficients: pd.DataFrame
    task_summary: pd.DataFrame
    profile_scores: pd.DataFrame
    contribution_breakdown: pd.DataFrame


def _kfold_indices(n_obs: int, seed: int, n_splits: int = 5) -> List[np.ndarray]:
    n_splits = max(2, min(n_splits, n_obs))
    indices = np.arange(n_obs)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return [fold for fold in np.array_split(indices, n_splits) if len(fold) > 0]


def _ridge_fit_matrix(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + alpha * penalty) @ (x.T @ y)


def _cross_validated_ridge(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    alpha_grid: Sequence[float] | None = None,
) -> Tuple[np.ndarray, float, float]:
    alpha_grid = list(alpha_grid or np.logspace(-3, 3, 25))
    folds = _kfold_indices(len(y), seed=seed, n_splits=5)

    best_alpha = float(alpha_grid[0])
    best_cv_mse = float("inf")
    for alpha in alpha_grid:
        fold_mses: List[float] = []
        for fold in folds:
            mask = np.ones(len(y), dtype=bool)
            mask[fold] = False
            beta = _ridge_fit_matrix(x[mask], y[mask], alpha)
            preds = x[fold] @ beta
            fold_mses.append(float(np.mean((y[fold] - preds) ** 2)))
        cv_mse = float(np.mean(fold_mses))
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_alpha = float(alpha)

    beta = _ridge_fit_matrix(x, y, best_alpha)
    return beta, best_alpha, best_cv_mse


def _default_feature_columns(
    df: pd.DataFrame,
    excluded_columns: Iterable[str] | None = None,
) -> List[str]:
    excluded = set(excluded_columns or []) | LEGACY_EXCLUDED_FEATURE_COLUMNS
    columns: List[str] = []
    for column in sorted(df.columns):
        is_continuous = column.startswith("profile_cont_")
        is_categorical = column.startswith("profile_cat__")
        if not (is_continuous or is_categorical):
            continue
        if column in excluded:
            continue
        if df[column].nunique(dropna=True) <= 1:
            continue
        columns.append(column)
    return columns


def _fit_feature_scaler(
    unique_profiles_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, float], List[str], List[str]]:
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    continuous_columns: List[str] = []
    categorical_columns: List[str] = []

    for column in feature_columns:
        if column.startswith("profile_cont_"):
            continuous_columns.append(column)
            means[column] = float(unique_profiles_df[column].astype(float).mean())
            std = float(unique_profiles_df[column].astype(float).std(ddof=0))
            stds[column] = std if std > 0.0 else 1.0
        else:
            categorical_columns.append(column)

    return means, stds, continuous_columns, categorical_columns


def _transform_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    feature_means: Dict[str, float],
    feature_stds: Dict[str, float],
    continuous_columns: Sequence[str],
) -> pd.DataFrame:
    transformed = pd.DataFrame(index=df.index)
    continuous_set = set(continuous_columns)
    for column in feature_columns:
        if column not in df.columns:
            transformed[column] = 0.0
            continue
        values = df[column].astype(float)
        if column in continuous_set:
            transformed[column] = (values - float(feature_means[column])) / float(feature_stds[column])
        else:
            transformed[column] = values.fillna(0.0)
    return transformed


def _task_key(attack_leaf: str, opinion_leaf: str) -> str:
    return f"{attack_leaf} || {opinion_leaf}"


def fit_conditional_susceptibility_index(
    long_df: pd.DataFrame,
    *,
    outcome_metric: str = "abs_delta_score",
    feature_columns: Sequence[str] | None = None,
    excluded_feature_columns: Sequence[str] | None = None,
    seed: int = 42,
    alpha_grid: Sequence[float] | None = None,
    min_rows_per_task: int = 8,
) -> ConditionalSusceptibilityFitResult:
    required_columns = {"profile_id", "attack_leaf", "opinion_leaf", outcome_metric}
    missing = sorted(required_columns - set(long_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for conditional susceptibility fit: {missing}")

    attacked_df = long_df.copy()
    attacked_df = attacked_df.loc[attacked_df[outcome_metric].notna()].copy()
    if attacked_df.empty:
        raise ValueError("No attacked rows available for conditional susceptibility fitting.")

    feature_columns = list(feature_columns or _default_feature_columns(attacked_df, excluded_feature_columns))
    if not feature_columns:
        raise ValueError("No usable profile feature columns available for conditional susceptibility fitting.")

    unique_profiles_df = attacked_df[["profile_id", *feature_columns]].drop_duplicates(subset=["profile_id"]).reset_index(drop=True)
    feature_means, feature_stds, continuous_columns, categorical_columns = _fit_feature_scaler(
        unique_profiles_df=unique_profiles_df,
        feature_columns=feature_columns,
    )

    transformed_profiles = _transform_features(
        df=unique_profiles_df,
        feature_columns=feature_columns,
        feature_means=feature_means,
        feature_stds=feature_stds,
        continuous_columns=continuous_columns,
    )
    transformed_profiles.insert(0, "profile_id", unique_profiles_df["profile_id"])

    profile_lookup = transformed_profiles.set_index("profile_id")
    task_rows: List[Dict[str, object]] = []
    coeff_rows: List[Dict[str, object]] = []
    task_models: List[ConditionalSusceptibilityTaskModel] = []

    grouped = attacked_df.groupby(["attack_leaf", "opinion_leaf"], dropna=False)
    for offset, ((attack_leaf, opinion_leaf), task_df) in enumerate(grouped):
        if len(task_df) < min_rows_per_task:
            continue
        x_df = profile_lookup.loc[task_df["profile_id"]].reset_index(drop=True)
        x = np.column_stack(
            [
                np.ones(len(x_df), dtype=float),
                *[x_df[column].astype(float).to_numpy() for column in feature_columns],
            ]
        )
        y = task_df[outcome_metric].astype(float).to_numpy()
        beta, alpha, cv_mse = _cross_validated_ridge(
            x=x,
            y=y,
            seed=seed + offset,
            alpha_grid=alpha_grid,
        )
        reliability_weight = float(len(task_df) / max(cv_mse, 1e-6))
        key = _task_key(str(attack_leaf), str(opinion_leaf))
        coefficients = {column: float(beta[idx + 1]) for idx, column in enumerate(feature_columns)}

        task_models.append(
            ConditionalSusceptibilityTaskModel(
                task_key=key,
                attack_leaf=str(attack_leaf),
                opinion_leaf=str(opinion_leaf),
                outcome_metric=outcome_metric,
                n_obs=int(len(task_df)),
                alpha=float(alpha),
                cv_mse=float(cv_mse),
                reliability_weight=reliability_weight,
                intercept=float(beta[0]),
                coefficients=coefficients,
            )
        )

        task_rows.append(
            {
                "task_key": key,
                "attack_leaf": attack_leaf,
                "opinion_leaf": opinion_leaf,
                "outcome_metric": outcome_metric,
                "n_obs": int(len(task_df)),
                "alpha": float(alpha),
                "cv_mse": float(cv_mse),
                "reliability_weight_raw": reliability_weight,
            }
        )
        coeff_rows.append(
            pd.DataFrame(
                {
                    "task_key": key,
                    "attack_leaf": attack_leaf,
                    "opinion_leaf": opinion_leaf,
                    "outcome_metric": outcome_metric,
                    "term": ["Intercept", *feature_columns],
                    "estimate": [float(beta[0]), *[float(coefficients[column]) for column in feature_columns]],
                    "n_obs": int(len(task_df)),
                    "alpha": float(alpha),
                    "cv_mse": float(cv_mse),
                }
            )
        )

    if not task_models:
        raise ValueError("No task-specific models could be fit for the configured attack/opinion target set.")

    task_summary = pd.DataFrame(task_rows)
    task_summary["reliability_weight"] = (
        task_summary["reliability_weight_raw"] / float(task_summary["reliability_weight_raw"].sum())
    )
    task_summary = task_summary.drop(columns=["reliability_weight_raw"])

    normalized_weights = {
        row["task_key"]: float(row["reliability_weight"])
        for row in task_summary.to_dict(orient="records")
    }
    for model in task_models:
        model.reliability_weight = normalized_weights[model.task_key]

    coeff_df = pd.concat(coeff_rows, ignore_index=True)
    score_df, breakdown_df = score_profiles_with_conditional_artifact(
        unique_profiles_df,
        ConditionalSusceptibilityArtifact(
            outcome_metric=outcome_metric,
            attack_leaves=sorted({str(value) for value in attacked_df["attack_leaf"].dropna().unique().tolist()}),
            opinion_leaves=sorted({str(value) for value in attacked_df["opinion_leaf"].dropna().unique().tolist()}),
            feature_columns=list(feature_columns),
            continuous_feature_columns=continuous_columns,
            categorical_feature_columns=categorical_columns,
            excluded_feature_columns=sorted(set(excluded_feature_columns or []) | LEGACY_EXCLUDED_FEATURE_COLUMNS),
            feature_means=feature_means,
            feature_stds=feature_stds,
            task_models=task_models,
            notes=[
                "This artifact is valid only for the attack leaves and opinion leaves recorded in the target set metadata.",
                "The conditional susceptibility index is computed from fitted task-specific ridge models and is therefore model-based rather than directly observed.",
            ],
        ),
    )

    return ConditionalSusceptibilityFitResult(
        artifact=ConditionalSusceptibilityArtifact(
            outcome_metric=outcome_metric,
            attack_leaves=sorted({str(value) for value in attacked_df["attack_leaf"].dropna().unique().tolist()}),
            opinion_leaves=sorted({str(value) for value in attacked_df["opinion_leaf"].dropna().unique().tolist()}),
            feature_columns=list(feature_columns),
            continuous_feature_columns=continuous_columns,
            categorical_feature_columns=categorical_columns,
            excluded_feature_columns=sorted(set(excluded_feature_columns or []) | LEGACY_EXCLUDED_FEATURE_COLUMNS),
            feature_means=feature_means,
            feature_stds=feature_stds,
            task_models=task_models,
            notes=[
                "This artifact is valid only for the attack leaves and opinion leaves recorded in the target set metadata.",
                "The conditional susceptibility index is computed from fitted task-specific ridge models and is therefore model-based rather than directly observed.",
            ],
        ),
        task_coefficients=coeff_df,
        task_summary=task_summary,
        profile_scores=score_df,
        contribution_breakdown=breakdown_df,
    )


def score_profiles_with_conditional_artifact(
    profile_df: pd.DataFrame,
    artifact: ConditionalSusceptibilityArtifact,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "profile_id" not in profile_df.columns:
        raise ValueError("profile_df must contain profile_id for conditional susceptibility scoring.")

    unique_profile_df = profile_df[["profile_id", *artifact.feature_columns]].drop_duplicates(subset=["profile_id"]).reset_index(drop=True)
    transformed = _transform_features(
        df=unique_profile_df,
        feature_columns=artifact.feature_columns,
        feature_means=artifact.feature_means,
        feature_stds=artifact.feature_stds,
        continuous_columns=artifact.continuous_feature_columns,
    )
    transformed.insert(0, "profile_id", unique_profile_df["profile_id"])

    score_df = unique_profile_df[["profile_id"]].copy()
    score_df["conditional_target_attack_count"] = len(artifact.attack_leaves)
    score_df["conditional_target_opinion_count"] = len(artifact.opinion_leaves)
    score_df["conditional_target_task_count"] = len(artifact.task_models)

    raw_score = np.zeros(len(score_df), dtype=float)
    breakdown_rows: List[Dict[str, object]] = []

    for task_model in artifact.task_models:
        task_contribution = np.full(len(score_df), float(task_model.intercept), dtype=float)
        for column in artifact.feature_columns:
            beta = float(task_model.coefficients.get(column, 0.0))
            task_contribution = task_contribution + transformed[column].astype(float).to_numpy() * beta
        weighted_contribution = task_contribution * float(task_model.reliability_weight)
        task_slug = (
            task_model.task_key.lower()
            .replace(" ", "_")
            .replace(">", "_")
            .replace("|", "_")
            .replace("/", "_")
        )
        score_df[f"predicted_effectivity__{task_slug}"] = task_contribution
        score_df[f"weighted_effectivity__{task_slug}"] = weighted_contribution
        raw_score = raw_score + weighted_contribution

        for idx, profile_id in enumerate(score_df["profile_id"].tolist()):
            breakdown_rows.append(
                {
                    "profile_id": profile_id,
                    "component_type": "task",
                    "component_name": task_model.task_key,
                    "component_key": task_slug,
                    "attack_leaf": task_model.attack_leaf,
                    "opinion_leaf": task_model.opinion_leaf,
                    "contribution": float(weighted_contribution[idx]),
                    "reliability_weight": float(task_model.reliability_weight),
                }
            )

    score_df["conditional_susceptibility_raw_score"] = raw_score
    score_df["susceptibility_index_pct"] = pd.Series(raw_score, index=score_df.index).rank(method="average", pct=True) * 100.0

    signed_weight_lookup: Dict[str, float] = {}
    for task_model in artifact.task_models:
        for column, beta in task_model.coefficients.items():
            signed_weight_lookup[column] = signed_weight_lookup.get(column, 0.0) + float(beta) * float(task_model.reliability_weight)

    for column in artifact.feature_columns:
        values = transformed[column].astype(float).to_numpy()
        total_beta_weight = signed_weight_lookup.get(column, 0.0)
        contribution = values * total_beta_weight
        score_df[f"contribution__{column}"] = contribution
        for idx, profile_id in enumerate(score_df["profile_id"].tolist()):
            breakdown_rows.append(
                {
                    "profile_id": profile_id,
                    "component_type": "feature",
                    "component_name": column,
                    "component_key": column,
                    "attack_leaf": None,
                    "opinion_leaf": None,
                    "contribution": float(contribution[idx]),
                    "reliability_weight": float(total_beta_weight),
                }
            )

    return (
        score_df.sort_values("susceptibility_index_pct", ascending=False).reset_index(drop=True),
        pd.DataFrame(breakdown_rows),
    )


def build_conditional_weight_table(
    artifact: ConditionalSusceptibilityArtifact,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for column in artifact.feature_columns:
        signed_effect = 0.0
        abs_effect = 0.0
        n_tasks = 0
        for task_model in artifact.task_models:
            if column not in task_model.coefficients:
                continue
            beta = float(task_model.coefficients[column])
            weight = float(task_model.reliability_weight)
            signed_effect += weight * beta
            abs_effect += weight * abs(beta)
            n_tasks += 1
        rows.append(
            {
                "term": column,
                "weighted_mean_estimate": signed_effect,
                "weighted_mean_abs_estimate": abs_effect,
                "n_tasks": n_tasks,
            }
        )
    weight_df = pd.DataFrame(rows)
    if weight_df.empty:
        return weight_df
    denom = float(weight_df["weighted_mean_abs_estimate"].sum()) or 1.0
    weight_df["normalized_weight_pct"] = (weight_df["weighted_mean_abs_estimate"] / denom) * 100.0
    return weight_df.sort_values(["normalized_weight_pct", "term"], ascending=[False, True]).reset_index(drop=True)
