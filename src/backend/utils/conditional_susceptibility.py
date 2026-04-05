from __future__ import annotations

"""
Technical overview
------------------
This module implements the analysis-facing conditional susceptibility index used
after a run has completed. The key design choice is that susceptibility is not
treated as a fixed trait known before simulation. Instead, it is estimated
post hoc conditional on the exact target set that was modeled:

    target set = {(attack_leaf, opinion_leaf)}

Primary effectivity metric (run_7+): adversarial_effectivity
    = signed opinion delta × adversarially assigned direction per leaf.
    Positive = attack moved opinion in the adversary's intended direction.
    This replaces abs_delta_score as primary because direction matters:
    a profile that shifted but in the wrong direction for the adversary is
    less exploitable than one that shifted in the right direction.

For each attack-opinion task the module fits a regularized profile-only ridge
model on observed adversarial effectivity. Each task-specific model produces a
fitted mapping:

    profile features -> predicted adversarial effectivity for that task

Those task-level predictions are aggregated back to the profile level using
reliability weights derived from sample size and cross-validated error.

Hierarchical decomposition (run_7+):
    Profile features are organized into an ontology-aligned hierarchy:
    - Level 1: Demographics (age, sex) vs. Personality (all Big Five features)
    - Level 2: Within Personality — five Big Five trait groups
    - Level 3: Within each trait — trait mean vs. individual facets
    For each hierarchy level and group, marginal cross-validated R² is computed
    (full model R² minus ablated model R² with that group removed) to quantify
    unique hierarchical contribution to susceptibility variation.

Key public entry points:
- fit_conditional_susceptibility_index(...)
- score_profiles_with_conditional_artifact(...)
- build_conditional_weight_table(...)
- compute_hierarchical_decomposition(...)
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

# Kept for import compatibility only — not used internally. Use
# _build_feature_hierarchy(), which auto-discovers groups from column names.
BIG_FIVE_TRAITS = [
    "neuroticism",
    "openness_to_experience",
    "conscientiousness",
    "extraversion",
    "agreeableness",
]

# Known demographic singleton tokens (maps to the "demographics" ablation group
# alongside all profile_cat__* one-hot columns).
_DEMO_SINGLETONS: frozenset = frozenset({
    "age", "income", "education", "bmi", "weight", "height",
})

# Suffixes stripped when parsing tokens from column names
_COL_STRIP_SUFFIXES = ("_pct", "_years", "_score", "_proxy", "_index", "_z", "_norm")


@dataclass
class HierarchicalDecomposition:
    """Per-task and aggregated hierarchical R² decomposition."""
    # Aggregated across tasks (weighted by reliability_weight)
    group_marginal_r2: Dict[str, float] = field(default_factory=dict)
    group_relative_importance_pct: Dict[str, float] = field(default_factory=dict)
    full_model_cv_r2: float = 0.0
    # Per-task breakdown (task_key -> {group -> marginal_r2})
    task_group_r2: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ConditionalSusceptibilityFitResult:
    artifact: ConditionalSusceptibilityArtifact
    task_coefficients: pd.DataFrame
    task_summary: pd.DataFrame
    profile_scores: pd.DataFrame
    contribution_breakdown: pd.DataFrame
    hierarchical_decomposition: Optional[HierarchicalDecomposition] = None


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


def _cv_r2(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    folds: List[np.ndarray],
) -> float:
    """Cross-validated R² using a fixed alpha."""
    all_preds: List[float] = []
    all_actual: List[float] = []
    for fold in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[fold] = False
        if mask.sum() < 2:
            continue
        beta = _ridge_fit_matrix(x[mask], y[mask], alpha)
        all_preds.extend((x[fold] @ beta).tolist())
        all_actual.extend(y[fold].tolist())
    if not all_preds:
        return 0.0
    preds_arr = np.array(all_preds)
    actual_arr = np.array(all_actual)
    ss_res = float(np.sum((actual_arr - preds_arr) ** 2))
    ss_tot = float(np.sum((actual_arr - actual_arr.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-10)


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


def _col_to_tokens(col: str) -> Tuple[str, ...]:
    """Strip profile prefix and trailing suffix; return token tuple."""
    inner = col
    for pfx in ("profile_cont_", "profile_cat__"):
        if inner.startswith(pfx):
            inner = inner[len(pfx):]
            break
    for suf in _COL_STRIP_SUFFIXES:
        if inner.endswith(suf):
            inner = inner[: -len(suf)]
            break
    # Also strip a trailing _mean token (it belongs to the dimension, not a sub-level)
    if inner.endswith("_mean"):
        inner = inner[: -len("_mean")]
    return tuple(inner.split("_"))


def _build_feature_hierarchy(feature_columns: List[str]) -> Dict[str, List[str]]:
    """
    Auto-discover ontology-aligned feature groups from column names alone.

    No inventory names (Big Five, HEXACO, Dark Triad, …) or field names (age,
    sex) are hardcoded.  The algorithm:

    1.  Categorical columns (profile_cat__*) → "demographics" group.
    2.  Continuous columns (profile_cont_*) whose token-set overlaps
        _DEMO_SINGLETONS → "demographics" group.
    3.  Remaining continuous columns → prefix-trie to find:
          • inventory-level groups  (shallowest prefix with ≥2 cols + ≥2 child tokens)
          • dimension-level groups  (one level below each inventory)
        These produce groups named by their inferred key, e.g.:
          "big_five"              → all Big Five columns
          "big_five_neuroticism"  → all Neuroticism columns

    Groups with fewer than 2 columns are omitted (not meaningful for ablation).
    """
    from collections import defaultdict as _dd

    cat_cols = [c for c in feature_columns if c.startswith("profile_cat__")]
    cont_cols = [c for c in feature_columns if c.startswith("profile_cont_")]

    col_toks: Dict[str, Tuple[str, ...]] = {c: _col_to_tokens(c) for c in cont_cols}

    # ── Demographics: categorical + age/income/... singletons ────────────────
    demo_cont = [
        c for c, toks in col_toks.items()
        if any(t in _DEMO_SINGLETONS for t in toks)
    ]
    demo = sorted(set(cat_cols + demo_cont))

    # ── Prefix trie over non-demographic continuous columns ──────────────────
    non_demo_cont = [c for c in cont_cols if c not in demo_cont]
    nd_toks: Dict[str, Tuple[str, ...]] = {c: col_toks[c] for c in non_demo_cont}

    # trie[depth][prefix_tuple] → [col, …]
    trie: Dict[int, Dict[Tuple, List[str]]] = _dd(lambda: _dd(list))
    for col, toks in nd_toks.items():
        for d in range(1, len(toks)):
            trie[d][toks[:d]].append(col)

    inventory_prefixes: set = set()
    inventory_groups: Dict[str, List[str]] = {}

    for d in sorted(trie):
        for prefix, cols in trie[d].items():
            # Skip if a shallower parent is already claimed as an inventory
            if any(prefix[:i] in inventory_prefixes for i in range(1, d)):
                continue
            if len(cols) < 2:
                continue
            # Need ≥2 distinct child tokens at depth d (confirms a branching hierarchy)
            child_tokens = {nd_toks[c][d] for c in cols if len(nd_toks[c]) > d}
            if len(child_tokens) < 2:
                continue
            inv_key = "_".join(prefix)
            inventory_groups[inv_key] = cols
            inventory_prefixes.add(prefix)

    # Dimension groups: one level below each inventory
    dimension_groups: Dict[str, List[str]] = {}
    for inv_prefix in inventory_prefixes:
        inv_d = len(inv_prefix)
        dim_d = inv_d + 1
        for prefix, cols in trie.get(dim_d, {}).items():
            if prefix[:inv_d] == inv_prefix and len(cols) >= 2:
                dimension_groups["_".join(prefix)] = cols

    # ── Assemble final groups ────────────────────────────────────────────────
    groups: Dict[str, List[str]] = {}
    if demo:
        groups["demographics"] = demo
    groups.update(inventory_groups)
    groups.update(dimension_groups)

    return {k: sorted(set(v)) for k, v in groups.items() if len(v) >= 2}


def compute_hierarchical_decomposition(
    x_full: np.ndarray,
    y: np.ndarray,
    feature_columns: List[str],
    best_alpha: float,
    seed: int,
) -> Dict[str, float]:
    """Compute leave-one-group-out marginal CV-R² for each feature hierarchy group.

    Returns a dict mapping group_name -> marginal_cv_r2 (contribution of that group
    to the full model's explanatory power). Also includes 'full_model' -> full CV-R².
    """
    folds = _kfold_indices(len(y), seed=seed, n_splits=5)
    r2_full = _cv_r2(x_full, y, best_alpha, folds)

    hierarchy = _build_feature_hierarchy(feature_columns)
    # col_idx_map: feature_columns[i] is at column index i+1 in x_full (index 0 is intercept)
    col_idx_map = {name: idx + 1 for idx, name in enumerate(feature_columns)}

    result: Dict[str, float] = {"full_model": r2_full}

    for group_name, group_cols in hierarchy.items():
        group_indices = [col_idx_map[c] for c in group_cols if c in col_idx_map]
        if not group_indices:
            continue
        keep = [i for i in range(x_full.shape[1]) if i not in group_indices]
        if len(keep) < 2:
            result[f"marginal_{group_name}"] = r2_full
            continue
        x_ablated = x_full[:, keep]
        r2_ablated = _cv_r2(x_ablated, y, best_alpha, folds)
        result[f"marginal_{group_name}"] = r2_full - r2_ablated

    return result


def fit_conditional_susceptibility_index(
    long_df: pd.DataFrame,
    *,
    outcome_metric: str = "adversarial_effectivity",
    feature_columns: Sequence[str] | None = None,
    excluded_feature_columns: Sequence[str] | None = None,
    seed: int = 42,
    alpha_grid: Sequence[float] | None = None,
    min_rows_per_task: int = 8,
    compute_hierarchy: bool = True,
) -> ConditionalSusceptibilityFitResult:
    # Fall back gracefully if adversarial_effectivity is not yet in the data
    if outcome_metric not in long_df.columns:
        fallback = "abs_delta_score"
        if fallback in long_df.columns:
            import warnings
            warnings.warn(
                f"outcome_metric='{outcome_metric}' not found in long_df; falling back to '{fallback}'. "
                "Run Stage 05 with --ontology-root to generate adversarial_effectivity.",
                stacklevel=2,
            )
            outcome_metric = fallback
        else:
            raise ValueError(
                f"Neither '{outcome_metric}' nor 'abs_delta_score' found in long_df columns."
            )

    required_columns = {"profile_id", "attack_leaf", "opinion_leaf", outcome_metric}
    missing = sorted(required_columns - set(long_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for conditional susceptibility fit: {missing}")

    attacked_df = long_df.copy()
    attacked_df = attacked_df.loc[attacked_df[outcome_metric].notna()].copy()
    if attacked_df.empty:
        raise ValueError("No rows with non-null outcome available for conditional susceptibility fitting.")

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
    transformed_profiles = transformed_profiles.copy()
    transformed_profiles.insert(0, "profile_id", unique_profiles_df["profile_id"])

    profile_lookup = transformed_profiles.set_index("profile_id")
    task_rows: List[Dict[str, object]] = []
    coeff_rows: List[pd.DataFrame] = []
    task_models: List[ConditionalSusceptibilityTaskModel] = []
    task_hierarchy_r2: Dict[str, Dict[str, float]] = {}

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

        if compute_hierarchy:
            try:
                hier_r2 = compute_hierarchical_decomposition(
                    x_full=x,
                    y=y,
                    feature_columns=list(feature_columns),
                    best_alpha=alpha,
                    seed=seed + offset,
                )
                task_hierarchy_r2[key] = hier_r2
            except Exception:
                task_hierarchy_r2[key] = {}

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

    artifact = ConditionalSusceptibilityArtifact(
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
            f"Primary effectivity metric: {outcome_metric}. Positive = opinion moved in adversary's intended direction.",
            "The conditional susceptibility index is computed from fitted task-specific ridge models and is model-based rather than directly observed.",
        ],
    )

    score_df, breakdown_df = score_profiles_with_conditional_artifact(unique_profiles_df, artifact)

    # Aggregate hierarchical decomposition across tasks weighted by reliability
    hier_decomp: Optional[HierarchicalDecomposition] = None
    if compute_hierarchy and task_hierarchy_r2:
        agg_group_r2: Dict[str, float] = {}
        total_weight = sum(normalized_weights.values())
        full_r2_weighted = 0.0
        for model in task_models:
            w = float(model.reliability_weight)
            task_r2 = task_hierarchy_r2.get(model.task_key, {})
            full_r2_weighted += w * float(task_r2.get("full_model", 0.0))
            for k, v in task_r2.items():
                if k == "full_model":
                    continue
                agg_group_r2[k] = agg_group_r2.get(k, 0.0) + w * float(v)
        # Normalize weighted marginals by total weight
        if total_weight > 0:
            agg_group_r2 = {k: v / total_weight for k, v in agg_group_r2.items()}
            full_r2_weighted /= total_weight

        # Relative importance: marginal R² / sum(|marginal R²|)
        total_abs = sum(abs(v) for v in agg_group_r2.values()) or 1.0
        rel_importance = {k: abs(v) / total_abs * 100.0 for k, v in agg_group_r2.items()}

        hier_decomp = HierarchicalDecomposition(
            group_marginal_r2=agg_group_r2,
            group_relative_importance_pct=rel_importance,
            full_model_cv_r2=full_r2_weighted,
            task_group_r2=task_hierarchy_r2,
        )

    return ConditionalSusceptibilityFitResult(
        artifact=artifact,
        task_coefficients=coeff_df,
        task_summary=task_summary,
        profile_scores=score_df,
        contribution_breakdown=breakdown_df,
        hierarchical_decomposition=hier_decomp,
    )


def score_profiles_with_conditional_artifact(
    profile_df: pd.DataFrame,
    artifact: ConditionalSusceptibilityArtifact,
    target_attacks: Optional[List[str]] = None,
    target_opinions: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score profiles using a fitted artifact, optionally filtering to a subset of tasks.

    Args:
        profile_df: DataFrame with profile_id and feature columns. Missing feature
            columns are imputed with the artifact's stored training-set means.
        artifact: Fitted ConditionalSusceptibilityArtifact.
        target_attacks: If given, restrict to tasks whose attack_leaf is in this list.
        target_opinions: If given, restrict to tasks whose opinion_leaf is in this list.

    Returns:
        (profile_scores_df, breakdown_df) — same structure as fit outputs.
    """
    if "profile_id" not in profile_df.columns:
        raise ValueError("profile_df must contain profile_id for conditional susceptibility scoring.")

    # Select tasks matching filter (or all tasks)
    active_tasks = [
        t for t in artifact.task_models
        if (target_attacks is None or t.attack_leaf in target_attacks)
        and (target_opinions is None or t.opinion_leaf in target_opinions)
    ]
    if not active_tasks:
        raise ValueError(
            "No tasks matched the specified target_attacks/target_opinions filter. "
            f"Available attacks: {artifact.attack_leaves}. "
            f"Available opinions: {artifact.opinion_leaves}."
        )

    # Re-normalize weights for selected tasks
    weight_sum = sum(t.reliability_weight for t in active_tasks)
    weight_sum = max(weight_sum, 1e-10)

    # Build per-profile feature matrix — impute missing columns with training means
    avail_features = [c for c in artifact.feature_columns if c in profile_df.columns]
    missing_features = [c for c in artifact.feature_columns if c not in profile_df.columns]

    unique_profile_df = profile_df[["profile_id", *avail_features]].drop_duplicates(subset=["profile_id"]).reset_index(drop=True)

    # Fill missing feature columns with training-set means (imputation)
    for col in missing_features:
        imputed = artifact.feature_means.get(col, 0.0)
        unique_profile_df[col] = imputed

    transformed = _transform_features(
        df=unique_profile_df,
        feature_columns=artifact.feature_columns,
        feature_means=artifact.feature_means,
        feature_stds=artifact.feature_stds,
        continuous_columns=artifact.continuous_feature_columns,
    )
    transformed = transformed.copy()
    transformed.insert(0, "profile_id", unique_profile_df["profile_id"])

    score_df = unique_profile_df[["profile_id"]].copy()
    score_df["conditional_target_attack_count"] = len({t.attack_leaf for t in active_tasks})
    score_df["conditional_target_opinion_count"] = len({t.opinion_leaf for t in active_tasks})
    score_df["conditional_target_task_count"] = len(active_tasks)
    score_df["imputed_feature_count"] = len(missing_features)
    score_df["imputed_features"] = ", ".join(missing_features) if missing_features else ""

    raw_score = np.zeros(len(score_df), dtype=float)
    breakdown_rows: List[Dict[str, object]] = []

    for task_model in active_tasks:
        normalized_w = float(task_model.reliability_weight) / weight_sum
        task_contribution = np.full(len(score_df), float(task_model.intercept), dtype=float)
        for column in artifact.feature_columns:
            beta = float(task_model.coefficients.get(column, 0.0))
            task_contribution = task_contribution + transformed[column].astype(float).to_numpy() * beta
        weighted_contribution = task_contribution * normalized_w
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
    for task_model in active_tasks:
        normalized_w = float(task_model.reliability_weight) / weight_sum
        for column, beta in task_model.coefficients.items():
            signed_weight_lookup[column] = signed_weight_lookup.get(column, 0.0) + float(beta) * normalized_w

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
    """
    Aggregate per-task ridge coefficients into a reliability-weighted feature
    importance table.

    Columns returned:
      term                     – raw feature column name
      moderator_label          – human-readable label (from SemanticScaleRegistry)
      ontology_group           – auto-discovered hierarchy group (inventory or dim)
      weighted_mean_estimate   – signed task-reliability-weighted mean coefficient
      weighted_mean_abs_estimate – unsigned magnitude
      normalized_weight_pct    – |effect| as % of total |effect|
      direction                – "higher susceptibility" | "lower susceptibility" | "neutral"
      n_tasks                  – how many tasks contributed a non-zero coefficient
    """
    # Build ontology groups from column names (no hardcoding)
    hierarchy = _build_feature_hierarchy(artifact.feature_columns)
    # Invert: column → group name (use most specific group that contains it)
    col_to_group: Dict[str, str] = {}
    # Sort by group key length descending so more specific groups win
    for grp, cols in sorted(hierarchy.items(), key=lambda kv: len(kv[0]), reverse=True):
        for c in cols:
            if c not in col_to_group:
                col_to_group[c] = grp

    # Import semantic scale registry lazily to avoid circular deps
    try:
        from src.backend.utils.semantic_scale import get_default_registry as _gdr
        scale_reg = _gdr()
    except Exception:
        scale_reg = None

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

        # Human-readable label
        if scale_reg is not None:
            sc = scale_reg.get_scale(column)
            if sc is not None:
                label = sc.dimension_label
            elif column.startswith("profile_cat__"):
                # Categorical one-hot: derive "Sex: Female" style label
                inner = column.removeprefix("profile_cat__")
                # Strip any leading "profile_cat_" fragment kept by double-prefix encoding
                inner = inner.removeprefix("profile_cat_")
                parts = inner.split("_")
                # Last token = level, everything before = group key
                level = parts[-1]
                group = " ".join(parts[:-1]).title() if len(parts) > 1 else inner.title()
                label = f"{group}: {level}"
            else:
                inner = column.removeprefix("profile_cont_")
                # Strip known suffixes
                for suf in ("_pct", "_years", "_score"):
                    if inner.endswith(suf):
                        inner = inner[: -len(suf)]
                        break
                label = inner.replace("_", " ").strip().title()
        else:
            label = column

        group = col_to_group.get(column, "other")
        direction = (
            "higher susceptibility" if signed_effect > 0.01
            else "lower susceptibility" if signed_effect < -0.01
            else "neutral"
        )

        rows.append(
            {
                "term": column,
                "moderator_label": label,
                "ontology_group": group,
                "weighted_mean_estimate": signed_effect,
                "weighted_mean_abs_estimate": abs_effect,
                "direction": direction,
                "n_tasks": n_tasks,
            }
        )

    weight_df = pd.DataFrame(rows)
    if weight_df.empty:
        return weight_df
    denom = float(weight_df["weighted_mean_abs_estimate"].sum()) or 1.0
    weight_df["normalized_weight_pct"] = (
        weight_df["weighted_mean_abs_estimate"] / denom
    ) * 100.0
    return weight_df.sort_values(
        ["normalized_weight_pct", "term"], ascending=[False, True]
    ).reset_index(drop=True)
