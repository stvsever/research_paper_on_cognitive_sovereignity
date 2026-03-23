from __future__ import annotations

"""
Technical overview
------------------
Stage 05 is the bridge between raw scenario-level simulation outputs and the
 statistical datasets used downstream for moderation analysis.

It performs three jobs:
1. construct attacked effectivity outcomes for each scenario row
2. encode profile variables and opinion fixed effects into analysis-ready form
3. roll the long attacked table up into profile-level repeated-outcome tables

The stage keeps both signed and absolute opinion movement:

    delta_score     = post_score - baseline_score
    abs_delta_score = |post_score - baseline_score|

The absolute shift is important because one fixed attack can move different
opinion leaves in different signed directions. If only signed deltas were kept,
cross-leaf movement could cancel out.

This stage also creates the profile-level wide panel used by Stage 06. In that
wide table, each profile receives separate attacked outcome indicators for each
opinion leaf, which enables repeated-outcome SEM/path modeling rather than a
premature collapse to a single summary score.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.data_utils import choose_primary_moderator_column, one_hot_profile_categoricals, zscore_series
from src.backend.utils.io import (
    abs_path,
    ensure_dir,
    read_jsonl,
    stage_manifest_path,
    write_json,
    write_jsonl,
)
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.scenario_realism import extract_leaf_label, extract_opinion_domain
from src.backend.utils.schemas import (
    AttackExposure,
    DeltaRecord,
    OpinionAssessment,
    ScenarioRecord,
    SemRow,
    StageArtifactManifest,
    StageConfig,
)


LOGGER = logging.getLogger(__name__)


class Stage05Config(StageConfig):
    primary_moderator: str = "profile_cont_age_years"


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_").replace(">", "_")


def _add_fixed_effects(
    df: pd.DataFrame,
    source_column: str,
    prefix: str,
) -> tuple[pd.DataFrame, str | None]:
    unique_values = sorted(df[source_column].dropna().unique().tolist())
    if len(unique_values) <= 1:
        return df, unique_values[0] if unique_values else None

    reference_value = unique_values[0]
    for value in unique_values[1:]:
        column_name = f"{prefix}_{_slugify(value)}"
        df[column_name] = (df[source_column] == value).astype(float)
    return df, reference_value


def _profile_level_rollup(df_encoded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_columns = [
        column
        for column in df_encoded.columns
        if column.startswith("profile_cont_") or column.startswith("profile_cat__")
    ]

    aggregate_spec = {
        "baseline_score": "mean",
        "post_score": "mean",
        "delta_score": "mean",
        "abs_delta_score": "mean",
        "baseline_abs_score": "mean",
        "exposure_quality_score": "mean",
        "attack_realism_score": "mean",
        "attack_coherence_score": "mean",
        "post_plausibility_score": "mean",
        "post_consistency_score": "mean",
        "scenario_id": "count",
    }
    available_aggregates = {
        key: value for key, value in aggregate_spec.items() if key in df_encoded.columns
    }

    grouped = df_encoded.groupby("profile_id", as_index=False).agg(available_aggregates)
    grouped = grouped.rename(
        columns={
            "baseline_score": "mean_baseline_score",
            "post_score": "mean_post_score",
            "delta_score": "mean_signed_delta_score",
            "abs_delta_score": "mean_abs_delta_score",
            "baseline_abs_score": "mean_baseline_abs_score",
            "exposure_quality_score": "mean_exposure_quality_score",
            "attack_realism_score": "mean_attack_realism_score",
            "attack_coherence_score": "mean_attack_coherence_score",
            "post_plausibility_score": "mean_post_plausibility_score",
            "post_consistency_score": "mean_post_consistency_score",
            "scenario_id": "n_attacked_opinion_leaves",
        }
    )

    if profile_columns:
        profile_values = df_encoded.groupby("profile_id", as_index=False)[profile_columns].first()
        grouped = grouped.merge(profile_values, on="profile_id", how="left")

    leaf_key_col = "opinion_leaf_label"
    abs_pivot = df_encoded.pivot_table(
        index="profile_id",
        columns=leaf_key_col,
        values="abs_delta_score",
        aggfunc="mean",
    )
    signed_pivot = df_encoded.pivot_table(
        index="profile_id",
        columns=leaf_key_col,
        values="delta_score",
        aggfunc="mean",
    )

    abs_pivot = abs_pivot.rename(columns=lambda value: f"abs_delta_indicator__{_slugify(str(value))}")
    signed_pivot = signed_pivot.rename(columns=lambda value: f"signed_delta_indicator__{_slugify(str(value))}")

    wide = grouped.merge(abs_pivot.reset_index(), on="profile_id", how="left")
    wide = wide.merge(signed_pivot.reset_index(), on="profile_id", how="left")

    if "mean_baseline_abs_score" in wide.columns:
        wide["mean_baseline_abs_score_z"] = zscore_series(wide["mean_baseline_abs_score"].astype(float))
    if "mean_exposure_quality_score" in wide.columns:
        wide["mean_exposure_quality_score_z"] = zscore_series(wide["mean_exposure_quality_score"].astype(float))
    if "mean_abs_delta_score" in wide.columns:
        wide["mean_abs_delta_score_z"] = zscore_series(wide["mean_abs_delta_score"].astype(float))
    if "mean_signed_delta_score" in wide.columns:
        wide["mean_signed_delta_score_z"] = zscore_series(wide["mean_signed_delta_score"].astype(float))

    indicator_columns = [column for column in wide.columns if column.startswith("abs_delta_indicator__")]
    for column in indicator_columns:
        wide[f"{column}_z"] = zscore_series(wide[column].astype(float))

    return grouped, wide


def run_stage(input_path: str, output_dir: str, config: Stage05Config) -> StageArtifactManifest:
    ensure_dir(output_dir)
    rows = read_jsonl(input_path)

    deltas: List[DeltaRecord] = []
    sem_rows: List[SemRow] = []
    flat_rows: List[Dict[str, object]] = []

    for row in rows:
        scenario = ScenarioRecord.model_validate(
            {
                k: v
                for k, v in row.items()
                if k
                not in {
                    "baseline_assessment",
                    "attack_exposure",
                    "post_attack_assessment",
                }
            }
        )
        baseline = OpinionAssessment.model_validate(row["baseline_assessment"])
        post = OpinionAssessment.model_validate(row["post_attack_assessment"])
        exposure = AttackExposure.model_validate(row["attack_exposure"])
        review = row.get("attack_realism_review", {}) if isinstance(row, dict) else {}
        heuristics = row.get("attack_heuristic_checks", {}) if isinstance(row, dict) else {}
        baseline_review = row.get("baseline_coherence_review", {}) if isinstance(row, dict) else {}
        baseline_heuristics = row.get("baseline_heuristic_checks", {}) if isinstance(row, dict) else {}
        post_review = row.get("post_coherence_review", {}) if isinstance(row, dict) else {}
        post_heuristics = row.get("post_heuristic_checks", {}) if isinstance(row, dict) else {}

        signed_delta = int(post.score - baseline.score)
        abs_delta = int(abs(signed_delta))

        delta_record = DeltaRecord(
            scenario_id=scenario.scenario_id,
            opinion_leaf=scenario.opinion_leaf,
            baseline_score=baseline.score,
            post_score=post.score,
            delta_score=signed_delta,
            abs_delta_score=abs_delta,
            attack_present=scenario.attack_present,
            attack_leaf=scenario.attack_leaf,
            profile_id=scenario.profile.profile_id,
            profile_categorical=scenario.profile.categorical_attributes,
            profile_continuous=scenario.profile.continuous_attributes,
        )
        deltas.append(delta_record)

        features = {
            **{f"profile_cont_{k}": float(v) for k, v in scenario.profile.continuous_attributes.items()},
            **{f"profile_cat_{k}": v for k, v in scenario.profile.categorical_attributes.items()},
        }

        flat_row: Dict[str, object] = {
            "scenario_id": scenario.scenario_id,
            "opinion_leaf": scenario.opinion_leaf,
            "opinion_domain": extract_opinion_domain(scenario.opinion_leaf),
            "opinion_leaf_label": extract_leaf_label(scenario.opinion_leaf),
            "attack_present": int(scenario.attack_present),
            "attack_leaf": scenario.attack_leaf or "CONTROL_NONE",
            "baseline_score": float(baseline.score),
            "post_score": float(post.score),
            "delta_score": float(signed_delta),
            "abs_delta_score": float(abs_delta),
            "profile_id": scenario.profile.profile_id,
            "exposure_intensity_hint": float(exposure.intensity_hint),
            "attack_realism_score": review.get("realism_score"),
            "attack_coherence_score": review.get("coherence_score"),
            "attack_rewrite_required": review.get("rewrite_required"),
            "attack_heuristic_pass": (
                heuristics.get("checks", {}).get("overall_pass")
                if isinstance(heuristics, dict)
                else None
            ),
            "baseline_plausibility_score": baseline_review.get("plausibility_score"),
            "baseline_consistency_score": baseline_review.get("consistency_score"),
            "baseline_rewrite_required": baseline_review.get("rewrite_required"),
            "baseline_heuristic_pass": (
                baseline_heuristics.get("checks", {}).get("overall_pass")
                if isinstance(baseline_heuristics, dict)
                else None
            ),
            "post_plausibility_score": post_review.get("plausibility_score"),
            "post_consistency_score": post_review.get("consistency_score"),
            "post_rewrite_required": post_review.get("rewrite_required"),
            "post_heuristic_pass": (
                post_heuristics.get("checks", {}).get("overall_pass")
                if isinstance(post_heuristics, dict)
                else None
            ),
            "baseline_fallback_used": baseline.model_name == "fallback_deterministic",
            "post_fallback_used": post.model_name == "fallback_deterministic",
            "scenario_design": scenario.metadata.get("scenario_design"),
            "profile_panel_index": scenario.metadata.get("profile_panel_index"),
            "leaf_repeat_index_within_profile": scenario.metadata.get("leaf_repeat_index_within_profile"),
        }
        flat_row.update(features)
        flat_rows.append(flat_row)

        numeric_profile = {
            k: float(v)
            for k, v in features.items()
            if k.startswith("profile_cont_")
        }
        sem_rows.append(
            SemRow(
                scenario_id=scenario.scenario_id,
                opinion_leaf=scenario.opinion_leaf,
                baseline_score=float(baseline.score),
                post_score=float(post.score),
                delta_score=float(signed_delta),
                abs_delta_score=float(abs_delta),
                attack_present=int(scenario.attack_present),
                attack_leaf=scenario.attack_leaf or "CONTROL_NONE",
                profile_id=scenario.profile.profile_id,
                profile_features=numeric_profile,
            )
        )

    df_raw = pd.DataFrame(flat_rows)
    df_encoded = one_hot_profile_categoricals(df_raw.copy())
    df_encoded["baseline_abs_score"] = df_encoded["baseline_score"].abs().astype(float)
    df_encoded["baseline_extremity_norm"] = df_encoded["baseline_abs_score"] / 1000.0
    quality_columns = [
        column
        for column in ["exposure_intensity_hint", "attack_realism_score", "attack_coherence_score"]
        if column in df_encoded.columns
    ]
    if quality_columns:
        df_encoded["exposure_quality_score"] = (
            df_encoded[quality_columns]
            .astype(float)
            .mean(axis=1, skipna=True)
            .fillna(df_encoded["exposure_intensity_hint"].astype(float) if "exposure_intensity_hint" in df_encoded.columns else 0.5)
        )
    else:
        df_encoded["exposure_quality_score"] = 0.5
    df_encoded["exposure_quality_z"] = zscore_series(df_encoded["exposure_quality_score"].astype(float))
    df_encoded, reference_leaf = _add_fixed_effects(df_encoded, "opinion_leaf", "opinion_leaf_fe")
    df_encoded, reference_domain = _add_fixed_effects(df_encoded, "opinion_domain", "opinion_domain_fe")

    moderator_col = choose_primary_moderator_column(df_encoded, preferred=config.primary_moderator)
    df_encoded["primary_moderator_value"] = df_encoded[moderator_col].astype(float)
    df_encoded["primary_moderator_z"] = zscore_series(df_encoded[moderator_col].astype(float))

    profile_summary_df, profile_wide_df = _profile_level_rollup(df_encoded)

    delta_jsonl = Path(output_dir) / "effectivity_deltas.jsonl"
    sem_rows_jsonl = Path(output_dir) / "sem_long_rows.jsonl"
    sem_raw_csv = Path(output_dir) / "sem_long_raw.csv"
    sem_encoded_csv = Path(output_dir) / "sem_long_encoded.csv"
    sem_encoded_jsonl = Path(output_dir) / "sem_long_encoded.jsonl"
    profile_summary_csv = Path(output_dir) / "profile_level_effectivity.csv"
    profile_wide_csv = Path(output_dir) / "profile_sem_wide.csv"
    summary_json = Path(output_dir) / "delta_summary.json"

    write_jsonl(delta_jsonl, (x.model_dump() for x in deltas))
    write_jsonl(sem_rows_jsonl, (x.model_dump() for x in sem_rows))
    df_raw.to_csv(sem_raw_csv, index=False)
    df_encoded.to_csv(sem_encoded_csv, index=False)
    write_jsonl(sem_encoded_jsonl, df_encoded.to_dict(orient="records"))
    profile_summary_df.to_csv(profile_summary_csv, index=False)
    profile_wide_df.to_csv(profile_wide_csv, index=False)

    write_json(
        summary_json,
        {
            "n_records": len(deltas),
            "n_profiles": int(df_encoded["profile_id"].nunique()),
            "analysis_mode": (
                "treated_only"
                if len(df_encoded) and int(df_encoded["attack_present"].min()) == 1 and int(df_encoded["attack_present"].max()) == 1
                else "mixed_condition"
            ),
            "mean_signed_delta": float(df_encoded["delta_score"].mean()),
            "std_signed_delta": float(df_encoded["delta_score"].std(ddof=0)),
            "mean_abs_delta": float(df_encoded["abs_delta_score"].mean()),
            "std_abs_delta": float(df_encoded["abs_delta_score"].std(ddof=0)),
            "primary_moderator_column": moderator_col,
            "reference_opinion_leaf": reference_leaf,
            "reference_opinion_domain": reference_domain,
            "n_unique_opinion_leaves": int(df_encoded["opinion_leaf"].nunique()),
            "scenarios_per_profile": float(df_encoded.groupby("profile_id")["scenario_id"].count().mean()),
            "attack_present_count": int(df_encoded["attack_present"].sum()),
            "control_count": int((1 - df_encoded["attack_present"]).sum()),
            "exposure_quality_mean": float(df_encoded["exposure_quality_score"].mean()),
        },
    )

    manifest = StageArtifactManifest(
        stage_id="05",
        stage_name="compute_effectivity_deltas",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(sem_encoded_csv),
        output_files=[
            abs_path(delta_jsonl),
            abs_path(sem_rows_jsonl),
            abs_path(sem_raw_csv),
            abs_path(sem_encoded_csv),
            abs_path(sem_encoded_jsonl),
            abs_path(profile_summary_csv),
            abs_path(profile_wide_csv),
            abs_path(summary_json),
        ],
        record_count=len(deltas),
        metadata={
            "primary_moderator_column": moderator_col,
            "reference_opinion_leaf": reference_leaf,
            "reference_opinion_domain": reference_domain,
            "n_unique_opinion_leaves": int(df_encoded["opinion_leaf"].nunique()),
            "n_profiles": int(df_encoded["profile_id"].nunique()),
            "analysis_mode": (
                "treated_only"
                if len(df_encoded) and int(df_encoded["attack_present"].min()) == 1 and int(df_encoded["attack_present"].max()) == 1
                else "mixed_condition"
            ),
            "effectivity_outcome": "absolute_shift_primary_signed_shift_secondary",
        },
    )

    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 05 - Compute effectivity deltas")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-moderator", default="profile_cont_age_years")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)

    config = Stage05Config(
        stage_name="compute_effectivity_deltas",
        run_id=args.run_id,
        seed=args.seed,
        primary_moderator=args.primary_moderator,
    )
    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 05 completed: %s records", manifest.record_count)


if __name__ == "__main__":
    main()
