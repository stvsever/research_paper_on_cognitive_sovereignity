from __future__ import annotations

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
    primary_moderator: str = "profile_cont_susceptibility_index"


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

        delta = post.score - baseline.score

        delta_record = DeltaRecord(
            scenario_id=scenario.scenario_id,
            opinion_leaf=scenario.opinion_leaf,
            baseline_score=baseline.score,
            post_score=post.score,
            delta_score=delta,
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
            "delta_score": float(delta),
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
                delta_score=float(delta),
                attack_present=int(scenario.attack_present),
                attack_leaf=scenario.attack_leaf or "CONTROL_NONE",
                profile_id=scenario.profile.profile_id,
                profile_features=numeric_profile,
            )
        )

    df_raw = pd.DataFrame(flat_rows)
    df_encoded = one_hot_profile_categoricals(df_raw.copy())

    moderator_col = choose_primary_moderator_column(df_encoded, preferred=config.primary_moderator)
    df_encoded["primary_moderator_value"] = df_encoded[moderator_col].astype(float)
    if moderator_col != "baseline_score":
        df_encoded["primary_moderator_z"] = zscore_series(df_encoded[moderator_col].astype(float))
    else:
        df_encoded["primary_moderator_z"] = zscore_series(df_encoded["baseline_score"].astype(float))

    df_encoded["attack_x_primary_moderator"] = (
        df_encoded["attack_present"].astype(float) * df_encoded["primary_moderator_z"].astype(float)
    )
    df_encoded["moderator_z"] = df_encoded["primary_moderator_z"]
    df_encoded["attack_x_moderator"] = df_encoded["attack_x_primary_moderator"]

    delta_jsonl = Path(output_dir) / "effectivity_deltas.jsonl"
    sem_rows_jsonl = Path(output_dir) / "sem_long_rows.jsonl"
    sem_raw_csv = Path(output_dir) / "sem_long_raw.csv"
    sem_encoded_csv = Path(output_dir) / "sem_long_encoded.csv"
    sem_encoded_jsonl = Path(output_dir) / "sem_long_encoded.jsonl"
    summary_json = Path(output_dir) / "delta_summary.json"

    write_jsonl(delta_jsonl, (x.model_dump() for x in deltas))
    write_jsonl(sem_rows_jsonl, (x.model_dump() for x in sem_rows))

    df_raw.to_csv(sem_raw_csv, index=False)
    df_encoded.to_csv(sem_encoded_csv, index=False)
    write_jsonl(sem_encoded_jsonl, df_encoded.to_dict(orient="records"))

    write_json(
        summary_json,
        {
            "n_records": len(deltas),
            "delta_mean": float(df_encoded["delta_score"].mean()),
            "delta_std": float(df_encoded["delta_score"].std(ddof=0)),
            "primary_moderator_column": moderator_col,
            "attack_present_count": int(df_encoded["attack_present"].sum()),
            "control_count": int((1 - df_encoded["attack_present"]).sum()),
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
            abs_path(summary_json),
        ],
        record_count=len(deltas),
        metadata={"primary_moderator_column": moderator_col},
    )

    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 05 - Compute effectivity deltas")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-moderator", default="profile_cont_susceptibility_index")
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
