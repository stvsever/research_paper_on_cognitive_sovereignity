from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.agentic_framework.agents import ExposureReviewResponse
from src.backend.agentic_framework.factory import AgentFactory
from src.backend.utils.io import (
    abs_path,
    ensure_dir,
    env_get_required,
    read_jsonl,
    stage_manifest_path,
    write_json,
    write_jsonl,
)
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.scenario_realism import (
    assess_attack_exposure_heuristics,
    build_attack_context,
    control_exposure_template,
    profile_context_snapshot,
)
from src.backend.utils.schemas import (
    AttackExposure,
    OpinionAssessment,
    ScenarioRecord,
    StageArtifactManifest,
    StageConfig,
)


LOGGER = logging.getLogger(__name__)


class Stage03Config(StageConfig):
    self_supervise_attack_realism: bool = True
    realism_threshold: float = 0.72


def _default_review() -> ExposureReviewResponse:
    return ExposureReviewResponse(
        realism_score=0.0,
        coherence_score=0.0,
        rewrite_required=False,
        rewrite_feedback="",
        notes="review_unavailable",
    )


def run_stage(input_path: str, output_dir: str, config: Stage03Config) -> StageArtifactManifest:
    if config.openrouter_model is None:
        raise RuntimeError("Stage 03 requires --openrouter-model")

    ensure_dir(output_dir)
    raw_dir = config.raw_llm_dir if config.save_raw_llm else None
    project_root = Path(__file__).resolve().parents[5]
    prompts_dir = project_root / "src" / "backend" / "agentic_framework" / "prompts"

    factory = AgentFactory(
        prompts_dir=prompts_dir,
        openrouter_api_key=env_get_required("OPENROUTER_API_KEY"),
        openrouter_model=config.openrouter_model,
        max_repair_iter=config.max_repair_iter,
        temperature=config.temperature,
        timeout_sec=config.timeout_sec,
        save_raw_dir=raw_dir,
    )
    attack_agent = factory.attack_exposure_agent()
    reviewer_agent = factory.attack_realism_reviewer_agent()

    rows = read_jsonl(input_path)

    exposures: List[AttackExposure] = []
    enriched_rows: List[Dict[str, object]] = []
    realism_scores: List[float] = []
    coherence_scores: List[float] = []
    review_rewrite_count = 0
    heuristic_fail_count = 0

    for row in rows:
        scenario = ScenarioRecord.model_validate({k: v for k, v in row.items() if k != "baseline_assessment"})
        baseline = OpinionAssessment.model_validate(row["baseline_assessment"])

        if not scenario.attack_present:
            exposure = AttackExposure(
                scenario_id=scenario.scenario_id,
                attack_present=False,
                attack_leaf=None,
                exposure_text=control_exposure_template(scenario.opinion_leaf),
                platform="news_digest_control",
                persuasion_strategy="neutral_topic_matched_control",
                intensity_hint=0.0,
                model_name="control_rule",
            )
            review = _default_review()
            heuristics = {
                "checks": {
                    "length_reasonable": True,
                    "contains_issue_anchor": True,
                    "contains_attack_theme": True,
                    "non_hacking_language": True,
                    "overall_pass": True,
                },
                "pass_count": 4,
            }
        else:
            attack_leaf = scenario.attack_leaf or "unknown_attack"
            attack_context = build_attack_context(
                opinion_leaf=scenario.opinion_leaf,
                attack_leaf=attack_leaf,
                profile=scenario.profile,
                baseline_score=baseline.score,
            )

            review = _default_review()
            try:
                exposure = attack_agent.generate(
                    run_id=config.run_id,
                    call_id=f"{scenario.scenario_id}_attack_primary",
                    scenario_id=scenario.scenario_id,
                    opinion_leaf=scenario.opinion_leaf,
                    attack_leaf=attack_leaf,
                    profile=scenario.profile,
                    baseline_score=baseline.score,
                    attack_present=True,
                    attack_context=attack_context,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Attack generation failed for %s, fallback template used: %s",
                    scenario.scenario_id,
                    exc,
                )
                exposure = AttackExposure(
                    scenario_id=scenario.scenario_id,
                    attack_present=True,
                    attack_leaf=attack_leaf,
                    exposure_text=(
                        "A targeted social post presents selective anecdotes and unverified claims "
                        "to distort perceived outcomes of the policy issue."
                    ),
                    platform="social_media_feed",
                    persuasion_strategy="misleading_framing_fallback",
                    intensity_hint=0.58,
                    model_name="fallback_template",
                )

            heuristics = assess_attack_exposure_heuristics(
                exposure_text=exposure.exposure_text,
                attack_leaf=attack_leaf,
                opinion_leaf=scenario.opinion_leaf,
            )

            if config.self_supervise_attack_realism:
                try:
                    review = reviewer_agent.review(
                        run_id=config.run_id,
                        call_id=f"{scenario.scenario_id}_attack_review_1",
                        scenario_id=scenario.scenario_id,
                        opinion_leaf=scenario.opinion_leaf,
                        attack_leaf=attack_leaf,
                        baseline_score=baseline.score,
                        profile_snapshot=profile_context_snapshot(scenario.profile),
                        generated_exposure=exposure,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Attack realism reviewer failed for %s: %s",
                        scenario.scenario_id,
                        exc,
                    )
                    review = _default_review()

                needs_rewrite = (
                    review.rewrite_required
                    or review.realism_score < config.realism_threshold
                    or review.coherence_score < config.realism_threshold
                    or not bool(heuristics["checks"].get("overall_pass", False))
                )

                if needs_rewrite:
                    review_rewrite_count += 1
                    feedback_parts = []
                    if review.rewrite_feedback:
                        feedback_parts.append(review.rewrite_feedback)
                    if not bool(heuristics["checks"].get("overall_pass", False)):
                        feedback_parts.append(
                            "Heuristic check failed; improve topic anchoring and attack-theme coherence."
                        )
                    rewrite_feedback = " ".join(feedback_parts).strip()

                    try:
                        exposure = attack_agent.generate(
                            run_id=config.run_id,
                            call_id=f"{scenario.scenario_id}_attack_rewrite",
                            scenario_id=scenario.scenario_id,
                            opinion_leaf=scenario.opinion_leaf,
                            attack_leaf=attack_leaf,
                            profile=scenario.profile,
                            baseline_score=baseline.score,
                            attack_present=True,
                            attack_context=attack_context,
                            review_feedback=rewrite_feedback,
                        )
                        heuristics = assess_attack_exposure_heuristics(
                            exposure_text=exposure.exposure_text,
                            attack_leaf=attack_leaf,
                            opinion_leaf=scenario.opinion_leaf,
                        )
                        try:
                            review = reviewer_agent.review(
                                run_id=config.run_id,
                                call_id=f"{scenario.scenario_id}_attack_review_2",
                                scenario_id=scenario.scenario_id,
                                opinion_leaf=scenario.opinion_leaf,
                                attack_leaf=attack_leaf,
                                baseline_score=baseline.score,
                                profile_snapshot=profile_context_snapshot(scenario.profile),
                                generated_exposure=exposure,
                            )
                        except Exception:
                            pass
                    except Exception as exc:
                        LOGGER.warning(
                            "Attack rewrite failed for %s: %s",
                            scenario.scenario_id,
                            exc,
                        )

            if not bool(heuristics["checks"].get("overall_pass", False)):
                heuristic_fail_count += 1

            realism_scores.append(float(review.realism_score))
            coherence_scores.append(float(review.coherence_score))

        exposures.append(exposure)
        enriched_row = dict(row)
        enriched_row["attack_exposure"] = exposure.model_dump()
        enriched_row["attack_realism_review"] = review.model_dump()
        enriched_row["attack_heuristic_checks"] = heuristics
        enriched_rows.append(enriched_row)

    exposure_jsonl = Path(output_dir) / "attack_exposures.jsonl"
    enriched_jsonl = Path(output_dir) / "scenarios_with_exposure.jsonl"
    summary_json = Path(output_dir) / "attack_summary.json"

    write_jsonl(exposure_jsonl, (x.model_dump() for x in exposures))
    write_jsonl(enriched_jsonl, enriched_rows)

    n_attack = sum(1 for x in exposures if x.attack_present)
    write_json(
        summary_json,
        {
            "n_records": len(exposures),
            "n_attack": n_attack,
            "n_control": len(exposures) - n_attack,
            "attack_leaf_counts": {
                leaf: sum(1 for x in exposures if x.attack_leaf == leaf)
                for leaf in sorted({x.attack_leaf for x in exposures if x.attack_leaf})
            },
            "self_supervise_attack_realism": config.self_supervise_attack_realism,
            "review_rewrite_count": review_rewrite_count,
            "heuristic_fail_count": heuristic_fail_count,
            "mean_realism_score": (sum(realism_scores) / len(realism_scores)) if realism_scores else None,
            "mean_coherence_score": (sum(coherence_scores) / len(coherence_scores)) if coherence_scores else None,
        },
    )

    manifest = StageArtifactManifest(
        stage_id="03",
        stage_name="run_opinion_attacks",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(enriched_jsonl),
        output_files=[abs_path(exposure_jsonl), abs_path(enriched_jsonl), abs_path(summary_json)],
        record_count=len(exposures),
        metadata={
            "n_attack": n_attack,
            "openrouter_model": config.openrouter_model,
            "review_rewrite_count": review_rewrite_count,
            "heuristic_fail_count": heuristic_fail_count,
        },
    )

    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 03 - Attack exposure generation")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openrouter-model", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-repair-iter", type=int, default=2)
    parser.add_argument(
        "--self-supervise-attack-realism",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--realism-threshold", type=float, default=0.72)
    parser.add_argument("--save-raw-llm", action="store_true", default=False)
    parser.add_argument("--raw-llm-dir", default=None)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)
    load_dotenv(Path(__file__).resolve().parents[5] / ".env")

    config = Stage03Config(
        stage_name="run_opinion_attacks",
        run_id=args.run_id,
        seed=args.seed,
        openrouter_model=args.openrouter_model,
        temperature=args.temperature,
        max_repair_iter=args.max_repair_iter,
        save_raw_llm=args.save_raw_llm,
        raw_llm_dir=args.raw_llm_dir,
        timeout_sec=args.timeout_sec,
        self_supervise_attack_realism=args.self_supervise_attack_realism,
        realism_threshold=args.realism_threshold,
    )

    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 03 completed: %s records", manifest.record_count)


if __name__ == "__main__":
    main()
