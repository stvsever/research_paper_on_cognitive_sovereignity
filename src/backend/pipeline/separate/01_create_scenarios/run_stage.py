from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.agentic_framework.factory import AgentFactory
from src.backend.utils.io import (
    abs_path,
    ensure_dir,
    env_get_required,
    stage_manifest_path,
    write_json,
    write_jsonl,
)
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.ontology_utils import (
    default_ontology_root,
    find_primary_node,
    flatten_leaf_paths,
    load_ontology_triplet,
)
from src.backend.utils.profile_sampling import sample_profile
from src.backend.utils.scenario_realism import extract_opinion_domain
from src.backend.utils.schemas import ProfileConfiguration, ScenarioRecord, StageArtifactManifest, StageConfig

LOGGER = logging.getLogger(__name__)


class Stage01Config(StageConfig):
    n_scenarios: int = 10
    attack_ratio: float = 0.5
    attack_leaf: Optional[str] = None
    profile_generation_mode: str = "deterministic"


def _resolve_attack_leaf(attack_leaves: List[str], configured_leaf: Optional[str]) -> str:
    if configured_leaf:
        if configured_leaf not in attack_leaves:
            raise ValueError(f"Configured attack leaf not found in ontology: {configured_leaf}")
        return configured_leaf
    for leaf in attack_leaves:
        if "misleading_narrative_framing" in leaf.lower():
            return leaf
    return attack_leaves[0]


def run_stage(input_path: str, output_dir: str, config: Stage01Config) -> StageArtifactManifest:
    del input_path
    output_root = ensure_dir(output_dir)

    project_root = Path(__file__).resolve().parents[5]
    ontology_root = Path(config.ontology_root) if config.ontology_root else default_ontology_root(
        project_root, config.use_test_ontology
    )

    ontologies = load_ontology_triplet(ontology_root)
    opinion_leaves = flatten_leaf_paths(ontologies["OPINION"])
    attack_leaves = flatten_leaf_paths(ontologies["ATTACK"])
    profile_tree = ontologies["PROFILE"]

    if not opinion_leaves:
        raise RuntimeError("No OPINION leaf nodes found")
    if not attack_leaves:
        raise RuntimeError("No ATTACK leaf nodes found")

    run_attack_leaf = _resolve_attack_leaf(attack_leaves, config.attack_leaf)

    profile_agent = None
    if config.profile_generation_mode.lower() in {"llm", "hybrid"}:
        if config.openrouter_model is None:
            raise RuntimeError("openrouter model is required for llm/hybrid profile generation")
        prompts_dir = project_root / "src" / "backend" / "agentic_framework" / "prompts"
        raw_dir = config.raw_llm_dir if config.save_raw_llm else None
        factory = AgentFactory(
            prompts_dir=prompts_dir,
            openrouter_api_key=env_get_required("OPENROUTER_API_KEY"),
            openrouter_model=config.openrouter_model,
            max_repair_iter=config.max_repair_iter,
            temperature=config.temperature,
            timeout_sec=config.timeout_sec,
            save_raw_dir=raw_dir,
        )
        profile_agent = factory.profile_generation_agent()

    n_attack = int(round(config.n_scenarios * config.attack_ratio))
    n_attack = max(0, min(config.n_scenarios, n_attack))
    n_control = config.n_scenarios - n_attack

    sampled_profiles = []

    def llm_generator(
        profile_id: str,
        seed: int,
        profile_leaf_nodes: List[str],
        deterministic_seed_profile: ProfileConfiguration,
    ) -> Optional[ProfileConfiguration]:
        if profile_agent is None:
            return None
        try:
            return profile_agent.generate(
                run_id=config.run_id,
                call_id=f"{profile_id}_generate",
                profile_id=profile_id,
                seed=seed,
                profile_leaf_nodes=profile_leaf_nodes,
                deterministic_seed_profile=deterministic_seed_profile,
            )
        except Exception as exc:
            LOGGER.warning("LLM profile generation fallback for %s: %s", profile_id, exc)
            return None

    for idx in range(config.n_scenarios):
        scenario_seed = config.seed + idx * 9973
        profile_id = f"profile_{idx + 1:04d}"

        profile_result = sample_profile(
            profile_tree=profile_tree,
            profile_id=profile_id,
            seed=scenario_seed,
            generation_mode=config.profile_generation_mode,
            llm_generator=llm_generator,
        )

        sampled_profiles.append(
            {
                "scenario_index": idx,
                "scenario_seed": scenario_seed,
                "profile_result": profile_result,
            }
        )

    sampled_profiles.sort(
        key=lambda item: item["profile_result"].profile.continuous_attributes.get("susceptibility_index", 0.5)
    )

    attack_assignments = [False] * config.n_scenarios
    attack_count = 0
    for pair_start in range(0, len(sampled_profiles), 2):
        pair_indices = list(range(pair_start, min(pair_start + 2, len(sampled_profiles))))
        assign_attack_to = pair_indices[(pair_start // 2) % len(pair_indices)] if pair_indices else None
        if assign_attack_to is not None and attack_count < n_attack:
            attack_assignments[assign_attack_to] = True
            attack_count += 1

    if attack_count < n_attack:
        for idx in range(len(sampled_profiles) - 1, -1, -1):
            if not attack_assignments[idx]:
                attack_assignments[idx] = True
                attack_count += 1
                if attack_count >= n_attack:
                    break

    scenarios: List[ScenarioRecord] = []
    for ordered_idx, profile_bundle in enumerate(sampled_profiles):
        profile_result = profile_bundle["profile_result"]
        scenario_seed = int(profile_bundle["scenario_seed"])
        opinion_leaf = opinion_leaves[ordered_idx % len(opinion_leaves)]
        attack_present = attack_assignments[ordered_idx]
        attack_leaf = run_attack_leaf if attack_present else None

        scenarios.append(
            ScenarioRecord(
                scenario_id=f"scenario_{ordered_idx + 1:04d}",
                scenario_index=ordered_idx,
                random_seed=scenario_seed,
                profile=profile_result.profile,
                opinion_leaf=opinion_leaf,
                attack_present=attack_present,
                attack_leaf=attack_leaf,
                attack_primary_node=find_primary_node(run_attack_leaf) if attack_present else None,
                metadata={
                    "profile_sampling_mode": profile_result.sampling_mode_used,
                    "run_attack_leaf": run_attack_leaf,
                    "opinion_domain": extract_opinion_domain(opinion_leaf),
                    "scenario_locale": "Belgium_Flanders",
                    "scenario_year": 2026,
                    "sampling_strategy": "susceptibility_stratified_balanced_pilot",
                    "susceptibility_index": profile_result.profile.continuous_attributes.get(
                        "susceptibility_index"
                    ),
                },
            )
        )

    scenarios_jsonl = output_root / "scenarios.jsonl"
    scenarios_json = output_root / "scenarios.json"
    ontology_catalog = output_root / "ontology_leaf_catalog.json"

    write_jsonl(scenarios_jsonl, (s.model_dump() for s in scenarios))
    write_json(scenarios_json, [s.model_dump() for s in scenarios])
    write_json(
        ontology_catalog,
        {
            "ontology_root": abs_path(ontology_root),
            "opinion_leaf_count": len(opinion_leaves),
            "attack_leaf_count": len(attack_leaves),
            "selected_attack_leaf": run_attack_leaf,
            "opinion_leaves": opinion_leaves,
            "attack_leaves": attack_leaves,
        },
    )

    manifest = StageArtifactManifest(
        stage_id="01",
        stage_name="create_scenarios",
        primary_output_path=abs_path(scenarios_jsonl),
        output_files=[
            abs_path(scenarios_jsonl),
            abs_path(scenarios_json),
            abs_path(ontology_catalog),
        ],
        record_count=len(scenarios),
        metadata={
            "n_attack": n_attack,
            "n_control": n_control,
            "attack_ratio": config.attack_ratio,
            "profile_generation_mode": config.profile_generation_mode,
            "use_test_ontology": config.use_test_ontology,
            "ontology_root": abs_path(ontology_root),
            "sampling_strategy": "susceptibility_stratified_balanced_pilot",
        },
    )

    write_json(stage_manifest_path(output_root), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 01 - Create scenarios")
    parser.add_argument("--input-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-scenarios", type=int, default=10)
    parser.add_argument("--attack-ratio", type=float, default=0.5)
    parser.add_argument("--attack-leaf", default=None)
    parser.add_argument("--profile-generation-mode", default="deterministic", choices=["deterministic", "llm", "hybrid"])
    parser.add_argument("--use-test-ontology", action="store_true", default=False)
    parser.add_argument("--ontology-root", default=None)
    parser.add_argument("--openrouter-model", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-repair-iter", type=int, default=2)
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

    config = Stage01Config(
        stage_name="create_scenarios",
        run_id=args.run_id,
        seed=args.seed,
        n_scenarios=args.n_scenarios,
        attack_ratio=args.attack_ratio,
        attack_leaf=args.attack_leaf,
        profile_generation_mode=args.profile_generation_mode,
        use_test_ontology=args.use_test_ontology,
        ontology_root=args.ontology_root,
        openrouter_model=args.openrouter_model,
        temperature=args.temperature,
        max_repair_iter=args.max_repair_iter,
        save_raw_llm=args.save_raw_llm,
        raw_llm_dir=args.raw_llm_dir,
        timeout_sec=args.timeout_sec,
    )

    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 01 completed: %s records", manifest.record_count)


if __name__ == "__main__":
    main()
