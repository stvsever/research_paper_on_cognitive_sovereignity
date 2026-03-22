#!/usr/bin/env bash
set -euo pipefail

cd /app

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required}"
: "${OPENROUTER_MODEL:?OPENROUTER_MODEL is required}"

extra_flags=()

if [[ "${RUN_STAGE_CHECKS:-true}" == "true" ]]; then
  extra_flags+=(--run-stage-checks)
fi

if [[ "${GENERATE_VISUALS:-true}" == "true" ]]; then
  extra_flags+=(--generate-visuals)
else
  extra_flags+=(--no-generate-visuals)
fi

if [[ "${EXPORT_STATIC_FIGURES:-true}" == "true" ]]; then
  extra_flags+=(--export-static-figures)
else
  extra_flags+=(--no-export-static-figures)
fi

if [[ "${BUILD_REPORT:-true}" == "true" ]]; then
  extra_flags+=(--build-report)
else
  extra_flags+=(--no-build-report)
fi

if [[ "${SELF_SUPERVISE_ATTACK_REALISM:-true}" == "true" ]]; then
  extra_flags+=(--self-supervise-attack-realism)
else
  extra_flags+=(--no-self-supervise-attack-realism)
fi

if [[ "${SELF_SUPERVISE_OPINION_COHERENCE:-true}" == "true" ]]; then
  extra_flags+=(--self-supervise-opinion-coherence)
else
  extra_flags+=(--no-self-supervise-opinion-coherence)
fi

python src/backend/pipeline/full/run_full_pipeline.py \
  --output-root "${OUTPUT_ROOT:-evaluation/run_3}" \
  --run-id "${RUN_ID:-run_3}" \
  --n-scenarios "${N_SCENARIOS:-20}" \
  --seed "${PIPELINE_SEED:-126}" \
  --attack-ratio "${ATTACK_RATIO:-0.5}" \
  --attack-leaf "${ATTACK_LEAF:-ATTACK_VECTORS > Social_Media_Misinformation > Misleading_Narrative_Framing}" \
  --use-test-ontology \
  --openrouter-model "${OPENROUTER_MODEL}" \
  --temperature "${TEMPERATURE:-0.2}" \
  --max-repair-iter "${MAX_REPAIR_ITER:-2}" \
  --profile-generation-mode "${PROFILE_GENERATION_MODE:-deterministic}" \
  --realism-threshold "${REALISM_THRESHOLD:-0.72}" \
  --coherence-threshold "${COHERENCE_THRESHOLD:-0.72}" \
  --primary-moderator "${PRIMARY_MODERATOR:-profile_cont_susceptibility_index}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-500}" \
  --paper-title "${PAPER_TITLE:-PILOT: Inter-individual Differences in Susceptibility to Cyber-manipulation: A Multi-agent Simulation Approach with High-dimension State Space of Political Opinions}" \
  --report-root "${REPORT_ROOT:-research_report/report}" \
  --report-assets-root "${REPORT_ASSETS_ROOT:-research_report/assets}" \
  "${extra_flags[@]}" \
  "$@"
