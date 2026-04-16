#!/usr/bin/env bash
# ============================================================
# run_9.sh — Launch pipeline run 9 with extended test ontologies
#
# Extended attack taxonomy (18 leaves; 6 selected for this run):
#   - Misleading_Narrative_Framing        (social media misinformation)
#   - Fear_Appeal_Scapegoating_Post       (emotional manipulation)
#   - Astroturf_Comment_Wave              (coordinated amplification)
#   - Pseudo_Expert_Authority_Cue         (manipulative persuasion)
#   - LLM_Chatbot_Personalized_Persuasion (new: AI-based personalised attack)
#   - Deepfake_Audio_Speech_Impersonation (new: synthetic media)
#
# Extended opinion state space (62 leaves; 8 sampled across 3 domains):
#   - Defense_and_National_Security       (3 leaves)
#   - Information_Integrity_and_Platforms (3 leaves — new domain)
#   - Foreign_Policy_and_Geopolitics      (2 leaves)
#
# Profile ontology: 85 feature dimensions (Big Five + Dual Process +
#   Digital Literacy + Political Engagement + Demographics)
#
# Token budget estimate: ~6 attacks × ~8 opinions × 80 profiles × 3 LLM
#   calls per cell ≈ ~11500 calls × ~500 tok ≈ 5.75M tokens (Mistral Small)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Load .env for API key
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

ATTACK_LEAVES="Misleading_Narrative_Framing,Fear_Appeal_Scapegoating_Post,Astroturf_Comment_Wave,Pseudo_Expert_Authority_Cue,LLM_Chatbot_Personalized_Persuasion,Deepfake_Audio_Speech_Impersonation"

python src/backend/pipeline/full/run_full_pipeline.py \
    --output-root        "evaluation/run_9" \
    --run-id             "run_9" \
    --n-profiles         80 \
    --seed               99 \
    --attack-ratio       1.0 \
    --attack-leaves      "${ATTACK_LEAVES}" \
    --max-opinion-leaves 8 \
    --profile-candidate-multiplier 5 \
    --primary-moderator  "posthoc_profile_susceptibility_index" \
    --bootstrap-samples  600 \
    --use-test-ontology \
    --ontology-root      "src/backend/ontology/separate/test" \
    --openrouter-model   "mistralai/mistral-small-3.2-24b-instruct" \
    --temperature        0.15 \
    --max-repair-iter    2 \
    --profile-generation-mode deterministic \
    --self-supervise-attack-realism \
    --realism-threshold  0.76 \
    --self-supervise-opinion-coherence \
    --coherence-threshold 0.76 \
    --generate-visuals \
    --export-static-figures \
    --build-report \
    --resume-from-stage  "01" \
    --stop-after-stage   "09" \
    --save-raw-llm \
    --timeout-sec        90 \
    --max-concurrency    20 \
    --log-level          INFO \
    2>&1 | tee evaluation/logs/run_9.log

echo "=== run_9 complete ==="
