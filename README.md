# Ontology-Constrained Multi-Agent Simulation for Adversarial Opinion Susceptibility Auditing

Ontology-driven multi-agent simulation pipeline for attacked-only political-opinion experiments, interactive dashboards, publication assets, and automated manuscript generation.

## Study design

- **25 profiles × 4 attack vectors × 10 opinion leaves = 1,000 scenarios**
- Profile ontology: Big Five personality traits, SES, Political Psychology, Social Context, and demographics
- Attack coverage: one leaf per cognitive-warfare mechanism family (Social Media Misinformation, Coordinated Amplification, Manipulative Persuasion, Synthetic Media Generation)
- Analytical focus: conditional susceptibility estimation, hierarchical feature importance, profile feature network diagnostics, and structural equation modelling

> **Data note.** The checked-in opinion data was generated with deterministic fallbacks because API credits were exhausted during the original run. The network analysis, SEM, and hierarchical importance results are valid; substantive persuasion claims require regenerating stages 02–05 with live API calls.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env   # then set OPENROUTER_API_KEY
```

## Run the pipeline

Full pipeline:

```bash
./scripts/run_10.sh
```

Rebuild analytics / visuals / paper only (after code or data changes):

```bash
./.venv/bin/python src/backend/pipeline/full/run_full_pipeline.py \
  --output-root evaluation/run_10 \
  --run-id run_10 \
  --n-profiles 25 \
  --seed 100 \
  --attack-ratio 1.0 \
  --attack-leaves Misleading_Narrative_Framing,Astroturf_Comment_Wave,Fear_Appeal_Scapegoating_Post,LLM_Chatbot_Personalized_Persuasion \
  --max-opinion-leaves 10 \
  --profile-candidate-multiplier 3 \
  --primary-moderator posthoc_profile_susceptibility_index \
  --bootstrap-samples 200 \
  --use-test-ontology \
  --ontology-root src/backend/ontology/separate/test \
  --openrouter-model mistralai/mistral-small-3.2-24b-instruct \
  --temperature 0.15 \
  --max-repair-iter 3 \
  --profile-generation-mode deterministic \
  --self-supervise-attack-realism \
  --realism-threshold 0.70 \
  --self-supervise-opinion-coherence \
  --coherence-threshold 0.70 \
  --generate-visuals \
  --export-static-figures \
  --build-report \
  --resume-from-stage 06 \
  --stop-after-stage 09
```

Regenerate opinion data after recharging API credits:

```bash
# same flags but resume from stage 02
  --resume-from-stage 02 --stop-after-stage 09
```

## Outputs

| Path | Contents |
|---|---|
| `evaluation/run_10/stage_outputs/07_generate_research_visuals/interactive_sem_dashboard.html` | Interactive multi-tab dashboard |
| `evaluation/run_10/paper/publication_assets/` | Static figures and supplementary tables |
| `research_report/assets/` | Figures and tables linked into the manuscript |
| `research_report/report/main.pdf` | Compiled manuscript |

## Dashboard tabs

- **Ontology Explorer** — ontology leaf inventory with optional UMAP embedding
- **Factorial surfaces** — 2D and 3D attack × opinion effectivity heatmaps
- **Baseline / Post** — individual-level opinion shift scatter
- **Attack comparison** — per-attack Δ distributions
- **SEM network** — structural equation model path diagram
- **Conditional Susceptibility Estimator** — profile configurator with bootstrap rank uncertainty
- **Hierarchical importance** — Random Forest importance decomposed by ontology-aligned feature groups
- **Profile Network Explorer** — interactive correlation graph with:
  - Global metrics panel (density, modularity, diameter, clustering, etc.)
  - Local node metrics (degree, strength, eigenvector centrality, PageRank, betweenness, closeness, participation coefficient, bridge ratio, within-module Z, k-core)
  - Community convex hulls, hub/bridge presets
  - Force-directed layout with draggable nodes
  - Wheel zoom and background pan
  - Lasso multi-select
  - Shortest-path analysis (BFS)
  - Ontology family / feature-type / community / sign filters
  - SVG export and mini-map
- **Robustness diagnostics** — bootstrap rank stability, SEM sensitivity, audit trails

## Repo layout

```text
├── evaluation/
│   └── run_10/                      # pipeline outputs
├── research_report/
│   ├── assets/                      # manuscript figures and tables
│   └── report/                      # LaTeX source and compiled PDF
├── scripts/
│   └── run_10.sh                    # canonical run script
└── src/backend/
    ├── ontology/separate/test/      # PROFILE / ATTACK / OPINION ontologies
    ├── pipeline/full/               # orchestrator
    ├── pipeline/separate/           # per-stage scripts (01–09)
    └── utils/                       # dashboard, publication assets, SEM, etc.
```

## License

MIT. See [LICENSE](LICENSE).
