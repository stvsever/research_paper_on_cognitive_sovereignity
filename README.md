# PILOT: Inter-individual Differences in Susceptibility to Cyber-manipulation

Multi-agent Simulation Approach with High-dimension State Space of Political Opinions

Paper License: MIT · Python 3.11+ · Docker

Stijn Van Severen<sup>1,*</sup> · Thomas De Schryver<sup>1</sup>

<sup>1</sup> Ghent University · <sup>*</sup> Corresponding author

## Table of Contents

- Abstract
- Key Pilot Outputs
- Full Paper
- Repository Structure
- Setup & Installation
- Usage
- Pipeline Overview
- Figures & Tables
- Citation
- License

## Abstract

This repository contains the backend research pipeline, evaluation outputs, and manuscript assets for a pilot study on how inter-individual differences may moderate susceptibility to cyber-manipulation in political opinion spaces. The workflow represents `PROFILE`, `ATTACK`, and `OPINION` as explicit ontologies, generates ontology-constrained scenarios, obtains baseline and post-exposure opinion estimates with structured LLM agents, and estimates moderation through a parsimonious structural equation model supplemented by robust regression and bootstrap intervals.

The current codebase is designed as a methodological pilot rather than a claim-ready empirical study. Its purpose is to validate the architecture: ontology handling, structured agent prompting, self-supervised realism and coherence review, long-table SEM construction, interactive result inspection, print-ready figure generation, and automated manuscript compilation.

## Key Pilot Outputs

- `evaluation/run_1/`: initial backend pilot with test ontologies and baseline moderation reporting
- `evaluation/run_2/`: improved realism/coherence checks plus interactive HTML SEM dashboard
- `evaluation/run_3/`: main pilot run with 20 scenarios, publication assets, and LaTeX manuscript build
- `research_report/report/main.pdf`: compiled pilot manuscript
- `research_report/assets/`: paper-ready figures and tables copied from the pipeline

Current `run_3` pilot summary:

- `n = 20` scenarios, with `10` treated and `10` control
- mean treated delta: `+6.5`
- mean control delta: `0.0`
- mean treated attack realism score: `0.78`
- primary moderation term (`profile_cont_susceptibility_index`): robust OLS estimate `-21.71`, `p = 0.045`
- inferential caveat: SEM fit remains poor (`CFI = 0.080`, `RMSEA = 1.817`) and bootstrap `95%` interval for the interaction still crosses zero, so the pilot is workflow-validating rather than claim-establishing

Interpretive constraint: the pilot is intentionally underpowered and should be read as workflow validation, not as a substantive estimate of real-world causal susceptibility.

## Full Paper

The manuscript is built directly from the generated pilot outputs:

- PDF (typeset): [research_report/report/main.pdf](research_report/report/main.pdf)
- LaTeX source: [research_report/report/main.tex](research_report/report/main.tex)
- Paper assets: [research_report/assets](research_report/assets)

## Repository Structure

```text
Paper_CaseStudiesAnalysisExperimentalData/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .env.example
├── .gitignore
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
│
├── evaluation/
│   ├── run_1/                  # First backend pilot
│   ├── run_2/                  # Improved realism + HTML dashboard
│   └── run_3/                  # Main pilot run (20 scenarios)
│
├── research_report/
│   ├── assets/                 # Paper figures and tables
│   └── report/                 # main.tex, references.bib, main.pdf
│
└── src/
    ├── backend/
    │   ├── agentic_framework/  # OpenRouter client, base agent, prompts
    │   ├── ontology/           # Separate PROFILE / ATTACK / OPINION ontologies
    │   ├── pipeline/
    │   │   ├── full/           # Full orchestration entrypoint
    │   │   └── separate/       # Independently runnable stages 01-09
    │   ├── utils/              # Ontology, SEM, visuals, paper utilities
    │   └── requirements.txt
    └── frontend/               # Reserved for later interactive UI work
```

## Setup & Installation

### Option A — Local

```bash
# 1. Clone the repository
git clone https://github.com/stvsever/research_paper_on_cognitive_sovereignity.git
cd research_paper_on_cognitive_sovereignity

# 2. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure the environment
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env
```

### Option B — Docker

```bash
# 1. Clone the repository
git clone https://github.com/stvsever/research_paper_on_cognitive_sovereignity.git
cd research_paper_on_cognitive_sovereignity

# 2. Configure the environment
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env

# 3. Launch the pilot pipeline
cd docker
OPENROUTER_MODEL=mistralai/mistral-small-3.2-24b-instruct docker compose up --build
```

By default, the Docker entrypoint runs the full pilot configuration for `evaluation/run_3` and writes manuscript outputs to `research_report/report/`.

## Usage

### Run the full pilot pipeline locally

```bash
python src/backend/pipeline/full/run_full_pipeline.py \
  --output-root evaluation/run_3 \
  --run-id run_3 \
  --n-scenarios 20 \
  --seed 126 \
  --attack-ratio 0.5 \
  --attack-leaf "ATTACK_VECTORS > Social_Media_Misinformation > Misleading_Narrative_Framing" \
  --use-test-ontology \
  --openrouter-model mistralai/mistral-small-3.2-24b-instruct \
  --temperature 0.2 \
  --max-repair-iter 2 \
  --profile-generation-mode deterministic \
  --self-supervise-attack-realism \
  --realism-threshold 0.72 \
  --self-supervise-opinion-coherence \
  --coherence-threshold 0.72 \
  --generate-visuals \
  --export-static-figures \
  --build-report \
  --primary-moderator profile_cont_susceptibility_index \
  --bootstrap-samples 500 \
  --run-stage-checks \
  --paper-title "PILOT: Inter-individual Differences in Susceptibility to Cyber-manipulation: A Multi-agent Simulation Approach with High-dimension State Space of Political Opinions" \
  --report-root research_report/report \
  --report-assets-root research_report/assets
```

### Run individual stages

Each stage under `src/backend/pipeline/separate/` is independently runnable. The main stages are:

- `01_create_scenarios`
- `02_assess_baseline_opinions`
- `03_run_opinion_attacks`
- `04_assess_post_attack_opinions`
- `05_compute_effectivity_deltas`
- `06_construct_structural_equation_model`
- `07_generate_research_visuals`
- `08_generate_publication_assets`
- `09_build_research_report`

## Pipeline Overview

| Step | Stage | Purpose |
|---|---|---|
| 1/9 | Create scenarios | Build ontology-constrained pilot scenarios with balanced attack/control assignment |
| 2/9 | Assess baseline opinions | Estimate pre-exposure opinion scores from the profile-conditioned prompt |
| 3/9 | Run opinion attacks | Generate treated exposures and neutral control paths, then audit realism |
| 4/9 | Assess post-attack opinions | Re-estimate the same opinion after exposure and audit coherence |
| 5/9 | Compute effectivity deltas | Compute `post - baseline`, encode moderators, and build SEM tables |
| 6/9 | Construct SEM | Fit the main moderation SEM, robust OLS, and bootstrap summaries |
| 7/9 | Generate research visuals | Create the interactive HTML dashboard and exploratory plots |
| 8/9 | Generate publication assets | Export print-ready figures/tables in `PDF`, `PNG`, `CSV`, and `TeX` |
| 9/9 | Build research report | Compile the full LaTeX manuscript with Tectonic |

## Figures & Tables

Main publication figures are copied into `research_report/assets/figures/` and include:

- `figure_1_study_design`
- `figure_2_attack_control_delta_distribution`
- `figure_3_primary_moderation_interaction`
- `figure_4_moderator_coefficient_forest`

Supplementary figures include:

- `supplementary_figure_s1_baseline_post_scatter`
- `supplementary_figure_s2_attack_quality`
- `supplementary_figure_s3_scenario_composition`
- `supplementary_figure_s4_sem_overview`

Main and supplementary tables are copied into `research_report/assets/tables/`, each with a title, label, and note for direct manuscript inclusion.

Interactive inspection outputs are written to:

- `evaluation/run_2/visuals/interactive_sem_dashboard.html`
- `evaluation/run_3/visuals/interactive_sem_dashboard.html`

## Citation

If you use this code, outputs, or manuscript material, cite:

### APA 7

Van Severen, S., & De Schryver, T. (2026). *PILOT: Inter-individual differences in susceptibility to cyber-manipulation: A multi-agent simulation approach with high-dimension state space of political opinions*. Ghent University. https://github.com/stvsever/research_paper_on_cognitive_sovereignity

### BibTeX

```bibtex
@article{vanseveren2026cognitivepilot,
  title        = {PILOT: Inter-individual Differences in Susceptibility to Cyber-manipulation: A Multi-agent Simulation Approach with High-dimension State Space of Political Opinions},
  author       = {Van Severen, Stijn and De Schryver, Thomas},
  year         = {2026},
  institution  = {Ghent University},
  url          = {https://github.com/stvsever/research_paper_on_cognitive_sovereignity}
}
```

A machine-readable citation is also available in `CITATION.cff`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
