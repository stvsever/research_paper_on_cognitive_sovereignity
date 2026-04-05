<div align="center">

# Inter-individual Differences in Susceptibility to Cyber-manipulation

### Multi-agent Simulation Approach with High-dimensional State Space of Political Opinions

[![Paper PDF](https://img.shields.io/badge/PDF-Pilot_Report-1d4e89.svg)](research_report/report/main.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-2a9d8f.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-e9c46a.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](docker/)

**Stijn Van Severen<sup>1,*</sup> · Thomas De Schryver<sup>1</sup> · Mira Ostyn<sup>1</sup>**

<sup>1</sup> Ghent University · <sup>*</sup> Corresponding author

---

</div>

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Key Pilot Findings](#-key-pilot-findings)
- [Full Paper](#-full-paper)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Pipeline Overview](#-pipeline-overview)
- [Conditional Susceptibility Index](#-conditional-susceptibility-index)
- [Figures & Tables](#-figures--tables)
- [Citation](#-citation)
- [License](#-license)

---

## 📝 Abstract

This repository contains the backend research pipeline, evaluation outputs, manuscript assets, and reproducible pilot report for a study on how **inter-individual differences moderate the effectivity of cyber-manipulation** in political opinion spaces. The workflow represents `PROFILE`, `ATTACK`, and `OPINION` as explicit hierarchical ontologies, generates attacked-only profile-panel scenarios, elicits baseline and post-exposure opinions with structured LLM agents, audits exposure realism and response coherence, and estimates moderation through a **repeated-outcome path SEM** plus a **post hoc ridge-regularized susceptibility index**.

The current codebase is a **methodological pilot**, not a claim-ready population study. Its purpose is to validate the architecture end to end: leaf-only ontology sampling, mixed-type profile construction, platform-native attack generation, repeated attacked opinion design, multivariate moderation estimation, target-conditional susceptibility scoring, interactive result inspection, publication-asset export, and automated LaTeX manuscript compilation.

> **Interpretive constraint:** the current main run (`run_8`) addresses a narrower and cleaner question than the earlier pilots: **among attacked pseudoprofiles, which profile differences are associated with larger post-minus-baseline opinion shifts — in the direction of a hypothetical adversary's goal — across repeated political opinion leaves and multiple attack mechanisms?** It does **not** estimate a no-attack counterfactual effect.

---

## Key Pilot Findings

> **Main pilot result (`run_8`):** the multi-attack factorial design used **100 pseudoprofiles** crossed with **4 attack vectors** and **4 opinion leaves** (`n = 1,600` attacked rows). Attack vectors span three mechanism families: cognitive reframing (Misleading Narrative Framing), emotional manipulation (Fear Appeal Scapegoating), false consensus (Astroturf Comment Wave), and authority heuristic exploitation (Pseudo Expert Authority Cue). The primary effectivity outcome is **adversarial effectivity**: the signed opinion delta multiplied by each opinion leaf's pre-assigned adversarial goal direction. Statistical improvements include cluster bootstrap (profile-level resampling), FDR-corrected moderator p-values, intraclass correlation (ICC) diagnostics, and Cohen's d effect sizes.

### Headline Results

| Metric | Value |
|--------|-------|
| Mean \|delta\| | 43.16 (*SD* = 17.86) |
| Mean adversarial effectivity | -14.49 (*SD* = 44.40) |
| Positive AE rate | 32.6% of 1,600 scenarios |
| ICC(1) for \|delta\| | 0.052 |
| Top moderator (multivariate) | Conscientiousness (*b* = -3.26, *q*<sub>FDR</sub> = 0.030) |
| Hierarchical R² — Big Five | 32.0% relative importance |
| Hierarchical R² — Demographics | 28.8% relative importance |
| Conscientiousness share | 23.1% of relative importance |
| Most effective attack | Misleading Narrative Framing (AE = -11.98, 35.8% positive) |
| Most susceptible opinion | Strategic Autonomy Support (AE = +29.98, only positive mean) |
| Most resistant opinion | Domestic Security Powers Expansion (AE = -39.31) |
| SEM fit | CFI = 1.000, RMSEA = 0.000 |
| Mean attack realism | 0.71 |

### Methodological Position of `run_8`

- **Multi-attack factorial design**: 4 distinct ATTACK leaves × 4 OPINION leaves per profile, enabling cross-attack comparison of susceptibility moderators
- Effectivity is **directional**: each opinion leaf carries an adversarial goal direction (`+1`, `-1`, or `0`); `adversarial_effectivity = signed_delta × direction`
- The **adversarial operator goal** is maximizing aggregate erosion of defense cohesion, multilateral alliance commitment, and institutional security capacity — expressed as per-leaf direction annotations (`+1`, `−1`) in the OPINION ontology
- The SEM is a **profile-level repeated-outcome path model** with multiple attacked effectivity indicators
- The susceptibility index is computed **post hoc** from target-conditional ridge task models with **hierarchical R-squared decomposition**
- **Cluster bootstrap** at the profile level preserves within-profile dependence in inference
- **Benjamini-Hochberg FDR correction** for multiple moderator comparisons
- **ICC(1)** diagnostics for all outcome variables characterize between- vs. within-profile variance
- The pipeline produces **27+ research-grade visualizations** including baseline-to-post opinion state space transitions and per-attack comparison panels
- Fully auditable provenance: baseline scores, exposure texts, realism review, post-exposure scores, coherence review, SEM outputs, conditional susceptibility artifact

### Previous Run: `run_7` (Single-Attack Baseline)

Run 7 used the same 100 profiles with a single attack leaf (Misleading Narrative Framing) and 4 opinion leaves (400 scenarios). Mean |delta| = 43.36, mean adversarial effectivity = -8.46 (40% positive rate). Top moderator weights: Sex Male (8.5%), Age (8.5%), Sex Female (6.9%), Neuroticism Impulsiveness (6.0%). Hierarchical R-squared: Personality (33.1%), Demographics (24.1%), Agreeableness (20.4%).

### Conditional Susceptibility Scoring CLI

Score any new pseudoprofile against the fitted run artifact:

```bash
python src/backend/pipeline/separate/compute_conditional_susceptibility/score_profile.py \
  --artifact-path evaluation/run_8/stage_outputs/06_construct_structural_equation_model/conditional_susceptibility_artifact.json \
  --age 24 --sex Female --neuroticism-pct 84 --conscientiousness-pct 22
```

The CLI outputs a full `.txt` report with hierarchical opinion-domain and feature-group breakdowns, including the profile's CSI percentile rank within the training sample.

### Main Figures

<div align="center">
<img src="research_report/assets/figures/figure_3_profile_moderator_coefficient_forest.png" width="760" alt="Descriptive susceptibility weights across profile moderators.">

*Figure 3. Descriptive susceptibility weights across profile moderators in `run_8`, showing how the post hoc susceptibility index is decomposed over age, sex, and personality terms under the modeled multi-attack/opinion target set. The primary outcome is adversarial effectivity.*
</div>

<div align="center">
<img src="research_report/assets/figures/figure_4_annotated_sem_path_diagram.png" width="760" alt="Repeated-outcome path-SEM coefficient matrix for attacked opinion shifts.">

*Figure 4. Repeated-outcome path-SEM coefficients from profile moderators to attacked opinion shifts in `run_8`. Four attack vectors are crossed with four opinion leaves in a full factorial design; cells show how profile terms relate to each attacked adversarial-effectivity outcome.*
</div>

---

## 📖 Full Paper

The manuscript is built directly from the current pilot outputs:

- **PDF (typeset):** [research_report/report/main.pdf](research_report/report/main.pdf)
- **LaTeX source:** [research_report/report/main.tex](research_report/report/main.tex)
- **Report summary:** [research_report/report/report_summary.json](research_report/report/report_summary.json)
- **Paper assets:** [research_report/assets](research_report/assets)
- **Interactive dashboard (`run_8`):** generated locally at `evaluation/run_8/stage_outputs/07_generate_research_visuals/interactive_sem_dashboard.html` but intentionally not tracked in git to keep the repository lean

---

## 📁 Repository Structure

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
│   ├── run_1/                        # Initial mixed-condition pilot
│   ├── run_2/                        # Realism/coherence upgrades + dashboard
│   ├── run_3/                        # First publication bundle
│   ├── run_4/                        # Transitional redesign pilot
│   ├── run_5/                        # First attacked-only pilot
│   ├── run_6/                        # 50-profile repeated-outcome pilot (abs_delta_score)
│   ├── run_7/                        # 100-profile single-attack adversarial effectivity pilot
│   └── run_8/                        # Current 100-profile multi-attack factorial pilot (4 attacks × 4 opinions)
│
├── research_report/
│   ├── assets/
│   │   ├── figures/                  # PNG/PDF manuscript figures
│   │   └── tables/                   # CSV/TeX manuscript tables
│   └── report/
│       ├── main.tex
│       ├── references.bib
│       └── main.pdf
│
└── src/
    ├── backend/
    │   ├── agentic_framework/        # OpenRouter client, agents, prompts, repair logic
    │   ├── ontology/
    │   │   └── separate/
    │   │       └── test/             # PROFILE / ATTACK / OPINION test ontologies
    │   ├── pipeline/
    │   │   ├── full/                 # Full orchestration entrypoint
    │   │   └── separate/             # Independently runnable stages 01-09
    │   ├── utils/                    # Ontology, SEM, visualization, and report utilities
    │   └── requirements.txt
    └── frontend/                     # Reserved for later interactive UI work
```

> **Note:** the repository is intentionally backend-first. The current primary deliverables are the attacked-only evaluation runs, the run 8 manuscript, and the reusable methodological pipeline.

---

## ⚙️ Setup & Installation

### 🔧 Option A — Local

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

### 🐳 Option B — Docker

```bash
# 1. Clone the repository
git clone https://github.com/stvsever/research_paper_on_cognitive_sovereignity.git
cd research_paper_on_cognitive_sovereignity

# 2. Configure the environment
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env

# 3. Launch the current pilot workflow
cd docker
OPENROUTER_MODEL=mistralai/mistral-small-3.2-24b-instruct docker compose up --build
```

By default, the Docker entrypoint runs the current pilot configuration for `evaluation/run_8/` and writes manuscript outputs to `research_report/report/`.

---

## 🚀 Usage

### Run the current full pilot pipeline locally

**Multi-attack factorial design (run_8):**

```bash
python src/backend/pipeline/full/run_full_pipeline.py \
  --output-root evaluation/run_8 \
  --run-id run_8 \
  --n-profiles 100 \
  --seed 88 \
  --attack-ratio 1.0 \
  --attack-leaves "Misleading_Narrative_Framing,Fear_Appeal_Scapegoating_Post,Astroturf_Comment_Wave,Pseudo_Expert_Authority_Cue" \
  --focus-opinion-domain Defense_and_National_Security \
  --max-opinion-leaves 4 \
  --profile-candidate-multiplier 5 \
  --use-test-ontology \
  --openrouter-model mistralai/mistral-small-3.2-24b-instruct \
  --temperature 0.15 \
  --max-repair-iter 2 \
  --profile-generation-mode deterministic \
  --self-supervise-attack-realism \
  --realism-threshold 0.76 \
  --self-supervise-opinion-coherence \
  --coherence-threshold 0.76 \
  --generate-visuals \
  --export-static-figures \
  --build-report \
  --bootstrap-samples 800 \
  --max-concurrency 20 \
  --paper-title "PILOT: Inter-individual Differences in Susceptibility to Cyber-manipulation: A Multi-agent Simulation Approach with High-dimensional State Space of Political Opinions" \
  --report-root research_report/report \
  --report-assets-root research_report/assets
```

The `--attack-leaves` parameter accepts a comma-separated list of attack leaf names (matched case-insensitively against the ontology). The pipeline creates a full factorial design: every profile is crossed with every attack and every opinion leaf. For the configuration above: 100 profiles x 4 attacks x 4 opinions = 1,600 scenarios.

**Single-attack design (backward compatible):**

```bash
python src/backend/pipeline/full/run_full_pipeline.py \
  --output-root evaluation/run_7 \
  --run-id run_7 \
  --n-profiles 100 \
  --seed 77 \
  --attack-ratio 1.0 \
  --attack-leaf "ATTACK_VECTORS > Social_Media_Misinformation > Misleading_Narrative_Framing" \
  --focus-opinion-domain Defense_and_National_Security \
  --max-opinion-leaves 4 \
  --use-test-ontology \
  --openrouter-model mistralai/mistral-small-3.2-24b-instruct \
  --generate-visuals --export-static-figures --build-report
```

### Run individual stages

Each stage under `src/backend/pipeline/separate/` is independently runnable:

- `01_create_scenarios`
- `02_assess_baseline_opinions`
- `03_run_opinion_attacks`
- `04_assess_post_attack_opinions`
- `05_compute_effectivity_deltas`
- `06_construct_structural_equation_model`
- `07_generate_research_visuals`
- `08_generate_publication_assets`
- `09_build_research_report`

---

## 🔄 Pipeline Overview

```mermaid
flowchart TD
    P["PROFILE ontology\ncontinuous + categorical leaves\ngeneric inventory discovery"] --> S
    O["OPINION ontology\nrepeated sampled leaves\n+ adversarial direction per leaf"] --> S
    A["ATTACK ontology\nmulti-leaf factorial\n3 mechanism families"] --> S

    S["Scenario design\nprofile × attack × opinion factorial\n100 × 4 × 4 = 1600 scenarios"] --> B["Baseline opinion agent\nscore in [-1000, 1000]"]
    B --> E["Attack exposure agent\nplatform-native manipulative message"]
    E --> R["Realism reviewer\nthreshold + repair loop"]
    R --> P2["Post-exposure opinion agent\nsame leaf, same scale"]
    P2 --> C["Coherence reviewer\nplausibility + boundedness checks"]
    C --> D["Adversarial effectivity\nA_ik = delta_ik × direction_k\nICC, cluster bootstrap, FDR correction"]
    D --> W["Profile-panel wide table\nmulti-attack → repeated adversarial outcomes"]
    W --> SEM["Repeated-outcome path SEM\nmoderator → effectivity indicators"]
    W --> RIDGE["Target-conditional ridge task models\npost hoc susceptibility index\nhierarchical R² decomposition"]
    SEM --> OUT["Paper figures, tables, PDF report\n27+ research-grade visualizations"]
    RIDGE --> OUT
    RIDGE --> CLI["Conditional susceptibility CLI\nscore new profiles against fitted artifact"]
```

---

## 🧮 Conditional Susceptibility Index

From `run_8`, the profile-level susceptibility index is **directional** and **conditional** on both the attack vectors and opinion leaves being modeled.

### Adversarial Effectivity Outcome

Each opinion leaf in the OPINION ontology carries a pre-assigned adversarial goal direction:

```text
direction ∈ {+1, −1, 0}
  +1  adversary wants this opinion score to increase
  −1  adversary wants this opinion score to decrease
   0  directionally ambiguous — excluded from adversarial scoring
```

The adversarial effectivity for profile `i` on opinion leaf `k` is:

```text
adversarial_effectivity_ik = signed_delta_ik × direction_k
```

Positive adversarial effectivity means the profile's opinion moved in the direction the adversary intended. Zero-direction leaves are assigned a neutral direction of 1 (no directional contribution). This replaces raw `|Δ|` as the primary susceptibility outcome.

### Conditional Index Formulation

Let the configured target set be:

```text
T = {(attack_leaf, opinion_leaf)}
```

For each task `t in T`, the pipeline fits a regularized profile-only ridge model on observed adversarial effectivity:

```text
E_hat_it = beta_hat_0t + sum over features j of [ beta_hat_jt * X_ij ]
```

Aggregated with reliability weights and converted to percentile rank:

```text
S_i(T)   = sum over tasks t in T of [ w_t * E_hat_it ]
CSI_i(T) = percentile_rank( S_i(T) )
w_t      ∝ n_t / CV-MSE_t     (task reliability; normalized)
```

Higher `CSI_i(T)` means the fitted model expects opinion movement more strongly aligned with the adversary's goal for that profile under the configured attack/opinion target set.

### Hierarchical Feature Decomposition

The fitted task models also compute a leave-one-group-out marginal CV-R² decomposition across an ontology-aligned feature hierarchy:

- **Demographics** — age, sex
- **Personality (overall)** — all Big Five features
- **Per-trait groups** — Neuroticism, Openness, Conscientiousness, Extraversion, Agreeableness

This decomposition answers questions like: "how much of the cross-profile susceptibility variance is explained by demographic differences versus personality trait differences?"

### Implementation

- callable utility: [src/backend/utils/conditional_susceptibility.py](src/backend/utils/conditional_susceptibility.py)
- profile scoring CLI: [src/backend/pipeline/separate/compute_conditional_susceptibility/score_profile.py](src/backend/pipeline/separate/compute_conditional_susceptibility/score_profile.py)
- Stage 06 saves a reusable fitted artifact:
  - `conditional_susceptibility_artifact.json`
  - `conditional_susceptibility_task_coefficients.csv`
  - `conditional_susceptibility_task_summary.csv`
  - `conditional_susceptibility_hierarchical_decomposition.json`

Minimal fit-and-score usage:

```python
import pandas as pd
from src.backend.utils.conditional_susceptibility import (
    fit_conditional_susceptibility_index,
    score_profiles_with_conditional_artifact,
)

long_df = pd.read_csv("evaluation/run_8/stage_outputs/05_compute_effectivity_deltas/sem_long_encoded.csv")

fit = fit_conditional_susceptibility_index(
    long_df,
    outcome_metric="adversarial_effectivity",
    seed=42,
    compute_hierarchy=True,
)

artifact = fit.artifact
profile_scores, breakdown = score_profiles_with_conditional_artifact(
    long_df[["profile_id", *artifact.feature_columns]].drop_duplicates(),
    artifact,
)
```

Score a new profile config via the CLI:

```bash
python src/backend/pipeline/separate/compute_conditional_susceptibility/score_profile.py \
  --artifact-path evaluation/run_8/stage_outputs/06_construct_structural_equation_model/conditional_susceptibility_artifact.json \
  --config evaluation/run_8/compute_conditional_susceptibility/profile_high_susceptibility.json \
  --output-dir evaluation/run_8/compute_conditional_susceptibility/
```

Important constraint:

- the analysis-facing susceptibility construct is the post hoc **conditional susceptibility index**, not any a priori resilience proxy
- susceptibility is **conditional on the target set**: a profile can rank highly under one attack/opinion configuration and not under another

---

## 📊 Figures & Tables

Main publication figures are copied into `research_report/assets/figures/`:

- `figure_1_study_design`
- `figure_2_absolute_delta_distribution`
- `figure_3_profile_moderator_coefficient_forest`
- `figure_4_annotated_sem_path_diagram`

Supplementary figures include:

- `supplementary_figure_s1_baseline_post_scatter`
- `supplementary_figure_s2_profile_effectivity_heatmap`
- `supplementary_figure_s3_susceptibility_distribution`
- `supplementary_figure_s4_attack_comparison_panel`

Main tables are copied into `research_report/assets/tables/`:

- `table_1_pilot_design_and_configuration`
- `table_2_attacked_effectivity_descriptive_statistics`
- `table_3_multivariate_profile_moderator_model`

Supplementary tables include:

- `supplementary_table_s1_ontology_leaves_used`
- `supplementary_table_s2_moderator_comparison`
- `supplementary_table_s3_assumption_and_risk_register`
- `supplementary_table_s4_reproducibility_manifest`
- `supplementary_table_s5_sem_path_coefficients`

Interactive inspection outputs for the main pilot are written locally to:

- `evaluation/run_8/stage_outputs/07_generate_research_visuals/interactive_sem_dashboard.html`

These generated HTML dashboards are intentionally excluded from git so the repository remains centered on the Python pipeline, manuscript assets, and evaluation datasets rather than bulky browser-rendered output files.

---

## 📖 Citation

If you use this code, outputs, or manuscript material, cite:

### APA 7

> Van Severen, S., De Schryver, T., & Ostyn, M. (2026). *PILOT: Inter-individual differences in susceptibility to cyber-manipulation: A multi-agent simulation approach with high-dimensional state space of political opinions*. Ghent University. https://github.com/stvsever/research_paper_on_cognitive_sovereignity

### BibTeX

```bibtex
@article{vanseveren2026cognitivepilot,
  title        = {PILOT: Inter-individual Differences in Susceptibility to Cyber-manipulation: A Multi-agent Simulation Approach with High-dimensional State Space of Political Opinions},
  author       = {Van Severen, Stijn and De Schryver, Thomas and Ostyn, Mira},
  year         = {2026},
  institution  = {Ghent University},
  url          = {https://github.com/stvsever/research_paper_on_cognitive_sovereignity}
}
```

A machine-readable citation is also available in [`CITATION.cff`](CITATION.cff).

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this code for academic and non-academic use.

---

## Next Steps

### Ablation Studies

A rigorous ablation program is needed to isolate which components of the simulation pipeline contribute genuine measurement value and to establish the baseline against which the simulation should be compared.

**1. Simulation vs. direct one-shot elicitation (most critical ablation)**
For each profile in a matched human-data sample, collect both (a) a direct LLM susceptibility rating ("how susceptible is this profile?") and (b) the full simulation-derived CSI. Compare both against observed human opinion shifts under controlled adversarial exposure. This directly tests whether the staged multi-step pipeline produces better-calibrated, more generalizable susceptibility estimates than a scalar judgment — the central methodological claim. Without this ablation, the simulation's advantage over one-shot prompting remains asserted rather than demonstrated.

**How to implement**: Sample 50–100 real participants with matched profile features; collect baseline opinions, adversarial exposure, and post-exposure opinions on ontology-aligned items; compute observed adversarial effectivity as criterion. Compute: (i) demographics-only regression baseline, (ii) Big Five-only baseline, (iii) direct one-shot LLM rating, (iv) simulation-derived CSI. Evaluate by MAE / rank correlation against observed effectivity.

**2. Attack ontology ablation — which mechanism families drive variance?**
Hold profiles and opinions constant; systematically remove one attack family at a time (cognitive / emotional / social-proof / authority-heuristic). Measure change in: (a) inter-profile discriminability of the susceptibility index, (b) absolute shift variance, (c) moderator effect sizes. This identifies whether all four attack mechanisms contribute independent variance to susceptibility estimation or whether a leaner attack set is sufficient.

**3. Opinion-domain ablation — how domain-specific is the moderation pattern?**
Replace the Defense_and_National_Security opinion domain with alternative domains (health misinformation, financial fraud, civic trust erosion). If Conscientiousness remains the dominant moderator across domains, this supports a domain-general buffer interpretation. If moderation patterns change substantially, it indicates that susceptibility estimates are strongly domain-conditioned and cannot be generalized across opinion spaces without retraining.

**4. Profile-feature ablation — what is the minimum feature set for reliable CSI ranking?**
Systematically remove feature groups (demographics only → Big Five only → combined) and measure degradation in test-set rank correlation of the susceptibility index. Given the near-zero full-model CV-R², this ablation tests whether any feature combination produces reliable cross-profile discrimination, or whether prediction requires substantially larger n.

**5. LLM backend sensitivity — is susceptibility an artifact of training distribution?**
Run the same profile × attack × opinion factorial design across multiple LLM backends (e.g., Mistral, LLaMA, GPT-4o) with identical prompts and scoring. High cross-model consistency in susceptibility rankings would support construct validity; high inconsistency would indicate that susceptibility estimates reflect idiosyncratic LLM priors rather than simulated psychological dynamics.

### Scale-Up and Validation

- Expand to production ontologies with broader OPINION domains (immigration, foreign policy, economics) and additional ATTACK mechanism families
- Evaluate nonlinear susceptibility estimators (gradient-boosted trees, neural networks) behind the same scoring API
- Add multi-level susceptibility: attack-family-specific, opinion-domain-specific, and overall target-set scores
- Stress-test with larger profile panels (500+), more repeated opinion leaves, and cross-domain opinion transfer
- Collect human experimental data as ground-truth criterion for CSI validation (see Ablation 1 above)

### Deployment

- Stabilize the artifact contract for fitted conditional susceptibility models
- Add calibration and out-of-sample ranking diagnostics
- Increase concurrency, caching, and resumability for production-scale runs
- Build frontend dashboard for interactive susceptibility exploration

---

<div align="center">

Built at **Ghent University** 🏫 for course  '**Case Studies** in the **Analysis of Experimental Data**' 🎓

</div>
