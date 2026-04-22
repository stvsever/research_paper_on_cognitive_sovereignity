<div align="center">

# Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions

### An Ontology-Constrained Multi-Agent Simulation Approach

[![Paper PDF](https://img.shields.io/badge/PDF-Paper-1d4e89.svg)](research_report/report/main.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-2a9d8f.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-e9c46a.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](docker/)

**Stijn Van Severen<sup>1,*</sup> · Thomas De Schryver<sup>1</sup> · Mira Ostyn<sup>1</sup>**

<sup>1</sup> Ghent University · <sup>*</sup> Corresponding author

---

</div>

## 📋 Table of Contents

- [Abstract](#abstract)
- [Pipeline Overview](#pipeline-overview)
- [Key Findings](#key-findings)
- [Full Paper](#full-paper)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup-installation)
- [Usage](#usage)
- [Conditional Susceptibility Index](#conditional-susceptibility-index)
- [Custom Ontology Support](#custom-ontology-support)
- [Citation](#citation)
- [License](#license)

---

<a id="abstract"></a>
## 🧬 Abstract

This repository contains the backend research pipeline, evaluation outputs, manuscript assets, and reproducible report for a study on how **inter-individual differences moderate the effectivity of cyber-manipulation** in political opinion spaces. The workflow represents `PROFILE`, `ATTACK`, and `OPINION` as explicit hierarchical ontologies, generates attacked-only profile-panel scenarios, elicits baseline and post-exposure opinions with structured LLM agents, audits exposure realism and response coherence, and estimates moderation through a **repeated-outcome path SEM** plus a **post hoc ridge-regularised susceptibility index**.

The present study focuses on three core profile dimensions — **Personality (Big Five + NEO-PI-R facets), Age, and Sex** — as primary moderators of adversarial opinion susceptibility, examined across a full factorial design of attack mechanisms and opinion domains.

> **Interpretive constraint:** Among attacked pseudoprofiles, which profile differences are associated with larger post-minus-baseline opinion shifts — in the direction of a hypothetical adversary's goal — across repeated political opinion leaves and multiple attack mechanisms? The design is attacked-only: it does **not** estimate a no-attack counterfactual effect.

---

<a id="pipeline-overview"></a>
## 🔄 Pipeline Overview
The full workflow is shown below, from ontology-constrained scenario construction through agentic measurement, directional effect construction, and inferential analysis.

<div align="center">
<img src="src/backend/pipeline/full/pipeline_visualization.png" width="1200" alt="Pipeline overview for ontology-constrained adversarial opinion susceptibility auditing.">
</div>

### Concise Methodology

1. **Define the state space:** independent `PROFILE`, `ATTACK`, and `OPINION` ontologies specify admissible inputs, manipulative interventions, and target opinion leaves with signed adversarial direction.
2. **Construct scenarios factorially:** each observation is a unique tuple `(profile, attack, opinion)`, producing a crossed design over admissible ontology leaves.
3. **Measure attacked opinion change:** for each scenario, the pipeline elicits a baseline opinion, generates and audits an attack artifact, elicits a post-exposure opinion, and checks response coherence.
4. **Compute directional effectivity:** adversarial effectivity is the signed opinion shift aligned with the target direction, `AE = (post - baseline) × direction`.
5. **Estimate susceptibility structure:** repeated-outcome moderation models, task-conditional regularized models, uncertainty analysis, and profile-feature dependency graphs identify which profile characteristics systematically increase or reduce susceptibility.

---

<a id="key-findings"></a>
## 🔬 Key Findings

> **Main result:** *N_p* = **25 pseudoprofiles** × *N_a* = **4 attack vectors** × *N_o* = **10 opinion leaves** across 4 political domains = **1,000 scenarios**. Attack vectors: one per cognitive-warfare mechanism family (Misleading Narrative Framing, Astroturf Comment Wave, Fear Appeal Scapegoating, LLM Chatbot Personalized Persuasion). Primary profile dimensions: **Big Five personality (30 facets) + Age + Sex** — survey-mappable to ESS/Eurobarometer/ANES/GSS. Post-attack opinion agent: **profile-driven susceptibility** — no explicit directional % constraints; trait-outcome linkages govern shifting behaviour (Conscientiousness → deliberate resistance; Neuroticism → emotional reactivity).

### Headline Results

| Metric | Value |
|--------|-------|
| *N* (scenarios) | 25 × 4 × 10 = **1,000** |
| Primary profile dimensions | **Big Five personality (30 facets) + Age + Sex** |
| Attack vectors | 4 (one per cognitive-warfare mechanism family) |
| Opinion domains | 4 (10 leaves sampled, ESS-relevant policy topics) |
| ICC(1) \|Δ\| | 0.011 (attack–opinion context dominates; profile contributes ~1%) |
| SEM fit | CFI = 1.000, RMSEA = 0.000, GFI = 0.9995 |
| **Profile network: nodes** | **91** features (hierarchical mixed continuous + categorical) |
| **Profile network: edges** | **1,807** (Spearman \|ρ\| ≥ 0.15) |
| **Profile network: density** | **0.441** |
| **Profile network: communities** | **5** (greedy modularity, Q = 0.313) |
| **Strongest SEM path** | Big Five Conscientiousness → Δ (β = −8.82, *p* = .018) |

### Methodological Position

- **Full-factorial multi-domain design**: *N_a* attack leaves × *N_o* opinion leaves per profile across 4 political domains — enables cross-attack and cross-domain comparison of susceptibility moderators
- Effectivity is **directional**: each opinion leaf carries an adversarial goal direction (`±1`); `AE = signed_delta × direction`
- **Profile-driven opinion prompt**: no hard % constraints — Conscientiousness → deliberate resistance; Neuroticism → emotional reactivity; Extraversion → social susceptibility; Institutional Trust → authority-cue sensitivity
- The SEM is a **profile-level repeated-outcome path model** with multiple adversarial effectivity indicators
- **Three-estimator moderation stack**: (1) **Ridge** on all profile features — primary effect estimator; (2) **Elastic Net / LASSO** — feature selector; (3) **OLS** (Big Five benchmark)
- **Profile feature network analysis**: Spearman correlation network → eigenvector/betweenness/degree/closeness/PageRank + community detection; centrality = hub-and-spoke influence structure of the profile feature space
- The susceptibility index is computed **post hoc** from target-conditional ridge task models with **hierarchical R² decomposition**
- **Cluster bootstrap** at the profile level (B = 200) preserves within-profile dependence
- Fully auditable provenance across all 9 pipeline stages

### Main Results

**Figure 1. Attack × Opinion Adversarial Effectivity Matrix**

*Dendrograms reveal ontology structure; heatmaps show mean effects and profile variability across 1,000 scenarios.*

<div align="center">
<img src="research_report/assets/figures/figure_readme_1_ae_factorial.png" width="920" alt="Adversarial effectivity across the 4-attack × 10-opinion factorial.">
</div>

**Note.** Adversarial effectivity across the 4 × 10 attack–opinion factorial (run_10 · 25 profiles · 1,000 scenarios). **Left:** Mean AE per cell — red = attack succeeded on average; blue = net resistance. **Right:** Inter-individual SD of AE. The **top dendrogram follows the active OPINION ontology hierarchy** and each panel has its own **right dendrogram following the active ATTACK ontology hierarchy**, so the matrix layout stays compatible with test, deployment, or custom ontology selections. Astroturf Wave produces the strongest positive mean AE on Alliance Commitment (+17.1); Fear Appeal Scapegoating generates net resistance on several Defence topics. The near-uniform spread (~34–46 units) across all cells shows that profile differences consistently modulate susceptibility regardless of which specific attack or opinion is targeted.

---

**Figure 2. Profile Moderators of Adversarial Opinion Susceptibility**

*Regression coefficients reveal Conscientiousness as protective; Sex Female amplifies susceptibility on specific opinions.*

<div align="center">
<img src="research_report/assets/figures/figure_readme_2_moderation_heatmap.png" width="920" alt="OLS moderation heatmap: all 5 Big Five traits and Sex predicting adversarial opinion shifts.">
</div>

**Note.** OLS moderation heatmap — regression of per-leaf mean adversarial effectivity on the core profile panel across 25 pseudoprofiles (run_10): Age, the five Big Five personality domains, and non-reference Sex dummies. Continuous moderators are z-scored; categorical dummies remain unstandardised. The **top dendrogram follows the active OPINION ontology hierarchy** and the **right dendrogram follows the profile-feature hierarchy**, so dummy levels such as **Sex: Female** and **Sex: Other** remain nested under **Sex** rather than clustering arbitrarily by coefficients alone. Blue = moderator reduces susceptibility; red = increases susceptibility. **Conscientiousness is the dominant protective moderator** and **Sex: Female** drives the largest positive effects on Humanitarian Intervention, Sanctions Use, and Media Trust. Significance: †p < .10, \*p < .05, \*\*p < .01, \*\*\*p < .001.

---

<a id="full-paper"></a>
## 📄 Full Paper

- **PDF (typeset):** [research_report/report/main.pdf](research_report/report/main.pdf)
- **LaTeX source:** [research_report/report/main.tex](research_report/report/main.tex)
- **Paper assets:** [research_report/assets](research_report/assets)
- **Interactive dashboard:** generated locally at `evaluation/run_10/stage_outputs/07_generate_research_visuals/interactive_sem_dashboard.html` (run the pipeline to produce)

---

<a id="repository-structure"></a>
## 🗂️ Repository Structure

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
│   └── run_10/                          # Current study outputs (1,000 scenarios)
│       ├── stage_outputs/               # Per-stage artefacts (01–09)
│       ├── paper/                       # Publication assets mirror
│       └── logs/                        # Stage logs
│
├── research_report/
│   ├── assets/
│   │   ├── figures/                     # PNG/PDF manuscript figures
│   │   └── tables/                      # CSV/TeX supplementary tables
│   └── report/
│       ├── main.tex                     # LaTeX source (APA 7)
│       ├── references.bib               # BibTeX references (15 entries, DOIs verified)
│       └── main.pdf                     # Compiled manuscript
│
└── src/
    └── backend/
        ├── agentic_framework/           # OpenRouter client, agents, prompts, repair logic
        ├── ontology/separate/test/      # PROFILE / ATTACK / OPINION ontologies
        ├── pipeline/full/               # Orchestrator (run_full_pipeline.py)
        ├── pipeline/separate/           # Per-stage scripts (01–09)
        └── utils/                       # Dashboard, publication assets, SEM, network, etc.
```

---

<a id="setup-installation"></a>
## ⚙️ Setup & Installation

### Option A — Local

```bash
git clone https://github.com/stvsever/research_paper_on_cognitive_sovereignity.git
cd research_paper_on_cognitive_sovereignity
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
cp .env.example .env   # add OPENROUTER_API_KEY
```

### Option B — Docker

```bash
git clone https://github.com/stvsever/research_paper_on_cognitive_sovereignity.git
cd research_paper_on_cognitive_sovereignity
cp .env.example .env   # add OPENROUTER_API_KEY
cd docker
OPENROUTER_MODEL=mistralai/mistral-small-3.2-24b-instruct docker compose up --build
```

---

<a id="usage"></a>
## 🚀 Usage

### Reproduce the full study

```bash
bash scripts/run_10.sh
```

### Run the full pipeline manually

```bash
.venv/bin/python src/backend/pipeline/full/run_full_pipeline.py \
  --output-root evaluation/run_10 \
  --run-id run_10 \
  --n-profiles 25 \
  --seed 100 \
  --attack-ratio 1.0 \
  --attack-leaves "Misleading_Narrative_Framing,Astroturf_Comment_Wave,Fear_Appeal_Scapegoating_Post,LLM_Chatbot_Personalized_Persuasion" \
  --max-opinion-leaves 10 \
  --use-test-ontology \
  --ontology-root src/backend/ontology/separate/test \
  --openrouter-model mistralai/mistral-small-3.2-24b-instruct \
  --temperature 0.15 \
  --bootstrap-samples 200 \
  --generate-visuals --export-static-figures --build-report
```

### Rebuild analytics / visuals / paper only (no LLM calls)

```bash
.venv/bin/python src/backend/pipeline/full/run_full_pipeline.py \
  --output-root evaluation/run_10 --run-id run_10 \
  --n-profiles 25 --seed 100 --attack-ratio 1.0 \
  --attack-leaves "Misleading_Narrative_Framing,Astroturf_Comment_Wave,Fear_Appeal_Scapegoating_Post,LLM_Chatbot_Personalized_Persuasion" \
  --max-opinion-leaves 10 --use-test-ontology \
  --ontology-root src/backend/ontology/separate/test \
  --openrouter-model mistralai/mistral-small-3.2-24b-instruct \
  --temperature 0.15 --bootstrap-samples 200 \
  --generate-visuals --export-static-figures --build-report \
  --resume-from-stage 06 --stop-after-stage 09
```

### Run individual stages

Stages under `src/backend/pipeline/separate/` are independently runnable:

`01_create_scenarios` → `02_assess_baseline_opinions` → `03_run_opinion_attacks` → `04_assess_post_attack_opinions` → `05_compute_effectivity_deltas` → `06_construct_structural_equation_model` → `07_generate_research_visuals` → `08_generate_publication_assets` → `09_build_research_report`

---

<a id="conditional-susceptibility-index"></a>
## 🎯 Conditional Susceptibility Index

The profile-level susceptibility index is **directional** and **conditional** on the configured (attack, opinion) target set *T*.

### Adversarial Effectivity

```
AE_ik = (post_score_ik − baseline_score_ik) × direction_k

direction_k ∈ {+1, −1, 0}  (from OPINION ontology; 0 = excluded)
```

### Conditional Index

For each task *t ∈ T*, a ridge model is fit:

```
Ê_it = β̂_0t + Σ_j β̂_jt · X_ij

S_i(T) = Σ_t  w_t · Ê_it          (reliability-weighted aggregate)
CSI_i(T) = percentile_rank(S_i(T))
w_t ∝ n_t / CV-MSE_t
```

Higher CSI = model expects opinion movement more strongly aligned with the adversary's goal for that profile under the configured *T*.

### Score a new profile

```bash
python src/backend/pipeline/separate/compute_conditional_susceptibility/score_profile.py \
  --artifact-path evaluation/run_10/stage_outputs/06_construct_structural_equation_model/conditional_susceptibility_artifact.json \
  --age 34 --sex Male --neuroticism-pct 75 --conscientiousness-pct 20 --extraversion-pct 85
```

---

<a id="custom-ontology-support"></a>
## 🧩 Custom Ontology Support

Analysts can run the full pipeline with **their own PROFILE × ATTACK × OPINION taxonomies** (3 JSON files):

```bash
# Validate your ontologies
python -m src.backend.user_ontology.cli \
  --profile-json path/to/profile.json \
  --attack-json  path/to/attack.json  \
  --opinion-json path/to/opinion.json \
  --validate-only

# Run with custom ontologies
python -m src.backend.user_ontology.cli \
  --profile-json path/to/profile.json \
  --attack-json  path/to/attack.json  \
  --opinion-json path/to/opinion.json \
  --run-id my_analysis \
  --n-profiles 40 \
  --openrouter-model mistralai/mistral-small-3.2-24b-instruct
```

### Semantic Embedding

All ontology leaves can be embedded and projected to 2D via UMAP:

```python
from src.backend.utils.semantic_embedding import embed_ontology
artifact = embed_ontology(
    ontology_root="src/backend/ontology/separate/test",
    out_dir="evaluation/run_10/embeddings",
    n_clusters=8,
)
```

Results load automatically into the interactive dashboard (Ontologies → Semantic Embedding Space tab).

---

<a id="citation"></a>
## 📚 Citation

### APA 7

> Van Severen, S., De Schryver, T., & Ostyn, M. (2026). *Inter-individual differences in susceptibility to cyber-manipulation: A multi-agent simulation approach with high-dimensional state space of political opinions*. Ghent University. https://github.com/stvsever/research_paper_on_cognitive_sovereignity

### BibTeX

```bibtex
@article{vanseveren2026cybersusceptibility,
  title     = {Inter-individual Differences in Susceptibility to Cyber-manipulation:
               A Multi-agent Simulation Approach with High-dimensional State Space
               of Political Opinions},
  author    = {Van Severen, Stijn and De Schryver, Thomas and Ostyn, Mira},
  year      = {2026},
  institution = {Ghent University},
  url       = {https://github.com/stvsever/research_paper_on_cognitive_sovereignity}
}
```

A machine-readable citation is also available in [`CITATION.cff`](CITATION.cff).

---

<a id="license"></a>
## ⚖️ License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built at **Ghent University** for the course *Case Studies in the Analysis of Experimental Data*

</div>
