from __future__ import annotations

from typing import Dict, List

from src.backend.utils.schemas import ProfileConfiguration


def extract_opinion_domain(opinion_leaf: str) -> str:
    parts = [part.strip() for part in opinion_leaf.split(">")]
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def extract_leaf_label(path: str) -> str:
    return path.split(">")[-1].strip()


def pretty_label(value: str) -> str:
    return value.replace("_", " ").strip()


def bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_susceptibility_index(profile: ProfileConfiguration) -> float:
    values = profile.continuous_attributes
    neuro = values.get("big_five_neuroticism_mean_pct", 50.0) / 100.0
    consc = values.get("big_five_conscientiousness_mean_pct", 50.0) / 100.0
    open_ = values.get("big_five_openness_to_experience_mean_pct", 50.0) / 100.0
    age = values.get("age_years", 42.0)
    age_factor = 1.0 - bounded(age / 80.0, 0.0, 1.0)

    score = 0.45 * neuro + 0.25 * (1.0 - consc) + 0.2 * age_factor + 0.1 * (1.0 - open_)
    return round(bounded(score, 0.0, 1.0), 4)


def compute_resilience_index(profile: ProfileConfiguration) -> float:
    return round(1.0 - compute_susceptibility_index(profile), 4)


def profile_context_snapshot(profile: ProfileConfiguration) -> Dict[str, float | str]:
    values = profile.continuous_attributes
    return {
        "sex": profile.categorical_attributes.get("sex", "Unknown"),
        "age_years": float(values.get("age_years", 42.0)),
        "big_five_neuroticism_mean_pct": float(values.get("big_five_neuroticism_mean_pct", 50.0)),
        "big_five_conscientiousness_mean_pct": float(values.get("big_five_conscientiousness_mean_pct", 50.0)),
        "big_five_openness_to_experience_mean_pct": float(values.get("big_five_openness_to_experience_mean_pct", 50.0)),
        "susceptibility_index": compute_susceptibility_index(profile),
        "resilience_index": compute_resilience_index(profile),
    }


def control_exposure_template(opinion_leaf: str) -> str:
    domain = pretty_label(extract_opinion_domain(opinion_leaf))
    leaf = pretty_label(extract_leaf_label(opinion_leaf))
    return (
        f"A neutral civic-information post summarizes balanced arguments about {domain}. "
        f"It explains policy background for {leaf} without persuasive or manipulative framing."
    )


def build_attack_context(
    opinion_leaf: str,
    attack_leaf: str,
    profile: ProfileConfiguration,
    baseline_score: int,
) -> Dict[str, object]:
    domain = extract_opinion_domain(opinion_leaf)
    leaf_label = extract_leaf_label(opinion_leaf)
    context = profile_context_snapshot(profile)
    susceptibility = float(context["susceptibility_index"])
    openness = float(context["big_five_openness_to_experience_mean_pct"])
    neuroticism = float(context["big_five_neuroticism_mean_pct"])
    conscientiousness = float(context["big_five_conscientiousness_mean_pct"])

    if domain == "Defense_and_National_Security":
        issue_frame = "national security, civil liberties, deterrence, alliance burden-sharing, and public preparedness"
    elif domain == "Foreign_Policy_and_Geopolitics":
        issue_frame = "alliances, strategic autonomy, diplomacy, sanctions, and international order"
    else:
        issue_frame = "migration, citizenship, border governance, and social cohesion"

    if neuroticism >= 60:
        motivational_lever = "threat salience and uncertainty reduction"
    elif conscientiousness >= 60:
        motivational_lever = "order, duty, and rule-compliance framing"
    elif openness >= 60:
        motivational_lever = "pluralism, complexity, and reputational self-image"
    else:
        motivational_lever = "everyday fairness, trust, and social-proof cues"

    baseline_strength = abs(baseline_score)
    if baseline_strength < 180:
        persuasion_goal = "nudge an ambivalent opinion into a clearer directional stance"
    elif baseline_strength < 500:
        persuasion_goal = "reinforce and modestly intensify an existing leaning"
    else:
        persuasion_goal = "reinforce an existing strong position rather than attempting an implausible reversal"

    context.update(
        {
            "opinion_domain": domain,
            "opinion_leaf_label": leaf_label,
            "attack_leaf_label": extract_leaf_label(attack_leaf),
            "baseline_score": baseline_score,
            "likely_emotional_trigger": "threat" if baseline_score < 0 else "identity_reinforcement",
            "issue_frame": issue_frame,
            "motivational_lever": motivational_lever,
            "persuasion_goal": persuasion_goal,
            "recommended_shift_band": {
                "low": round(20 + 40 * susceptibility, 1),
                "typical": round(40 + 120 * susceptibility, 1),
                "upper": round(90 + 210 * susceptibility, 1),
            },
            "paper_goal": (
                "Investigate how inter-individual differences moderate the effectivity of cyber-manipulation "
                "on cognitive sovereignty within a high-dimensional political opinion state space."
            ),
            "targeting_note": (
                "Use realistic platform-native misinformation tactics, preserve policy-topic anchoring, "
                "avoid impossible claims, and avoid generic propaganda language."
            ),
        }
    )
    return context


def assess_attack_exposure_heuristics(exposure_text: str, attack_leaf: str, opinion_leaf: str) -> Dict[str, object]:
    text = exposure_text.lower()
    checks: Dict[str, bool] = {
        "length_reasonable": 35 <= len(exposure_text) <= 900,
        "contains_issue_anchor": any(
            token in text
            for token in [
                extract_leaf_label(opinion_leaf).split("_")[0].lower(),
                extract_opinion_domain(opinion_leaf).split("_")[0].lower(),
            ]
        ),
        "contains_attack_theme": any(
            token in text for token in extract_leaf_label(attack_leaf).lower().split("_")[:2]
        ),
        "non_hacking_language": all(term not in text for term in ["malware", "exploit", "phishing kit", "ddos"]),
    }
    checks["overall_pass"] = sum(1 for value in checks.values() if value) >= 3
    return {
        "checks": checks,
        "pass_count": sum(1 for key, value in checks.items() if key != "overall_pass" and value),
    }


def assess_baseline_opinion_heuristics(score: int, confidence: float) -> Dict[str, object]:
    checks: Dict[str, bool] = {
        "within_scale": -1000 <= score <= 1000,
        "high_resolution": abs(score) % 50 != 0,
        "confidence_bounded": 0.0 <= confidence <= 1.0,
    }
    checks["overall_pass"] = all(checks.values())
    return {
        "checks": checks,
        "pass_count": sum(1 for key, value in checks.items() if key != "overall_pass" and value),
    }


def assess_post_opinion_heuristics(
    baseline_score: int,
    post_score: int,
    attack_present: bool,
    intensity_hint: float,
    susceptibility_index: float,
) -> Dict[str, object]:
    delta = post_score - baseline_score
    max_shift = 80.0 + (280.0 * intensity_hint) + (220.0 * susceptibility_index)
    if not attack_present:
        max_shift = 120.0

    baseline_sign = 0 if baseline_score == 0 else (1 if baseline_score > 0 else -1)
    post_sign = 0 if post_score == 0 else (1 if post_score > 0 else -1)
    strong_baseline = abs(baseline_score) >= 500
    strong_reversal = baseline_sign != 0 and post_sign != 0 and baseline_sign != post_sign and abs(post_score) >= 250

    checks: Dict[str, bool] = {
        "within_scale": -1000 <= post_score <= 1000,
        "high_resolution": abs(post_score) % 50 != 0,
        "bounded_shift": abs(delta) <= max_shift,
        "control_stability": True if attack_present else abs(delta) <= 120,
        # A neutral repeated measure on a high-resolution scale should not collapse into exact cloning
        # across the entire control arm. Small drift keeps the counterfactual more realistic.
        "control_not_exact_clone": True if attack_present else abs(delta) >= 3,
        "no_implausible_reversal": not (strong_baseline and strong_reversal and abs(delta) > max_shift * 0.75),
    }
    checks["overall_pass"] = all(checks.values())
    return {
        "checks": checks,
        "pass_count": sum(1 for key, value in checks.items() if key != "overall_pass" and value),
        "delta": delta,
        "max_reasonable_shift": round(max_shift, 3),
    }
