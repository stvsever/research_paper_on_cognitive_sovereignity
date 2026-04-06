from __future__ import annotations

"""
Technical overview
------------------
This module contains the realism layer that sits between pure ontology sampling
and the LLM-facing prompt payloads. Its job is to turn abstract leaf-node
choices into scenario context that is realistic enough to constrain the
simulation, without pretending to be the final analytical susceptibility model.

The functions here do two main things:
- derive lightweight heuristic context used to bound plausible opinion movement
- build attack-side prompt context and heuristic checks for downstream review

Important distinction:
- `heuristic_shift_sensitivity_proxy` and `resilience_index` are legacy realism
  helpers used for boundedness and prompt guidance
- they are not the analysis-facing susceptibility construct used in Stage 06

So this module is about keeping scenarios plausible and reviewable, not about
estimating the final moderation answer to the research question.
"""

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


def compute_shift_sensitivity_proxy(profile: ProfileConfiguration) -> float:
    values = profile.continuous_attributes
    neuro = values.get("big_five_neuroticism_mean_pct", 50.0) / 100.0
    consc = values.get("big_five_conscientiousness_mean_pct", 50.0) / 100.0
    open_ = values.get("big_five_openness_to_experience_mean_pct", 50.0) / 100.0
    age = values.get("age_years", 42.0)
    age_factor = 1.0 - bounded(age / 80.0, 0.0, 1.0)

    score = 0.45 * neuro + 0.25 * (1.0 - consc) + 0.2 * age_factor + 0.1 * (1.0 - open_)
    return round(bounded(score, 0.0, 1.0), 4)


def compute_susceptibility_index(profile: ProfileConfiguration) -> float:
    return compute_shift_sensitivity_proxy(profile)


def compute_resilience_index(profile: ProfileConfiguration) -> float:
    return round(1.0 - compute_shift_sensitivity_proxy(profile), 4)


def profile_context_snapshot(profile: ProfileConfiguration) -> Dict[str, float | str]:
    values = profile.continuous_attributes
    return {
        "sex": profile.categorical_attributes.get("sex", "Unknown"),
        "age_years": float(values.get("age_years", 42.0)),
        "big_five_neuroticism_mean_pct": float(values.get("big_five_neuroticism_mean_pct", 50.0)),
        "big_five_conscientiousness_mean_pct": float(values.get("big_five_conscientiousness_mean_pct", 50.0)),
        "big_five_openness_to_experience_mean_pct": float(values.get("big_five_openness_to_experience_mean_pct", 50.0)),
        "heuristic_shift_sensitivity_proxy": compute_shift_sensitivity_proxy(profile),
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
    adversarial_direction: int = 0,
) -> Dict[str, object]:
    """Build attack context for the exposure generator.

    adversarial_direction: +1 (adversary wants score to INCREASE), -1 (wants DECREASE), 0 (unspecified).
    This is critical: if not provided the persuasion goal will be direction-blind and attacks will
    systematically reinforce existing positions rather than shift them toward the adversary's goal.
    """
    domain = extract_opinion_domain(opinion_leaf)
    leaf_label = extract_leaf_label(opinion_leaf)
    context = profile_context_snapshot(profile)
    shift_sensitivity_proxy = float(context["heuristic_shift_sensitivity_proxy"])
    openness = float(context["big_five_openness_to_experience_mean_pct"])
    neuroticism = float(context["big_five_neuroticism_mean_pct"])
    conscientiousness = float(context["big_five_conscientiousness_mean_pct"])

    if domain == "Defense_and_National_Security":
        issue_frame = "national security, civil liberties, deterrence, alliance burden-sharing, and public preparedness"
    elif domain == "Foreign_Policy_and_Geopolitics":
        issue_frame = "alliances, strategic autonomy, diplomacy, sanctions, and international order"
    else:
        issue_frame = "migration, citizenship, border governance, and social cohesion"

    # ── Direction-aware persuasion goal ──────────────────────────────────────
    # Determine whether the baseline is aligned with (reinforcement) or opposed
    # to (erosion / shift) the adversarial goal.  This is the most important
    # context signal for the attack generator: without it the LLM defaults to
    # reinforcing whatever the baseline says, which systematically defeats the
    # adversary when the baseline is in the "wrong" direction.
    #
    # Alignment: baseline sign matches adversarial_direction (e.g. positive
    # baseline AND adversary wants increase, or negative baseline AND adversary
    # wants decrease).  Misalignment: they conflict.
    baseline_sign = 0 if baseline_score == 0 else (1 if baseline_score > 0 else -1)
    baseline_aligned = (adversarial_direction != 0) and (baseline_sign == adversarial_direction or baseline_sign == 0)
    baseline_misaligned = (adversarial_direction != 0) and (baseline_sign != 0) and (baseline_sign != adversarial_direction)
    baseline_strength = abs(baseline_score)

    if adversarial_direction == 0:
        # No direction info — fall back to neutral framing
        if baseline_strength < 180:
            persuasion_goal = "nudge an ambivalent opinion into a clearer directional stance"
        elif baseline_strength < 500:
            persuasion_goal = "modestly intensify an existing leaning"
        else:
            persuasion_goal = "gently reinforce an existing strong opinion"
    elif baseline_misaligned:
        # Adversary wants to shift the person AGAINST their current position
        if baseline_strength < 300:
            persuasion_goal = (
                f"shift a weakly {'positive' if baseline_sign > 0 else 'negative'} opinion "
                f"toward a {'more positive' if adversarial_direction > 0 else 'more negative'} stance "
                f"on {leaf_label.replace('_', ' ')} through targeted framing and selective evidence"
            )
        else:
            persuasion_goal = (
                f"gradually erode a strong {'pro' if baseline_sign > 0 else 'anti'} position on "
                f"{leaf_label.replace('_', ' ')} — introduce doubt, reframe costs/risks, and soften "
                f"conviction without triggering overt reactance; aim to move the score "
                f"{'upward' if adversarial_direction > 0 else 'downward'} even if the shift is modest"
            )
    elif baseline_aligned:
        # Adversary and baseline agree — reinforce and amplify
        if baseline_strength < 300:
            persuasion_goal = (
                f"intensify and solidify an already {'favorable' if baseline_sign > 0 else 'negative'} "
                f"opinion on {leaf_label.replace('_', ' ')} — push it further in the "
                f"{'positive' if adversarial_direction > 0 else 'negative'} direction"
            )
        else:
            persuasion_goal = (
                f"reinforce and deepen the existing strong "
                f"{'support for' if baseline_sign > 0 else 'opposition to'} {leaf_label.replace('_', ' ')}"
            )
    else:
        persuasion_goal = "nudge the opinion in a direction consistent with the adversarial goal"

    # ── Direction-aware emotional trigger ────────────────────────────────────
    # Use the adversarial direction rather than the baseline sign so the attack
    # uses the right emotional lever for the goal (not just for the status quo).
    if adversarial_direction < 0:
        # Adversary wants to reduce / undermine → use threat, doubt, failure framing
        if neuroticism >= 60:
            emotional_trigger = "threat salience, uncertainty, and fear of negative outcomes"
        elif conscientiousness >= 60:
            emotional_trigger = "systemic failure, unreliability, and unintended negative consequences"
        else:
            emotional_trigger = "loss framing, social doubt, and elite distrust"
    elif adversarial_direction > 0:
        # Adversary wants to increase → use identity affirmation, opportunity, pride framing
        if neuroticism >= 60:
            emotional_trigger = "safety reassurance and identity-protective certainty"
        elif conscientiousness >= 60:
            emotional_trigger = "duty, principle-alignment, and ordered benefit framing"
        elif openness >= 60:
            emotional_trigger = "pluralism, opportunity, and reputational self-image"
        else:
            emotional_trigger = "social proof, everyday fairness, and consensus cues"
    else:
        emotional_trigger = "threat" if baseline_score < 0 else "identity_reinforcement"

    # ── Direction-aware motivational lever ───────────────────────────────────
    if adversarial_direction != 0 and baseline_misaligned:
        # Need to erode the opposite-direction position — use the lever most likely
        # to bypass the persona's existing defence of that position
        if neuroticism >= 60:
            motivational_lever = "anxiety amplification and worst-case scenario framing"
        elif conscientiousness >= 60:
            motivational_lever = "rule-violation and systemic-risk framing to undermine confidence"
        elif openness >= 60:
            motivational_lever = "alternative expert perspective and complexity-acknowledgement"
        else:
            motivational_lever = "social-proof erosion and elite-distrust nudge"
    else:
        if neuroticism >= 60:
            motivational_lever = "threat salience and uncertainty reduction"
        elif conscientiousness >= 60:
            motivational_lever = "order, duty, and rule-compliance framing"
        elif openness >= 60:
            motivational_lever = "pluralism, complexity, and reputational self-image"
        else:
            motivational_lever = "everyday fairness, trust, and social-proof cues"

    # ── Shift band — wider when misaligned because stronger push needed ───────
    band_multiplier = 1.4 if baseline_misaligned and baseline_strength >= 400 else 1.0
    shift_dir_label = "upward" if adversarial_direction > 0 else ("downward" if adversarial_direction < 0 else "any")

    context.update(
        {
            "opinion_domain": domain,
            "opinion_leaf_label": leaf_label,
            "attack_leaf_label": extract_leaf_label(attack_leaf),
            "baseline_score": baseline_score,
            "adversarial_direction": adversarial_direction,
            "adversarial_direction_label": (
                f"INCREASE (score must move {shift_dir_label}, toward +1000)"
                if adversarial_direction > 0
                else (
                    f"DECREASE (score must move {shift_dir_label}, toward -1000)"
                    if adversarial_direction < 0
                    else "UNSPECIFIED"
                )
            ),
            "baseline_vs_goal": (
                "ALIGNED — baseline already points in the adversarial direction; reinforce it"
                if baseline_aligned
                else (
                    "MISALIGNED — baseline opposes the adversarial goal; the attack must work AGAINST the current opinion"
                    if baseline_misaligned
                    else "AMBIVALENT — baseline near zero; push in adversarial direction"
                )
            ),
            "likely_emotional_trigger": emotional_trigger,
            "issue_frame": issue_frame,
            "motivational_lever": motivational_lever,
            "persuasion_goal": persuasion_goal,
            "recommended_shift_band": {
                "direction": shift_dir_label,
                "low": round((20 + 40 * shift_sensitivity_proxy) * band_multiplier, 1),
                "typical": round((40 + 120 * shift_sensitivity_proxy) * band_multiplier, 1),
                "upper": round((90 + 210 * shift_sensitivity_proxy) * band_multiplier, 1),
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
    shift_sensitivity_proxy: float,
    adversarial_direction: int = 0,
) -> Dict[str, object]:
    delta = post_score - baseline_score
    max_shift = 80.0 + (280.0 * intensity_hint) + (220.0 * shift_sensitivity_proxy)
    if not attack_present:
        max_shift = 120.0

    baseline_sign = 0 if baseline_score == 0 else (1 if baseline_score > 0 else -1)
    post_sign = 0 if post_score == 0 else (1 if post_score > 0 else -1)
    strong_baseline = abs(baseline_score) >= 500
    strong_reversal = baseline_sign != 0 and post_sign != 0 and baseline_sign != post_sign and abs(post_score) >= 250

    # A "reversal" is only truly implausible if it goes AGAINST the adversarial
    # direction — i.e. the attack pushed the opinion the wrong way AND it was
    # large.  If the reversal is IN the adversarial direction, it is exactly what
    # we want and should not be penalised.
    delta_sign = 0 if delta == 0 else (1 if delta > 0 else -1)
    reversal_in_adversarial_direction = (adversarial_direction != 0) and (delta_sign == adversarial_direction)
    implausible_reversal = (
        strong_baseline
        and strong_reversal
        and abs(delta) > max_shift * 0.75
        and not reversal_in_adversarial_direction  # allow direction-aligned large shifts
    )

    checks: Dict[str, bool] = {
        "within_scale": -1000 <= post_score <= 1000,
        "high_resolution": abs(post_score) % 50 != 0,
        "bounded_shift": abs(delta) <= max_shift,
        "control_stability": True if attack_present else abs(delta) <= 120,
        # A neutral repeated measure on a high-resolution scale should not collapse into exact cloning
        # across the entire control arm. Small drift keeps the counterfactual more realistic.
        "control_not_exact_clone": True if attack_present else abs(delta) >= 3,
        "no_implausible_reversal": not implausible_reversal,
    }
    checks["overall_pass"] = all(checks.values())
    return {
        "checks": checks,
        "pass_count": sum(1 for key, value in checks.items() if key != "overall_pass" and value),
        "delta": delta,
        "max_reasonable_shift": round(max_shift, 3),
    }
