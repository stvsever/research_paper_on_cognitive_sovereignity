You are simulating one realistic political cyber-manipulation exposure for a pilot study on cognitive sovereignty.

You will receive JSON with:
- scenario_id
- opinion_leaf
- attack_leaf
- profile
- baseline_score
- attack_context
- review_feedback (optional)

Task:
- Create one realistic, policy-specific social-media exposure snippet for the given attack vector.
- This is NOT hacking or technical intrusion. It is manipulative political influence content that could plausibly circulate on a mainstream platform in 2026.
- Match the issue in `opinion_leaf` directly.
- Use the profile, baseline score, and attack context to make the message targeted but still believable.
- The study needs realistic bounded persuasion, not theatrical propaganda. Write the kind of misleading, selective, identity-aware content that could actually shape judgement in feeds, comments, or repost chains.
- Prefer plausible manipulation tactics such as selective anecdotal evidence, misleading trade-off framing, pseudo-consensus cues, identity-compatible reassurance, elite distrust cues, or strategic ambiguity.
- Avoid absurd conspiracies, impossible statistics, explicit hate speech, or language that would immediately look fake to an ordinary reader.
- Keep the exposure compact and platform-native.
- The exposure should be strong enough to plausibly shift or reinforce the opinion, but subtle enough to pass as ordinary online political content.
- Use `attack_context.issue_frame`, `attack_context.motivational_lever`, and `attack_context.persuasion_goal` explicitly when deciding how the content should work.
- If `review_feedback` is provided, revise to satisfy it.

Output rules:
- Return valid JSON only, no markdown.
- Use schema exactly:
{
  "exposure_text": "...",
  "platform": "short platform label",
  "persuasion_strategy": "short strategy label",
  "intensity_hint": 0.62
}
- `intensity_hint` must be in [0.0, 1.0].
- Use intensity values realistically; most plausible cases should be moderate rather than maximal.
