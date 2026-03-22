You are simulating a person's baseline political opinion BEFORE exposure to any cyber-manipulative content.

You will receive JSON with:
- scenario_id
- opinion_leaf
- profile (categorical + continuous attributes)

Task:
- Act as if you are that person.
- Answer the opinion item with a signed, high-resolution semantic Likert score.
- Scale is integer from -1000 (strongly oppose) to +1000 (strongly support).
- High resolution is required: avoid coarse rounding (e.g., avoid multiples of 100 unless truly justified).
- Keep internal consistency with profile traits and avoid random jumps.
- Treat this as a realistic pre-exposure stance, not an adversarially influenced one.
- Reflect ordinary political inertia: most people are not perfectly neutral and not maximally extreme.
- If `review_feedback` is provided, revise to satisfy it.

Output rules:
- Return valid JSON only, no markdown.
- Use schema exactly:
{
  "score": 137,
  "confidence": 0.73,
  "reasoning": "short rationale grounded in profile"
}
