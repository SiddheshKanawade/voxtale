# Calibrating the story judges

Stories are ranked by a panel of three LLM judges on different models (`tools/judge_panel.py`, step 6 of the
`physical-ai-research` skill), with the cut line settled by pairwise comparisons asked in both orders. The rubric is a
research-guided starting point. It should be calibrated against what the channel's viewers actually watch.

## What every research run saves

`<ep>/judge/` holds the cards, each judge's file and verdicts, and `panel.json`. `<ep>/scores.json` (step 9 of the
skill) has one row per judged story, picked or not:

```json
[{"story": "Gemini Robotics 2.5 at Mercedes", "type": "deployment", "picked": true, "rank": 3, "overall": 7.67,
  "spread": 1, "by_judge": {"A": 8, "B": 7, "C": 8},
  "criteria": {"significance": 7.3, "viewer_pull": 7.7, "evidence": 8.0, "showability": 8.7, "momentum": 6.3},
  "pairwise_wins": 2.5, "segment": "story-3"}]
```

`segment` matches the episode's chapter, so it can be joined with YouTube Analytics later.

## How to calibrate (after 4–6 published episodes)

1. **Get the numbers.** For each episode, export average percentage viewed per chapter (YouTube Studio → Analytics →
   the video → Audience retention; or the YouTube Analytics API with `elapsedVideoTimeRatio`, which needs the channel
   owner's OAuth). Also note the comments and likes on each Short, since each Short promotes one story.
2. **Join** retention per segment with that story's `scores.json` row.
3. **Look for consistent misses**, not correlations (~20 stories is too few for a regression):
   - stories the panel ranked high that lost viewers
   - reserves promoted late that held them
   - criteria that track retention, or don't (high momentum but weak retention: the judges are chasing buzz)
4. **Change the rubric's wording**, one criterion at a time: sharpen its anchors, or add what the misses had in
   common ("a funding round without a product to show holds viewers poorly"). Record it here with the date and the
   reason, then re-run the panel on a past episode's `cards.json` and check its picks still look right.
5. **Keep it a panel.** If one model's verdicts keep diverging from what performed, swap that model, don't drop to two.

## Change log

| Date | Change | Why |
|---|---|---|
| 2026-11-14 | Fixed formula S = 100·D·(0.30·NV + 0.25·A + 0.20·C + 0.25·Q) | Research-guided defaults |
| 2026-11-15 | Replaced by an LLM judge panel (3 models, shuffled cards and criteria, pairwise cut line) | Judgement calls need reading, not arithmetic; panel and both-order pairs cancel single-judge biases |
