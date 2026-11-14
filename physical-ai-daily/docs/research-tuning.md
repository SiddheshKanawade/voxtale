# Tuning the research scores

The story score (`S = 100 × D × (0.30·NV + 0.25·A + 0.20·C + 0.25·Q)`) and the paper prior (`discover.py`
`paper_priors`) start from research-guided guesses: correlational studies for papers, news-values research and
editorial judgement for stories (sources in the `physical-ai-research` skill). They should be tuned against what the
channel's viewers actually watch.

## What every research run saves

`<ep>/scores.json`, written in step 9 of the research skill: one row per scored cluster, shortlisted or not.

```json
[{"story": "Gemini Robotics 2.5", "type": "release", "picked": true, "S": 71.2,
  "NV": 0.73, "NV_parts": {"magnitude": 3, "surprise": 1, "power_elite": 3, "audio_visuals": 3, "relevance": 2},
  "A": 0.81, "C": 0.75, "Q": 1.0, "D": 0.88, "prior": null, "segment": "story-1"}]
```

`segment` matches the episode's chapter, so it can be joined with YouTube Analytics later.

## How to tune (after 4–6 published episodes)

1. **Get the numbers.** For each episode, export average percentage viewed per chapter (YouTube Studio → Analytics →
   the video → Audience retention; or the YouTube Analytics API with `elapsedVideoTimeRatio`, which needs the channel
   owner's OAuth). Also note the comments and likes on each Short, since each Short promotes one story.
2. **Join** retention per segment with that story's `scores.json` row.
3. **Compare.** With ~20 picked stories a regression is noise, so look for consistent misses instead:
   - stories that scored high and lost viewers
   - stories that scored low (reserves promoted at the last minute) and held them
   Then ask which term misled. Examples: high A from a viral post but weak retention means A is overweighted; strong
   audio-visuals in NV and strong retention means visuals deserve more weight.
4. **Change one weight at a time**, by at most 0.05. Record it here with the date and the reason. Re-score the last
   episodes' `scores.json` with the new weights and check that the picks still look right.

## Change log

| Date | Change | Why |
|---|---|---|
| 2026-11-14 | Initial weights | Research-guided defaults |
