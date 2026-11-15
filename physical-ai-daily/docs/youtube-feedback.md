# YouTube feedback: what we collect and what it tells us

The research step picks five stories a week with an LLM judge panel. The only test of whether it picks well is what
viewers do. This file says which numbers to collect for each episode, how to read them fairly, and which decisions
they feed. It replaces the blind ranking by the user (2026-11-16): the user's taste is a proxy, and viewers are the
target.

## The questions

1. **Do the stories the panel ranks higher hold viewers better?** (Is the ranking any good?)
2. **Which criteria predict that:** significance, learning value, evidence, viewer pull, showability, momentum?
3. **Do "learning" stories hold viewers better than "news" stories?** (Is the 2-learning-stories rule right?)
4. **What, outside the story choice, loses viewers:** the opening, segment length, script, visuals?

## What to collect

### Per episode (long video), from YouTube Studio → Analytics → the video

| Metric | Where | Why |
|---|---|---|
| Impressions, click-through rate | Reach | Packaging (title, thumbnail), not story choice. Kept so it isn't confused with it |
| Views, average view duration, average percentage viewed | Engagement | Overall health |
| **Audience retention curve**, exported per second or per chapter | Engagement → Audience retention | The main input: where viewers leave |
| Key moments: spikes (rewatched) and dips | Audience retention | Which beats land |
| Likes, comments, shares, subscribers gained | Engagement | Interest beyond watching |
| Traffic sources (browse, suggested, search, external) | Reach | Who arrived: suggested viewers leave faster, so compare like with like |
| New vs returning viewers | Audience | Returning viewers are the channel's core |
| Test & Compare result (thumbnails) | Reach | For the thumbnail skill, not the research step |

### Per story segment (chapter)

From the retention curve, for each chapter:

- **Retention at the start and the end** of the segment (% of starting viewers still watching)
- **Drop rate**: % of viewers lost per minute *within* the segment. This is the number to compare across stories,
  because it doesn't punish a story for coming later in the video
- **Relative retention** vs YouTube's typical curve for videos of similar length (Studio shows it)
- Rewatch spikes inside the segment

### Per Short (one story each)

| Metric | Why |
|---|---|
| Views, **"viewed vs swiped away"** % | The cleanest per-story signal: no position effect, one story only |
| Average percentage viewed, loops | Did the story hold |
| Likes, comments, shares, subscribers gained | Interest |

### Comments (long video and Shorts)

An agent reads all comments and tags each one, quoting it: **which story** it's about, and whether it's a
**question** (curiosity, the learning signal we want), a **correction** (an accuracy problem, top priority), **praise**,
or **criticism**. No summaries without the quotes, per the sourcing rule.

## Where it goes

`episodes/<ep>/youtube.json`, one file per episode, filled in about 7 days after publishing (and again at 28 days):

```json
{"video_id": "...", "published": "2026-11-17", "collected": "2026-11-24", "days_live": 7,
 "video": {"impressions": 0, "ctr": 0.0, "views": 0, "avg_view_duration_s": 0, "avg_pct_viewed": 0.0,
           "likes": 0, "comments": 0, "shares": 0, "subs_gained": 0, "traffic": {"browse": 0.0, "suggested": 0.0}},
 "segments": [{"segment": "story-1", "start_s": 0, "end_s": 0, "retention_start": 0.0, "retention_end": 0.0,
               "drop_per_min": 0.0, "relative_retention": "above|typical|below"}],
 "shorts": [{"segment": "story-1", "video_id": "...", "views": 0, "viewed_pct": 0.0, "avg_pct_viewed": 0.0,
             "likes": 0, "comments": 0, "shares": 0}],
 "comments": [{"segment": "story-2", "kind": "question|correction|praise|criticism", "text": "<verbatim>",
               "likes": 0}]}
```

`segment` matches `scores.json`, so every story's judge scores line up with how it performed.

**How to get the numbers:** export from YouTube Studio by hand (Analytics → Advanced mode → export CSV), or later
through the YouTube Analytics API (needs the channel owner's OAuth; per-chapter retention via
`elapsedVideoTimeRatio`). Corrections in comments are acted on at once; everything else waits for enough data.

## How to read it fairly

A segment's numbers depend on more than the story. Control for what we can:

- **Position:** the first segment always keeps more viewers. Compare drop rate per minute, not absolute retention,
  and compare each segment with the same slot in other episodes.
- **Length:** a longer segment loses more viewers in total. Drop rate per minute handles it.
- **Execution:** a good story can be badly scripted. When a segment drops sharply, check the script and visuals at
  that timestamp before blaming the story choice.
- **The week:** some weeks are bigger. Compare stories within an episode (ranks), as well as across episodes.
- **Shorts as the tie-breaker:** when the long video and the Short disagree about a story, the Short is the cleaner
  read on the story itself.

## When to act

- **Corrections** in comments: immediately (fix the source handling that let it through, note it in the skill).
- **Everything else: after 6 episodes** (~30 segments, ~30 Shorts). Before that, patterns are mostly noise.
- Then, for questions 1–3: rank correlation (Spearman) between the panel's overall score and segment drop rate, and
  Short viewed-%; the same for each criterion; learning vs news on the same measures.
- **Change one thing at a time**, in the rubric's wording (not by adding formulas): for example, if momentum keeps
  predicting nothing, tighten its anchors; if learning stories hold viewers better, raise the minimum from 2. Log the
  change, its date and its evidence in `docs/research-tuning.md`, and re-judge a past episode's cards to see what
  would have changed.

## What we don't do

- Chase one viral Short: one data point is not a pattern.
- Change the editorial rules (sourcing, verification, learning over knowing) because a story performed badly. Those
  rules are the channel's identity; the feedback tunes the ranking within them.
