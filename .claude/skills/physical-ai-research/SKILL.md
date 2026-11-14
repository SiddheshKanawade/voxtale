---
name: physical-ai-research
description: Find, triage, verify and rank the day's physical-AI stories (robots, humanoids, embodied and world models, autonomy) for a Pixels2Actions episode (project physical-ai-daily). Harvests arXiv, X (twitterapi.io), NewsData.io and Hacker News with tools/discover.py, widens with as many web searches as the day needs, judges relevance and quality itself (no embeddings), verifies every survivor against its primary source, and writes a scored shortlist to <ep>/research.md. Use when the user asks what happened today in physical AI, to find stories, to research or shortlist an episode, or as step 1 of making an episode.
---

# Story research (Pixels2Actions)

Output: `episodes/<date>[-slug]/research.md`, a shortlist of 3–5 verified stories (plus reserves) that the
`physical-ai-daily` skill scripts from. Harvester: `physical-ai-daily/tools/discover.py`. Accounts, categories and
queries: `physical-ai-daily/sources.json`.

**Lineage.** The harvest methods come from prazo5, the engine behind the user's Pixels2Actions newsletter: batched
`from:` X queries on twitterapi.io, NewsData.io `/latest` with key pooling, HN Algolia with a points floor, and arXiv by
day. prazo5 then ranks by OpenAI embedding similarity to an anchor text. **This skill doesn't.** You read the
candidates and judge them against the editorial anchor below, because a model reading a story can tell "a humanoid
shipped to a factory" from "a paper that mentions humanoids in passing", and cosine similarity can't. The faithfulness
rules in step 5 are adapted from prazo5's summary prompts, which went through several adversarial rounds on real
digest data.

## Editorial anchor (what counts)

The viewer is a product manager with no robotics background who wants to know **what changed in machines that act in
the physical world, and why it matters**.

- **In:** robots of any shape (humanoids, arms, quadrupeds, drones, warehouse and surgical robots); robot foundation
  models, VLAs, and world models used for control or simulation; sim-to-real; autonomous vehicles and robotaxis;
  dexterous hands, actuators and sensors when they unlock a capability; real deployments, pilots, and safety or
  regulatory events; funding and M&A of $50M or more, or strategically notable; big labs entering or leaving robotics.
- **Out, unless it touches robots:** chatbots and LLM releases, generic AI funding, crypto, consumer gadgets, market
  research "reports", and listicles.
- **Borderline: keep it only if you can say in one sentence what changes for a physical system.** Examples: a video
  model sold as a "world model", or a chip launch aimed at robotics.

## Workflow

### 0. Window and keys

- **Window:** since the last episode. The default is the last 36 h for X, news and HN, and 72 h for arXiv, because
  announcements lag and skip weekends. Check `episodes/` for what the last episode already covered, and don't repeat it
  unless there's real news on it.
- **Keys** go in `physical-ai-daily/.env` (git-ignored; never print or commit them):
  - `TWITTERAPI_IO_KEY`: X search. Billed per tweet returned.
  - `NEWSDATAIO`: news. One key, or a JSON array of keys to pool free-tier quota.
- **If a key is missing:** `discover.py` says so on stderr and falls back (X) or skips (news). Tell the user which
  source ran degraded, and make up for it with web search in step 2.

### 1. Harvest

```
python3 -I physical-ai-daily/tools/discover.py --hours 36 --paper-hours 72 > <ep>/candidates.json
```

- **Read the per-source counts on stderr.** A source returning 0 usually means a failure, not a quiet day: check it.
- **Read every candidate's title or first line.** There are often 100–300, so print them compactly, for example
  `jq -r '.[] | [.type, (.likes // .points // ""), (.mentions|length), (.title // .text)[:140]] | @tsv'`.
- **Read the abstracts and full text only for items that pass a first relevance pass.**

### 2. Widen with web search (no fixed count: as many as the day needs)

The harvest misses things: company newsrooms that don't tweet, news outside NewsData's index, and anything in
another language. Search until new searches stop turning up new stories.

Cover at least these angles:

- **Follow-ups on the strongest leads.** Find the primary source behind every promising tweet or HN post, plus any
  coverage that adds facts.
- **Newsrooms and blogs** of the big players: DeepMind, NVIDIA, Tesla, Figure, Boston Dynamics, Agility, Apptronik,
  1X, Unitree, Physical Intelligence, Skild, Waymo, Amazon Robotics, OpenAI, Meta FAIR.
- **Money and deals:** funding, M&A, and IPO filings in robotics over the window.
- **Deployments and safety:** pilots, fleet sizes, recalls, regulator actions (NHTSA, the EU AI Act, China MIIT).
- **China's humanoid ecosystem:** Unitree, UBTech, AgiBot, Galbot, Xpeng. It's under-covered in English feeds.
- **Counter-signal:** for each top candidate, search for criticism, corrections, or "staged demo" claims before
  ranking it.

Add new finds to your working list with their URL and the date you saw on the page.

### 3. Triage (adapted from prazo5's gates; the first gate that fires wins)

1. **Aggregator:** a roundup or list naming 3 or more unrelated items with no central subject. Drop it, but mine it
   for leads.
2. **Promo:** a market-research report, sponsored post, or "Top 10 robots" list. Drop it.
3. **Empty:** nothing specific can be stated without inventing it, like a bare link or "big news soon". Drop it.
4. **Off-anchor:** fails the editorial anchor above. Drop it.
5. **Otherwise keep it.** Opinions and predictions from credible people stay, attributed ("Jim Fan argues…").

Then **cluster**: the same event seen as a tweet, a press release, a paper and an HN thread is one story. Merge them,
and keep every URL. Its `mentions` count and the number of independent sources are its buzz.

### 4. Score and rank

Score each cluster 0–2 on each criterion:

| Criterion | 2 | 0 |
|---|---|---|
| **Impact for a PM** | Changes what is possible, buyable or regulated | Incremental and academic |
| **Novelty** | First, new record, or new entrant or exit | A rerun of last month |
| **Evidence** | Primary source with concrete numbers | Secondary only, or vague |
| **Visual** | Video, demo stills or clear figures available | Text only |
| **Buzz** | Several independent sources, high engagement | One post, no pickup |
| **Freshness** | Inside the window on the primary source | Older, resurfacing |

- **Ranking:** order by total score. Break ties with the channel default: deployment > release > funding > paper >
  opinion.
- **Variety:** aim for a mix, for example not five humanoid funding rounds. At least one story should be something
  the viewer could not have read on a mainstream front page.
- **Buzz never rescues weak evidence.** A viral claim with no primary source stays out. At most, mention it as
  unconfirmed.

### 5. Verify (every story on the shortlist)

Apply the verification rules in the `physical-ai-daily` skill (primary source, date on the primary source, numbers
verbatim on a capturable page, vendor-reported results labelled as such). On top of those, these are prazo5's
faithfulness rules:

- **Use only stated facts.** Never infer an outcome, cause or reaction the source doesn't state. A filing is not a
  ruling.
- **Preserve epistemic status.** "Plans to", "in talks", "reportedly", "aims to", pilot, preprint: an announced plan is
  never a completed action.
- **Scope and attribution:**
  - Keep each number attached to the exact system and subset the source gives it ("78% on the IDB tasks", not "78%").
  - The subject of the result must be the system that achieved it, not the benchmark it was measured on.
- **Qualifiers:** use "first", "state-of-the-art" or "robust" only if the source says it, about the same claim.
- **Truncated text:**
  - A news API snippet that cuts off mid-sentence doesn't exist past its last full sentence.
  - Never use a number whose unit was cut off.
  - Open the article itself.
- **Copy names, versions and numbers character for character.**

### 6. Write `research.md`

```
# Research <date> (window: <from> → <to>; sources: arXiv ✓, X ✓/degraded, News ✓/skipped, HN ✓, web searches: N)

## Shortlist
| # | Story | Type | Score | Primary source (date) | Key fact, verbatim | Visual | Caveat |
...one row per story, best first. Then for each: a 24–30 word summary (named subject first, the single most important
number), every URL in the cluster, and the exact sentence to highlight with capture.mjs --find.

## Reserves
2–3 stories that scored next, with one line on why they didn't make it.

## Dropped (worth knowing)
Notable items cut at triage or verification, each with the reason ("no primary source", "announced in September").
```

Show the shortlist to the user before scripting, unless they said to run unattended. Then hand off to the
`physical-ai-daily` skill (step 4, `episode.json`).

## Keep improving

- **Add good finds to `sources.json`:** a new account that broke a story goes in `x_accounts`. A query that found
  something the harvest missed goes in `news_queries` or `x_keywords`.
- **Note failures** (sources down, keys out of quota, false leads) in the run's `research.md`.
