---
name: physical-ai-research
description: Find, triage, verify and rank the last 7 days of physical-AI stories (robots, humanoids, embodied and world models, autonomy) for the weekly Pixels2Actions episode (project physical-ai-daily). Checks API credits, syncs the P2A digest's six topic profiles, harvests arXiv exhaustively (with alphaXiv / Hugging Face / code / X signals and a paper prior), X (twitterapi.io) and NewsData.io with tools/discover.py, has subagents read every item in full, widens with web search, gathers what people are saying (X, Reddit, HN, YouTube), scores stories with a fixed formula, verifies each against its primary source, and writes a shortlist to <ep>/research.md. Use when the user asks what happened this week in physical AI, to find stories, to research or shortlist an episode, or as step 1 of making an episode.
---

# Story research (Pixels2Actions)

**Cadence:** one long-form video a week covering the week's top 3–5 stories, plus Shorts cut from it to promote it.
The P2A digest (prazo5) is daily; this channel is weekly.

**Output:** `episodes/<date>[-slug]/research.md`, a shortlist of 3–5 verified stories plus reserves. The
`physical-ai-daily` skill scripts from it.

**Tools** (all in `physical-ai-daily/tools/`, run with `python3 -I`):

| Tool | Does |
|---|---|
| `credits.py` | Key and remaining-credit check for every paid source |
| `discover.py` | Syncs the digest topics, harvests arXiv + X + news, adds signals and priors, dedupes |
| `sync_p2a_topics.py` | The topic sync on its own (discover.py calls it) |
| `triage_batches.py` | Splits the harvest into reading batches for triage subagents |
| `chatter.py` | What X, HN and YouTube (views, descriptions, top comments) say about one story |
| `news_daily.sh`, `install_news_schedule.sh` | Twice-daily saved news pull (launchd), folded into the weekly harvest |

Config: `physical-ai-daily/sources.json` (our accounts, arXiv terms, news queries, org tiers). Digest topics:
`physical-ai-daily/p2a_topics.json` (git-ignored; the repo is public and the prompts are the digest's edge).

**Lineage.** Harvest methods come from prazo5, the engine behind the Pixels2Actions digest. prazo5 ranks by OpenAI
embedding similarity; **this skill doesn't**. Models read every candidate in full and judge it, because a reader can
tell "a humanoid shipped to a factory" from "a paper that mentions humanoids in passing" and cosine similarity can't.
The faithfulness rules in step 8 are adapted from prazo5's summary prompts.

## Audience and editorial anchor

The viewer has **a slight technical background but little knowledge of the field**. Hard concepts are welcome as long
as they can be explained so that viewer follows them. The question every story answers: **what changed in machines that act in
the physical world, and why it matters**.

- **In:** robots of any shape (humanoids, arms, quadrupeds, drones, warehouse and surgical robots); robot foundation
  models, VLAs and world models used for control or simulation; sim-to-real; autonomous vehicles and robotaxis;
  dexterous hands, actuators and sensors that unlock a capability; real deployments, pilots, safety and regulatory
  events; funding and M&A of $50M+ or strategically notable; big labs entering or leaving robotics.
- **Out, unless it touches robots:** chatbots and LLM releases, generic AI funding, crypto, consumer gadgets,
  market-research "reports", listicles.
- **Borderline:** keep it only if you can say in one sentence what changes for a physical system.

**The digest's six topics are the depth layer, and they're required.** `p2a_topics.json` holds `vla-generalization`,
`latent-actions-world-models`, `video-gen-policies`, `human-video-demos`, `teleop-data-collection` and
`video-reward-eval`: each a hand-tuned prompt naming the systems, papers and benchmarks that matter, ranked keywords
and ~20 X accounts. They steer the harvest (accounts, X keywords, arXiv terms) and triage: a new model in a line a
prompt names (pi0.7, GR00T, V-JEPA, UMI) is news even when the post is quiet. Every item's `topics` tags are a lexical
hint, not a score; untagged items can still be the week's biggest story.

## Workflow

### 0. Window and credits

- **Window: the 7 days ending now** (`--days 7`), not "since the last episode". Still check `episodes/` for what was
  already covered, and don't repeat a story unless there's real news on it.
- **Run `credits.py` first.** It prints OK / LOW / OUT / MISSING per source (twitterapi.io balance, each pooled NewsData
  key's credits left today, ElevenLabs characters, Render access) and exits 1 if a harvest source is out. Tell the user
  which sources will run degraded before harvesting, and make up for them with web search. Never print a key.
- **News reaches back only ~36 h** on the NewsData free plan, so a launchd job on this Mac
  (`tools/install_news_schedule.sh`, already installed; `--remove` uninstalls) pulls news at 08:00 and 20:00 into
  `physical-ai-daily/news/` (git-ignored, kept 21 days, log in `news/pull.log`). The weekly harvest folds those saved
  items in automatically and says on stderr how many pulls it found. Gaps (days the Mac was off) need web search; say
  so in `research.md`.

### 1. Harvest

```
python3 -I physical-ai-daily/tools/credits.py
python3 -I physical-ai-daily/tools/discover.py --days 7 > <ep>/candidates-<date>.json   # syncs topics first
python3 -I physical-ai-daily/tools/discover.py --merge <ep>/candidates-*.json > <ep>/candidates.json
```

- **Topic sync is part of every harvest.** If it fails, discover.py says so and uses the last copy; with no copy it
  stops. Report a failed sync.
- **arXiv is exhaustive:** all of cs.RO, plus cs.CV / cs.LG / cs.AI / eess.SY papers whose title or abstract contains
  any of our terms or any of the ~90 digest keywords. Papers then get `signals` (alphaXiv organisations, votes, 7-day
  visits, researcher h-index, with OpenAlex as the fallback for organisations; Hugging Face Daily Papers upvotes, code repo and stars, demo video, project page, venue
  from the authors' comment; X accounts that posted it) and a `prior` (see Ranking). About 350 papers a week; the paper
  step alone takes ~4 min.
- **X:** our accounts plus ~120 digest accounts in `from:` batches of 10, plus the digest keywords with a likes floor.
  Each post gets `engagement` and its percentile `eng_pct`.
- **Hacker News and Reddit are not harvested.** Most of what they surface is off-topic for this channel; they're used
  per story in step 5 instead. (`--only hn` still works if a week needs it.)
- **Read the per-source counts on stderr.** A source returning 0 usually means a failure, not a quiet week.

### 2. Read everything (triage subagents)

Titles and first lines lose context, so every item is read in full, by subagents in parallel:

```
python3 -I physical-ai-daily/tools/triage_batches.py <ep>/candidates.json <ep>/triage
```

That writes ~8–10 batch files, one source each (papers with full abstracts and signals, posts with links and
engagement, news with descriptions), ordered by prior. **Spawn one subagent per batch, all in one message.** Give each:
the batch path, `p2a_topics.json` (to read the six prompts first), the Audience and editorial anchor section above, the
triage gates below, and this output contract, written to `<ep>/triage/verdicts_NN.json`:

```
[{"url": "...", "verdict": "keep" | "maybe" | "drop", "gate": "aggregator|promo|empty|off-anchor|null",
  "event": "<short name of the underlying event, for clustering>", "why": "<one line>",
  "leads": ["<url or name of anything the item points at worth chasing>"]}]
```

Tell them: judge relevance against the anchor and topic prompts, never by keyword match; don't drop a paper for being
hard (that is step 6's call); a post that only links somewhere is judged by what it links to, so list the link in
`leads`. Then **you** read every keep and maybe yourself, in full: open the abstract page, article or thread.

**Triage gates** (adapted from prazo5; the first gate that fires wins):
1. **Aggregator:** a roundup naming 3+ unrelated items with no central subject. Drop it, but mine it for leads.
2. **Promo:** market-research report, sponsored post, "Top 10 robots" list. Drop.
3. **Empty:** nothing specific can be stated without inventing it ("big news soon", a bare link). Drop, after
   following the link.
4. **Off-anchor:** fails the editorial anchor. Drop.
5. **Otherwise keep.** Opinions and predictions stay when the source is genuine and knows what it's talking about:
   a researcher, a lab, a company, an engineer with a track record. Attribute them ("DeepMind's team argues…").

### 3. Widen with web search

The harvest misses things: newsrooms that don't post on X, news outside NewsData's index and its 36 h reach, other
languages, filings. Search as much as the week needs, wherever you judge stories are hiding, until new searches stop
turning up new stories. Usual blind spots, as reminders and not a checklist: primary sources behind leads, company
newsrooms, funding and filings, deployments and safety or regulator actions, China's humanoid ecosystem, and anything
the digest prompts name that didn't show up. Add finds with their URL and the date shown on the page.

### 4. Cluster

The same event seen as a post, a press release, a paper and an article is one story. Merge by the subagents' `event`
names, keep every URL, and count independent sources (different outlets or accounts, not the company re-posting
itself) for C in step 6.

### 5. What people are saying (for the ~8–12 strongest clusters)

Before scoring, learn what's happening around each candidate:

```
python3 -I physical-ai-daily/tools/chatter.py "<story name>" --url <primary url> [--arxiv <id>] \
    [--transcripts 2 --out <ep>/chatter/<slug>] > <ep>/chatter/<slug>.json
```

- **X:** the most-engaged posts about it, including posts linking the primary source without naming it. Who's
  excited, who's sceptical, which experts weighed in.
- **Hacker News:** all-time search; the comments are where experts push back.
- **YouTube:** who has already covered it and with how many views (trend signal, and an angle to avoid repeating).
  For the top 3 videos: upload date, likes, the full description (channels often summarise the story there) and the
  top 20 comments. `--transcripts N` saves the N top videos' transcripts: captions when YouTube serves them, else a
  local whisper.cpp transcription of the audio (YouTube rate-limits captions with 429; about 1 min per video). Read
  them for how other channels explained the story and what they got wrong.
- **Reddit and forums:** web search (`site:reddit.com <name>`, r/robotics, r/MachineLearning, r/singularity) and
  respected blogs or forums. Reddit blocks unauthenticated API reads.

Record per story: the consensus, the strongest pushback, expert takes worth quoting (attributed), and existing
coverage. Pushback feeds the Reality-check beat. Hype doesn't count as evidence (Q stays as it is).

### 6. Score and rank

Every cluster gets a **story score S from 0 to 100**:

```
S = 100 × D × (0.30·NV + 0.25·A + 0.20·C + 0.25·Q)
```

| Term | What it is | How to compute |
|---|---|---|
| **NV** news value, 0–1 | Your judgement on five news values (Harcup & O'Neill), each 0–3, mean ÷ 3 | Table below |
| **A** attention, 0–1 | How much the field is paying attention | max of: the cluster's best paper `prior`; its best X post `eng_pct`; for YouTube coverage, min(1, log10(1 + total views) / 6) |
| | | A post found outside the harvest (via `chatter.py` or search) gets `discover.engagement()` and its percentile against the harvest's X posts. Script the arithmetic so it's reproducible |
| **C** corroboration, 0–1 | Independent sources covering it (Techmeme's link-count idea) | min(1, log2(1 + n) / 4): 1 source 0.25, 3 → 0.5, 7 → 0.75, 15+ → 1 |
| **Q** source quality, 0–1 | Strength of the best primary evidence | 1.0 primary source with concrete numbers or a demo; 0.7 primary but vague, or tier-1 press with named sources; 0.4 tech blog / secondary; 0.2 aggregator or unverified |
| **D** freshness | Mild decay; this is a weekly roundup | max(0.6, 1 − 0.06 × days since the primary source's date) |

**News values** (score each 0–3):

| Value | 0 | 3 |
|---|---|---|
| Magnitude | Incremental | A capability jump, deployment at scale, $500M+, or a new record |
| Surprise | Expected, a rerun of last month | Nobody saw it coming: a first, an exit, a reversal |
| Power elite | Unknown actor | A major lab, company or government |
| Audio-visuals | Text only | Real video or a striking demo we can show |
| Relevance | Only specialists care | Changes what the viewer could see, buy, use or worry about |

Then:
- **Gates:** Q < 0.4 can't be shortlisted (mention it as unconfirmed at most). Buzz never rescues weak evidence.
- **Ranking:** sort by S. Break ties with deployment > release > funding > paper > opinion.
- **Constraints:** at most 2 stories per company; at least one research story when one scores within 10 points of the
  5th pick; variety (not five humanoid funding rounds); at least one story the viewer couldn't have read on a
  mainstream front page.
- **Show the arithmetic** in `research.md` (each term per story), so the user can argue with it.

**Research stories need an explanation, but don't drop them easily.** A paper makes the video when it can be explained
to the viewer: an everyday analogy, a figure or demo that shows the idea, and a clear why-it-matters. If none comes to
mind, dig: the paper's intro and figures, the project page and video, the authors' threads, how YouTube explainers
put it (`chatter.py`). If it still won't explain, decide on what the field says: keep it when it's groundbreaking, or
when attention is high (a top prior, many X accounts with reach, YouTube coverage). YouTube runs on what's trending,
and a paper everyone is talking about is worth the extra effort to explain.

### 7. Paper ranking (how `prior` is computed, and why)

```
prior = 0.30·N(HF upvotes) + 0.20·N(X attention) + 0.10·N(alphaXiv votes + visits/20)
      + 0.15·code + 0.10·demo + 0.15·prestige                      (0–1; implemented in discover.py paper_priors)
```

- N(x) = log(1+x) / log(1+week's max). Upvotes and alphaXiv are divided by √(days old + 1) first, so Friday's papers
  aren't buried.
- **X attention** sums, over distinct accounts that posted the paper, min(log10(1 + followers), 7) + log10(1 + likes):
  who shares a paper matters more than how many times it's shared.
- **code** = 0.5 for a repo + 0.5·min(1, log10(1 + stars) / 3).
- **demo** = 1 for a video or project page.
- **prestige** = 1 for a lab in `orgs_lab` (Google DeepMind, NVIDIA, Meta, Physical Intelligence, …), 0.5 for
  `orgs_university` (Stanford, CMU, Tsinghua, …), else 0. It only counts a third when attention + code < 0.1.

Why: brand-new papers have no citations, and the best early predictors are attention signals. Tweets predicted later
citations in Eysenbach 2011 ([JMIR](https://pmc.ncbi.nlm.nih.gov/articles/PMC3278109/)). Papers shared by big AI
accounts got 2–3× the citations of matched controls ([arXiv 2401.13782](https://arxiv.org/abs/2401.13782)). Early
access patterns forecast citations in 2.3M arXiv papers ([arXiv 2511.03877](https://arxiv.org/abs/2511.03877)). Code
and GitHub engagement add precision ([GitScholar](https://arxiv.org/abs/2609.26361),
[NeurIPS code study](https://arxiv.org/abs/2308.10008)). Prestige is real but biased: reviewers favour famous authors
and top companies (odds ratios 1.6–2.1, [WSDM 2017](https://arxiv.org/abs/1702.00502)), and much of industry's
citation gap is selection ([NBER](https://conference.nber.org/conf_papers/f205421/f205421.pdf)). So prestige is capped,
and a big-lab paper with hype but no code or benchmark gets its claims checked harder. Semantic Scholar isn't used: for
0–7-day papers its author profiles are fragmented and affiliations empty. alphaXiv's metadata endpoint is undocumented
and may break; discover.py then logs fewer "on alphaXiv" papers, and the prior leans on the rest.

The prior is a **reading order and a term in A**, not the decision. A paper with prior 0.2 can still be the week's best
story.

### 8. Verify (every story on the shortlist)

Apply the verification rules in the `physical-ai-daily` skill (primary source, date on the primary source, numbers
verbatim on a capturable page, vendor-reported results labelled as such). On top of those, prazo5's faithfulness rules:

- **Use only stated facts.** Never infer an outcome, cause or reaction the source doesn't state. A filing is not a ruling.
- **Preserve epistemic status.** "Plans to", "in talks", "reportedly", "aims to", pilot, preprint: an announced plan is
  never a completed action.
- **Scope and attribution:** keep each number attached to the exact system and subset ("78% on the IDB tasks", not
  "78%"); the subject of a result is the system that achieved it, not the benchmark.
- **Qualifiers:** "first", "state-of-the-art" or "robust" only if the source says it, about the same claim.
- **Copy names, versions and numbers character for character.**
- **Snippets:** news-API descriptions and search snippets often cut off mid-sentence. List useful ones in
  `research.md` under Snippets, marked *unverified*: knowing something was reported can shape the video. But a snippet
  is never a source for a fact or number in the narration until the full article confirms it, and a number whose unit
  or context was cut off is never used.

### 9. Write `research.md`

```
# Research <date> (window: <from> → <to>)
Sources: arXiv ✓ (N papers), X ✓ (N), News ✓/degraded (N), topics synced ✓/stale, credits: <credits.py summary>,
web searches: N, triage batches: N

## Shortlist
| # | Story | Type | S | NV | A | C | Q | D | Primary source (date) | Key fact, verbatim | Visual | Caveat |
Then for each story:
- a 24–30 word summary (named subject first, the single most important number)
- how we'll explain it: the analogy, figure or demo
- what people are saying: consensus, strongest pushback, notable takes (attributed), existing YouTube coverage
- every URL in the cluster, and the exact sentence to highlight with capture.mjs --find

## Reserves
2–3 stories that scored next, with one line on why they didn't make it.

## Snippets (unverified)
Useful snippets not yet confirmed by a full article: quote, outlet, URL, "truncated" if cut off.

## Dropped (worth knowing)
Notable items cut at triage or verification, each with the reason ("no primary source", "announced in September").
```

Also write `<ep>/scores.json`: one row per scored cluster (picked or not) with S and every term, in the format in
`physical-ai-daily/docs/research-tuning.md`. That's what tunes the weights later.

Show the shortlist to the user before scripting, unless they said to run unattended. Then hand off to the
`physical-ai-daily` skill (step 4, `episode.json`).

## Keep improving

- **Tune the weights** after 4–6 episodes against what performed (YouTube retention per story segment), following
  `physical-ai-daily/docs/research-tuning.md`. Today's weights rest on correlational studies (papers) and editorial
  judgement guided by news-values research (stories).
- **Add good finds to `sources.json`:** an account that broke a story goes in `x_accounts`, a lab that keeps showing
  up in strong papers goes in `orgs_lab`, a query that found something the harvest missed goes in `news_queries`.
- **Note failures** (sources down, credits out, false leads) in the run's `research.md`.
