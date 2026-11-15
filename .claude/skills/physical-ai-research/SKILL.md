---
name: physical-ai-research
description: Find, triage, verify and rank the last 7 days of physical-AI stories (robots, humanoids, embodied and world models, autonomy) for the weekly Pixels2Actions episode (project physical-ai-daily). Checks API credits, syncs the P2A digest's six topic profiles, harvests arXiv exhaustively (with alphaXiv / Hugging Face / code / X signals and a paper prior), X (twitterapi.io) and NewsData.io with tools/discover.py, runs independent web-search discovery in parallel, has subagents read every item in full, gathers what people are saying (X, Reddit, HN, YouTube), ranks stories with an LLM judge panel, verifies each against its primary source, and writes a shortlist to <ep>/research.md. Use when the user asks what happened this week in physical AI, to find stories, to research or shortlist an episode, or as step 1 of making an episode.
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
| `chatter.py` | What X, HN and YouTube (views, descriptions, top comments, transcripts) say about one story |
| `judge_panel.py` | Prepares the LLM judge panel's files, aggregates their verdicts, settles the cut line pairwise |
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

**Web discovery runs at the same time, independently of the harvest.** X, arXiv and the news API miss whole kinds of
events: newsrooms that don't post on X, filings and regulator pages, lab blogs, news outside NewsData's index and its
36 h reach, other languages. So while discover.py runs (~5 min), spawn four web-search subagents in one message, one
per beat, each told to search as much as its beat needs until new searches stop turning up new events:

| Beat | Covers |
|---|---|
| companies | Humanoid and general-purpose robot makers; industrial, warehouse, home and surgical robots; launches, deployments, deals, funding, M&A, exits, big labs entering or leaving robotics |
| autonomy_policy | Robotaxis and AVs, trucking, delivery robots and drones; safety, recalls, investigations, permits, regulation and government programs in any country |
| models_research | Robot foundation models, VLAs, world models, sim-to-real, datasets, benchmarks, open-source tools, lab blogs, conference awards, genuine expert essays |
| global | China, Japan, Korea, India, Taiwan, Europe and the Gulf, searched in local languages too (人形机器人, 具身智能, ロボット, 로봇) |

Give each the window, its beat, the rules (primary-source date inside the window, and say when the original was
earlier; epistemic status kept; no invented facts) and this contract, written to `<ep>/web/finds_<beat>.json`:

```
[{"event": "...", "date": "YYYY-MM-DD (primary source)", "what": "<stated facts, numbers verbatim>",
  "primary_url": "...", "other_urls": [], "confidence": "primary-confirmed | press-only | snippet-only",
  "why_it_matters": "<one line>"}]
```

Web finds skip triage (they're already read and dated) and go straight into clustering in step 4.

**Search budget:** WebSearch is capped at about 200 calls per session, shared by every agent. Four beat agents left
uncapped used it all (2026-11-15), leaving none for step 3. Give each beat agent **35 searches** (it can fetch as many
pages as it likes; fetches aren't capped) and keep the rest for step 3. Verification agents use WebFetch on known URLs.

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

### 3. Follow leads with web search

The web-discovery agents searched by beat; now search from what triage found. Chase the `leads` the triage
subagents listed, find the primary source behind every keep, and fill any gap you notice (a digest prompt's system
that didn't show up, a story only one beat agent half-found). Search as much as the week needs, until new searches
stop turning up new stories. Usual blind spots, as reminders and not a checklist: primary sources behind leads, company
newsrooms, funding and filings, deployments and safety or regulator actions, China's humanoid ecosystem, and anything
the digest prompts name that didn't show up. Add finds with their URL and the date shown on the page.

### 4. Cluster

The same event seen as a post, a press release, a paper and an article is one story. Merge by the subagents' `event`
names, keep every URL, and count independent sources (different outlets or accounts, not the company re-posting
itself); the count goes on the story's card in step 6.

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
  **The search returns off-topic videos** (an older version's launch video, a namesake: "Praxis 1" is also a
  teacher exam). Count views only from videos actually about this week's story when putting them on a card.
- **Reddit and forums:** web search (`site:reddit.com <name>`, r/robotics, r/MachineLearning, r/singularity) and
  respected blogs or forums. Reddit blocks unauthenticated API reads.

Record per story: the consensus, the strongest pushback, expert takes worth quoting (attributed), and existing
coverage. Pushback feeds the Reality-check beat. Hype doesn't count as evidence (Q stays as it is).

### 6. Rank with an LLM judge panel

Ranking is a judgement call (is a factory pilot bigger than a paper everyone is sharing?), and LLM judges now agree
with human experts about as well as experts agree with each other on such calls. But a single judge has known biases:
it favours whatever it reads first or last, longer and more enthusiastic write-ups, and the rubric's criteria order
can flip its top pick. LLMs also tend to overvalue technically novel but vaguely described work and undervalue
concrete, high-impact news (Northwestern's newsroom study). So the ranking is a **panel**, built to cancel those biases:

1. **Write one card per cluster** to `<ep>/judge/cards.json` (format in `judge_panel.py`), for every cluster that
   survived triage and isn't gated out (below): typically 12–20. A card holds only verified facts, in the same shape
   for every story, so length doesn't sway a judge:
   - **What happened:** 2–3 sentences, numbers verbatim, epistemic status kept ("pilot", "plans to", "preprint").
   - **Evidence:** the primary source, its date, what it shows (numbers, demo, footage), whether results are
     vendor-reported, independent corroboration.
   - **Signals:** best paper `prior`, best X post `eng_pct` and its account, number of independent sources, YouTube
     coverage (videos, total views). Signals are evidence for the judges, not a score.
   - **What people say:** consensus and the strongest pushback from step 5, attributed.
   - **Visuals:** what footage, figures or demos exist; for papers, the best analogy you have.
   - **Freshness:** days since the primary source's date, and anything older about it.
2. **Prepare and run the judges.** `judge_panel.py prepare <ep>/judge/cards.json <ep>/judge` writes `judge_A.md`,
   `judge_B.md` and `judge_C.md`: each with the cards in its own shuffled order and the rubric's criteria in their own
   order. Spawn three fresh subagents in one message, **each on a different model** (`opus`, `sonnet`, `fable`: different
   models have different biases), each told only to read its file and write the JSON it asks for. They score each
   story 1–10 on significance, viewer pull, evidence, showability and momentum, reasoning first, then give an overall
   score and their top 5. Don't show them your own ranking.
3. **Aggregate:** `judge_panel.py aggregate <ep>/judge` ranks by mean overall (Borda points from the top-5 lists break
   ties) and flags stories where judges differ by 3+. Read the flagged stories' reasoning: a disagreement often means
   a card is missing a fact, so fix the card and re-run rather than average it away.
4. **Settle the cut line pairwise:** `judge_panel.py pairs <ep>/judge` writes every pair among ranks 3–7 in both
   orders. One fresh subagent (a model not used in step 2's majority is fine; use `opus`) answers them;
   `judge_panel.py settle <ep>/judge` re-orders those ranks by wins that held in both orders. Pairwise comparison
   tracks human preference better than scores do, which is why it decides the places that matter.
5. **Apply the editorial rules yourself, after the panel** (they're facts about the lineup, not judgement):
   at most 2 stories per company; at least one research story when one is ranked within 3 places of the 5th pick;
   variety (not five humanoid funding rounds); at least one story the viewer couldn't have read on a mainstream front
   page. Moving a story for a rule: say which rule in `research.md`.

**Gates, before any card is written:** a story with no primary source or a credible tier-1 report with named sources
can't be shortlisted (mention it as unconfirmed at most). Buzz never rescues weak evidence. Anything older than the
window with nothing new this week is out.

**Show the panel** in `research.md`: each shortlisted story's mean overall, per-criterion means, each judge's overall
and a one-line digest of their reasoning, so the user can argue with it.

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

The prior is a **reading order and a signal on the card**, not the decision. A paper with prior 0.2 can still be the week's best
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
Sources: arXiv ✓ (N papers), X ✓ (N), News ✓/degraded (N), web discovery (N events from 4 beats), topics synced ✓/stale,
credits: <credits.py summary>, triage batches: N, judges: <models>

## Shortlist
| # | Story | Type | Panel (A/B/C) | Sig | Pull | Evid | Show | Mom | Primary source (date) | Key fact, verbatim | Visual | Caveat |
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

Also write `<ep>/scores.json`: one row per judged cluster (picked or not), from `panel.json` plus `picked` and
`segment`, in the format in `physical-ai-daily/docs/research-tuning.md`. That's what calibrates the judges later.

Show the shortlist to the user before scripting, unless they said to run unattended. Then hand off to the
`physical-ai-daily` skill (step 4, `episode.json`).

## Keep improving

- **Calibrate the judges** after 4–6 episodes against what performed (YouTube retention per story segment),
  following `physical-ai-daily/docs/research-tuning.md`: change the rubric's wording, not the arithmetic.
- **Add good finds to `sources.json`:** an account that broke a story goes in `x_accounts`, a lab that keeps showing
  up in strong papers goes in `orgs_lab`, a query that found something the harvest missed goes in `news_queries`.
- **Note failures** (sources down, credits out, false leads) in the run's `research.md`.
