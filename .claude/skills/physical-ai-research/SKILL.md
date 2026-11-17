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
| `check_quotes.py` | Confirms every verification quote appears on its page |
| `build_cards.py` | Builds the judges' story cards from stored sources only (no hand-written card text) |
| `research_md.py` | Writes `research.md` and `scores.json` from the panel, verification and cards |
| `judge_panel.py` | Prepares the LLM judge panel's files, aggregates their verdicts, settles the cut line pairwise |
| `news_daily.sh`, `install_news_schedule.sh` | Twice-daily saved news pull (launchd), folded into the weekly harvest |

**The tools are starting points, not limits.** If one doesn't fit the job, change it or write your own script; a missing tool or option is never a reason to stop. Keep the data formats between steps and the rules intact (`physical-ai-daily/docs/tools.md`).

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

## Backlog weeks (user, 2026-11-17)

To build a posting buffer, past weeks (Aug 2 – Oct 25, 2026, newest first) are researched months after the fact.
For those: harvest with `discover.py --until <week end>` (NewsData can't reach back: use the web-discovery beats for
news), and lean even harder on learning: `research_md.py --min-learning 3`. A launch or deal from three months ago
must still be worth hearing about today; prefer stories whose result still stands, and note in the card anything that
has changed since the week (a later correction, a retraction, a product that shipped) when a source says so.
Engagement counts are mature by now: compare them within the week, not with current weeks.

## Learning over knowing (user, 2026-11-16)

The channel teaches more than it reports. Every week at least **2 of the 5 stories are "learning" stories**: the core
is a new capability, method or experiment with a result, something built or tested that the viewer wouldn't expect
("a robot that runs on light"), from a lab, university or company alike. **Deployments are judged by what they
show, not by happening:** a deployment is learning when it validates an experiment or tests a hypothesis in the real
world, or reports a large measured impact on something critical (e.g. ~50% lower cost, failures or time, or a job
that wasn't possible before); a robot simply replacing a worker, or a line running a bit faster, is news. Funding,
deals, listings, policy, executive moves and product sales are "news": they can run, but a generic trend
("a company moves from commercial to personal sales") never crowds out the learning stories. Each story in
`judge/stories.json` gets `"kind": "learning" | "news"` by this rule; the judges score `learning_value`; and
`research_md.py` enforces the minimum, swapping the lowest-ranked news pick for the best-ranked learning story and
saying so in `research.md`. In discovery and triage, look as hard for experiments and new results as for news.

## Sourcing rule (applies to every step)

**Never generate facts or opinions.** Every fact, number, reaction, criticism or expert view in `verify/`, the cards
and `research.md` comes from a source and carries it (URL, or handle with followers and likes). That rules out:
- summaries dressed as a source: "consensus is…", "critics say…", "pushback: …" when nobody said it
- one comment promoted to a trend (a 0-like YouTube comment is not "unions will resist")
- your own ratings or inferences on a card: "strong visuals", "biggest story of the week", "the angle is indirect"
- framing labels from a subagent ("Robotics angle:", "framed as a retreat")
What a source *doesn't* say is recorded as a gap ("Trial counts not stated"), not as a criticism. Your own words are
allowed only where they are clearly ours: plain story labels, the explanation and analogy for the script, and the
editorial rules applied after the panel.

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

### 4. Cluster and choose what gets a card

The same event seen as a post, a press release, a paper and an article is one story. Merge by the subagents' `event`
names, keep every URL, and count independent sources (different outlets or accounts, not the company re-posting
itself); the count goes on the story's card in step 6.

**Which clusters get a card (typically 18–21):** every cluster with a triage *keep* or a web find, minus the gates
(no primary source, or primary date outside the window), merged when they are the same event or the same claim
(several papers making one point, e.g. "frontier models as robot policies", go on one card). This is the one step
where your own judgement picks candidates, so keep it inclusive: when in doubt, give it a card and let the panel
decide. Never leave a story out because you think it would lose. In `research.md`, list the keeps you did *not* card,
each with its reason.

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

The chatter JSON is the record: don't summarise it into a "consensus" or "pushback". Expert takes found in
articles go into the verification record as verbatim quotes with their URL (step 6). A Reality-check beat in the
script uses a criticism only when someone actually made it; otherwise it states the gap ("DeepMind hasn't said how
many trials").

### 6. Rank with an LLM judge panel

Ranking is a judgement call (is a factory pilot bigger than a paper everyone is sharing?), and LLM judges now agree
with human experts about as well as experts agree with each other on such calls. But a single judge has known biases:
it favours whatever it reads first or last, longer and more enthusiastic write-ups, and the rubric's criteria order
can flip its top pick. LLMs also tend to overvalue technically novel but vaguely described work and undervalue
concrete, high-impact news (Northwestern's newsroom study). So the ranking is a **panel**, built to cancel those biases:

1. **Verify every story that will get a card, before judging** (typically 12–20, every cluster that survived
   triage and isn't gated out below), with verification subagents applying the rules in step 8. Each writes
   `<ep>/verify/<group>.json` in this contract, and nothing else:
   ```
   [{"story": "<unique name>", "primary_url": "...", "primary_date": "YYYY-MM-DD",
     "quotes": [{"text": "<exact words from the page, copied character for character>", "url": "..."}],
     "gaps": ["<what the source doesn't state: 'Trial counts not stated'>"],
     "visuals": ["<what exists, described, not rated: 'Launch video', 'Figure 2: harm examples'>"],
     "status": "confirmed | corrected | unconfirmed", "corrections": ["<claim> is wrong: source says <quote>"]}]
   ```
   Tell them: quotes only, no paraphrase; no framing labels or commentary in any field; an expert's view counts only
   as their quoted words with the URL; if a page can't be fetched, say so in `gaps`.
   Then run `python3 -I physical-ai-daily/tools/check_quotes.py <ep>`. It downloads each quoted page once (HTML,
   gzip, PDFs via pdftotext, script-embedded and double-escaped text) and marks every quote found, not found, or
   unchecked (the page refused it). A quote **not found** goes back to verification: fix it or drop it. Cards show
   unchecked quotes as such. First run (3 weeks, 547 quotes): 512 found (49 only after ignoring punctuation and
   spacing), 0 not found, 35 unchecked on 6 sites that block automated readers.
2. **Build the cards with `tools/build_cards.py <ep>`, never by hand.** You write only `<ep>/judge/stories.json`:
   per story a plain label ("Google DeepMind: Gemini Robotics 2.5", no adjectives, numbers or hooks), and pointers
   to its verification entry, web finds, chatter files, relevance keywords and the primary date (format in the
   tool's docstring). The tool assembles each card from those sources: verbatim facts, gaps, raw signals (harvest
   count, X likes and engagement percentile, HN points, YouTube views counting only videos uploaded in the window)
   and verbatim reactions (X posts, HN titles, YouTube comments with 5+ likes). Then **read every card**:
   - a post or video that matched a keyword but is off-topic goes in that story's `exclude` with the reason (the
     card shows it); never edit card text
   - a card the tool flags as over twice the median length gets split into separate stories or fewer sources,
     because judges favour longer cards
   - a missing fact goes back to verification, not onto the card
3. **Prepare and run the judges.** `judge_panel.py prepare <ep>/judge/cards.json <ep>/judge` writes six files,
   `judge_A1.md` … `judge_C2.md`: three models, each in two orderings, every file with its own shuffle of the cards
   and of the rubric's criteria (several orderings combined cut order bias in list judgments, arXiv 2310.07712).
   Spawn six fresh subagents in one message: A1 and A2 on `opus`, B1 and B2 on `sonnet`, C1 and C2 on `fable`, each
   told only to read its file and write the JSON it asks for. The panel is all Anthropic models by the user's choice
   (2026-11-15): published panels gain most from models of *different* makers (PoLL, arXiv 2404.18796), so shared
   blind spots are possible, and YouTube feedback (item 7 below) is the check on them. Evidence for every design
   choice: `physical-ai-daily/docs/judge-evidence.md`. They score each
   story 1–10 on significance, viewer pull, evidence, showability and momentum, reasoning first, then give an overall
   score and their top 5. Don't show them your own ranking.
4. **Aggregate:** `judge_panel.py aggregate <ep>/judge` averages each model's two orderings, then ranks by the mean
   over models (Borda points from the top-5 lists break ties). It flags stories where models differ by 3+, and reports
   `order_spread`, how far one model moved between its two orderings (a large one means the card reads ambiguously). Read the flagged stories' reasoning: a disagreement often means
   a card is missing a fact, so add it to the verification record, rebuild and re-run rather than average it away.
5. **Settle the cut line pairwise:** `judge_panel.py pairs <ep>/judge` writes every pair among ranks 3–7 in both
   orders. One fresh subagent (`opus`) answers them;
   `judge_panel.py settle <ep>/judge` re-orders those ranks by wins that held in both orders. Pairwise comparison
   tracks human preference better than scores do, which is why it decides the places that matter.
6. **Apply the editorial rules yourself, after the panel** (they're facts about the lineup, not judgement):
   at least 2 learning stories (enforced by `research_md.py`, above); at most 2 stories per company; at least one research story when one is ranked within 3 places of the 5th pick;
   variety (not five humanoid funding rounds); at least one story the viewer couldn't have read on a mainstream front
   page. Moving a story for a rule: say which rule in `research.md`.

7. **Calibrate against viewers, not the user** (user, 2026-11-16). After each episode is published, fill
   `<ep>/youtube.json` as described in `physical-ai-daily/docs/youtube-feedback.md` (segment retention, Shorts,
   comments), so each story's judge scores in `scores.json` line up with how it performed. Act on corrections at
   once; tune the rubric only after 6 episodes. Until then, say in `research.md` that the picks are uncalibrated.

**Gates, before any card is written:** a story with no primary source or a credible tier-1 report with named sources
can't be shortlisted (mention it as unconfirmed at most). Buzz never rescues weak evidence. Anything older than the
window with nothing new this week is out.

**Show the panel** in `research.md`: each shortlisted story's mean overall, per-criterion means, each judge's overall
and a short quote from each judge's reasoning, so the user can argue with it.

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

### 8. Verification rules (applied in step 6 to every carded story)

Apply the verification rules in the `physical-ai-daily` skill (primary source, date on the primary source, numbers
verbatim on a capturable page, vendor-reported results labelled as such). On top of those, prazo5's faithfulness rules:

- **Use only stated facts.** Never infer an outcome, cause or reaction the source doesn't state. A filing is not a ruling.
- **Preserve epistemic status.** "Plans to", "in talks", "reportedly", "aims to", pilot, preprint: an announced plan is
  never a completed action.
- **Scope and attribution:** keep each number attached to the exact system and subset ("78% on the IDB tasks", not
  "78%"); the subject of a result is the system that achieved it, not the benchmark.
- **Qualifiers:** "first", "state-of-the-art" or "robust" only if the source says it, about the same claim.
- **Copy names, versions and numbers character for character.**
- **A blocked page is not a dead end.** Work down `physical-ai-daily/docs/blocked-sources.md` before marking a
  quote unverifiable or dropping a fact; record which step got the source.
- **Snippets:** news-API descriptions and search snippets often cut off mid-sentence. List useful ones in
  `research.md` under Snippets, marked *unverified*: knowing something was reported can shape the video. But a snippet
  is never a source for a fact or number in the narration until the full article confirms it, and a number whose unit
  or context was cut off is never used.

### 9. Write `research.md`

`python3 -I physical-ai-daily/tools/research_md.py <ep> --window FROM:TO` writes `research.md` and `scores.json` from the
panel, the verification records and the cards, so nothing in it is in your own words. Add by hand only the notes
below the generated part (degraded sources, keeps you didn't card and why, notes for the next run).

```
# Research <date> (window: <from> → <to>)
Sources: arXiv ✓ (N papers), X ✓ (N), News ✓/degraded (N), web discovery (N events from 4 beats), topics synced ✓/stale,
credits: <credits.py summary>, triage batches: N, judges: <models>

## Shortlist
| # | Story | Type | Panel (A/B/C) | Sig | Pull | Evid | Show | Mom | Primary source (date) | Key fact, verbatim | Visual | Caveat |
Then for each story:
- a 24–30 word summary (named subject first, the single most important number)
- how we'll explain it: the analogy, figure or demo
- what people say: verbatim posts and comments from the chatter record and expert quotes from the verification
  record, each attributed; "none found" when there are none. No summaries of opinion
- gaps: what the sources don't state
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
