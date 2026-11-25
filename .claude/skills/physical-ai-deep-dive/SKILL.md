---
name: physical-ai-deep-dive
description: One story, explained in depth — a 12–20 minute Pixels2Actions deep-dive episode (project physical-ai-daily) for engineers working in the field, told as a story, never a lecture. Picks the week's story with five gates and a blind judge panel, researches it in depth (paper, prior work, authors, wider field), outlines it as a causal chain (obstacle → fix → test), scripts it with evidence-based teaching rules, builds step-by-step diagrams, renders 4K light and dark, and makes the full package (titles, thumbnails, A/B set, Shorts, publish.md). Takes the week (its end date, e.g. "2026-11-22"). Use when the user says "/physical-ai-deep-dive <week>", "make the deep dive for <week>", or asks for a single-story deep-dive video. Independent of the weekly digest (weekly-package).
---

# Deep dive: one story, 12–20 minutes

A second series, alongside the weekly digest (user, 2026-11-24; the digest may stop later if this works). One story per
video. The viewer is **an engineer working in the field**: keep the full technical depth, make it easy to follow, and
never talk down ("not too simplified that you assume viewer is dumb"). Tell it as a story: "we don't want to bore
users with lectures."

The evidence behind every rule here is in three docs; read the one a step points to:
- `physical-ai-daily/docs/research-deep-dive-format.md`: channel outliers, retention, the story criteria.
- `docs/research-deep-dive-teaching.md`: how to explain.
- `docs/research-deep-dive-storytelling.md`: how to keep it a story while staying factual.

`/physical-ai-deep-dive <week>` runs **unattended**. Stop only for a HARD gap: no story passes the gates, a disk or
RAM shortage you can't fix inside the project, or a decision that is genuinely the user's.

| Step | Output (in `physical-ai-daily/episodes/<end>-deepdive/`) |
|---|---|
| 0 Preflight | the credits, disk and RAM check |
| 1 Pick the story | `audit.md` / `audit.json`, `judge/`, `pick.md` |
| 2 Research it | `research.md`, `assets.md`, `verify/<slug>.json` (quotes checked) |
| 3 Outline | `outline.md`, reviewed by a fresh agent |
| 4 Script | `episode.json` and `script-sources.md` |
| 5 Visuals | `assets/` (clips, figures), diagrams in `episode.json` |
| 6 Voice, render, critics | `out/physical-ai-daily-<end>-deepdive-{light,dark}.mp4` |
| 7 Package | `thumbs.md`, `out/thumb_*`, `out/ab-test/`, `shorts/<slug>/`, `publish.md` |
| 8 Hand-off | the chat message, plus learnings appended to this skill |

Run long steps (the audit, research, the outline review, Shorts, critics) as background subagents. Brief each with
the week, the folder, the step's section of this skill, and the rules at the end. Then check its files yourself before
the next step.

## 0. Preflight

- **The week's research must exist:** `episodes/<end>-weekly/research.md`, `verify/`, `judge/cards.json`. If it
  doesn't, run the `physical-ai-research` skill for the week first: research only, no digest episode.
- **Voice:** edge-tts `en-US-AndrewNeural` at `"speed": 1.15` (user, 2026-11-24). A 15-minute script runs about
  14,000–16,000 characters, beyond the ElevenLabs free tier; use ElevenLabs only if the user says so and
  `python3 tools/credits.py` shows the characters.
- **Disk:** check `df -h /System/Volumes/Data` and keep about 12 GB free.
  - A 16-minute 4K final is about 0.8 GB per design, plus about 1.4 GB at the encode's peak.
  - The disk is shared, so check again before each 4K render.
  - If space is short, offer the user this project's own leftovers first: old episodes' ignored `out/`, `audio/`,
    `qa/` and `assets/`. Keep the reference episode and every git-tracked file. Never delete outside the project.
- **RAM:** Remotion encodes in parallel only with about 4.2 GB free at render start; otherwise it caches every frame to
  disk. Check `memory_pressure` before each render.
- **Locks and judges:** check that `physical-ai-daily/.render.lock` isn't held by a dead process, and that the judge
  models answer. Fable's quota runs out; fall back to Haiku and say so in `pick.md`.

## 1. Pick the story (the criteria the user approved, 2026-11-24)

**Candidates:** the week's digest shortlist, the 5 picks in `research.md`.

**Audit them first.** A subagent writes `audit.md` and `audit.json`: a factual inventory per story, with no ratings.
It covers:
- whether the primary document's full text is readable, and its pages, sections, figures and tables;
- the ablations, failure analysis, stated limits and future work (the authors' words);
- the mechanism in 2–4 sentences;
- the official videos (count, durations, real or simulated, sped up or not, sound or not);
- released code, data or weights;
- the authors, their posts and talks;
- the 3–6 earlier works the paper names, and the related efforts in the wider field;
- named reactions, and the gaps.
The 2026-11-22 audit is the model to follow.

**Gates.** Every gate must pass, decided from the audit and the verification record:

| Gate | Passes when |
|---|---|
| G1 Technical primary source | A readable paper or technical report with method, data and evaluation, or a project page plus code. For news: a primary document with technical substance (filing, recall report, regulator decision, engineering post). A press release alone fails, and so does a paywalled paper whose full text we can't read |
| G2 Verified | The digest's verification status is confirmed or corrected, with no quote marked "not found" |
| G3 Enough to show | Real footage or figures of the system itself, plus about 4+ distinct visual sources, so every chapter has an anchor; the mechanism can be drawn as our own diagram |
| G4 One answerable central question | One sentence ("How did X do Y?", "Why did X fail?") that the sources actually answer. Write it, and point to the answering passage |
| G5 Fills 12 minutes without padding | At least 5 distinct sourced sub-topics: earlier work, two or more parts of the mechanism, results, failure modes or limits, implications |

**Scored criteria.** Each is scored 1–10 with reasoning first, and the anchors are in `tools/deepdive/judge.py`. The
rank is the **plain mean of D1–D7** (equal weights until calibrated); D8 only breaks ties.
- **D1 Mechanism depth:** architecture, data, recipe, control loop, hardware, and ablations showing which parts matter.
- **D2 Surprise or misconception:** a sourced result a practitioner would predict the other way.
- **D3 Known subject, unknown mechanism:** the name or demo is known, but the "how" is not explained elsewhere.
- **D4 Story arc and stakes:** the problem, earlier attempts, the turns, the result, what's open.
- **D5 Practical transfer:** code, weights, data, a recipe, a design rule, cost or hardware numbers.
- **D6 Durability:** still worth watching in 6–12 months.
- **D7 Visual depth:** success and failure clips, sim and real pairs, figures that animate.
- **D8 Momentum:** this week's attention; tie-breaker only.

**Panel.** The panel is blind, as in the digest (`docs/judge-evidence.md`):
1. `python3 -I tools/deepdive/judge.py prepare <dd> --week episodes/<end>-weekly --ids <ids> --audit-keys <words>` builds
   each card from the digest card plus that story's audit section, never by hand. It writes `judge_A1.md` … `judge_C2.md`.
2. Spawn six fresh subagents in one message: A on `opus`, B on `sonnet`, C on `fable`, two orderings each. Tell each one
   only to read its file and write the JSON it asks for, with no web access.
3. Run `judge.py aggregate <dd>`, then `judge.py pairs <dd>`. Give one fresh `opus` only `judge/pairwise.md`, then run
   `judge.py settle <dd>`. A pairwise win counts only when it holds in both orders.
4. If the models differ by 3+ on a story, read their reasoning. A card is often missing a fact: fix the audit, rebuild
   the cards and re-run rather than average it away.

**Write `pick.md`:**
- the gate table, with the reason for every failure;
- the panel's means per criterion;
- one short quote from each judge;
- the pairwise result;
- the pick and its central question.
Show it in chat when chosen, then go on. **Calibrate** after 6 episodes: log each episode's scores next to its chapter
retention in `docs/youtube-feedback.md` and tune the anchors' wording, not the arithmetic.

## 2. Research the story in depth (user: paper, prior work, authors, and the wider field)

One subagent, briefed with the sourcing rules from `physical-ai-research` step 8, writes:
- **`research.md`**, a fact base organised by story beat:
  - the beats: the problem, before this, the setup, first results, what went wrong and why, the fix(es), results
    after, variants, cost and scale, limits, what it opens, reactions, background explainers, misconceptions;
  - each fact as a plain claim, then the **exact quote**, the URL and its location (section, figure or table);
  - every number with its denominator and conditions (sim or real, which subset, which model);
  - it ends with Gaps and Open questions.
- **`assets.md`:** every figure and table (number, caption, image URL), and every clip that matters (URL, duration,
  what it shows, real or simulated, success or failure, speed if stated). Metadata only.
- **`verify/<slug>.json`** in the research skill's verification contract, holding every quote `research.md` relies on.
  Then run `python3 -I tools/check_quotes.py <dd>` and fix or drop each "not found".

The research covers the whole paper and its appendix, the released code and data, the authors' posts and talks, the
named earlier works (what each did and left open), related labs and companies (trusted sources only), authoritative
background for terms the viewer may need (official docs, surveys), named reactions, and **sourced misconceptions**:
an expectation the paper or a named researcher says people hold. Background facts may come from any authentic,
trusted source (user, 2026-11-24), and the script explains them in our own words.

## 3. Outline as a story (`outline.md`)

The video is one causal chain: **question → obstacle → fix → test → new obstacle → … → answer**. Every piece of the
mechanism comes in as the *therefore* to the *but* just before it. Nothing is explained before the viewer has a
reason to want it (`research-deep-dive-storytelling.md` section 3).

| # | Chapter | 15-min target [range] | Story job | Teaching job |
|---|---|---|---|---|
| 0 | **Cold open** (format below) | 0:00–0:40 [0:35–0:45] | Scene → nut graf (what, who, when) → the title's finding → the promise as 2–3 questions → the intro line and the sting | The study named by 20 s; the one number (with its denominator) that makes it a question |
| 1 | **The goal and the stakes** | to 1:30 [to 2:00] | What the team set out to do and why it matters (sourced). Don't repeat the nut graf or the cold open's questions | The task concretely (input, output, pass test); the 2–4 paper-specific terms |
| 2 | **The obvious approach hits the wall** | to 4:00 [2–3 min] | What a practitioner would reach for (the paper's first attempt, its baseline, or named prior work) and its documented failure. A prediction prompt before the reveal. Earlier work enters here as earlier tries | Why the obvious approach fails, from evidence; the sourced misconception named and refuted |
| 3–4 | **Therefore: one fix per chapter** (2–3 chapters) | to 8:00 [4–7 min] | Each chapter is one beat chain (below), ending on the question the next one answers | Each viewer can draw one component, say which problem it solves and what it costs |
| 5 | **The turn** | to 9:00 [0:45–1:30] | A sourced result that changes what the viewer thought did the work (an ablation, a comparison, the authors' diagnosis). **Cut it if the sources hold none; never manufacture one** | The one thing to remember, in one sentence |
| 6 | **The verdict** | to 11:30 [2–3 min] | The full test answers the central question; return to the cold-open footage. A prediction prompt before the headline number | Headline results with baselines, denominators and scope |
| 7 | **What still breaks** | to 13:30 [1.5–3 min] | The authors' stated limits and the remaining failures. No tidy ending | Where it doesn't apply, and what the evidence doesn't show |
| 8 | **What you can use; what it opens** | to 14:40 [1–2 min] | Code, data and design lessons. What the research opens, **sourced only** (the authors' future work, trusted sources), 30–60 s | What they can take away |
| 9 | **Recap as questions** | to 15:20 [0:30–0:45] | The cold-open question answered in one line; 2–3 "why" questions, each with a one-line answer | Answer without the video |

**Length:** the story decides, within 12–20 minutes (user). At 12 minutes, use two "therefore" chapters, fold the
turn into the verdict, and give chapters 7–8 about a minute each. At 20, use three "therefore" chapters, each with its
own test. **Add depth, never background or padding.**

**The beat chain** inside every "therefore" chapter:
1. **BUT:** the obstacle, shown in footage or a figure, with the paper's numbers.
2. **ASK:** the question an engineer would ask now. On 3–4 key beats in the whole video, add a prediction prompt and a
   pause.
3. **THEREFORE:** the piece the team built. The concrete instance first, then the general rule, then the paper's name
   for it, with the diagram built as the words name each part.
4. **TEST:** its sourced result, with the denominator.
5. **BUT / AND SO:** the next obstacle or the cost of the fix, which becomes the next chapter's BUT.

**Two checks on every outline:**
- **The "and then" test:** join the beats with "and then". Every joint that still reads fine is a lecture seam: find
  the real "but" or "therefore" in the source, or merge the beats.
- **The first-sentence test:** a chapter must not open "Now let's look at X". It opens with the obstacle X answers. If
  no source shows one, X goes into the verdict or a short aside.

`outline.md` lists:
- the central question;
- the one thing to remember;
- per chapter: its BUT / ASK / THEREFORE / TEST / next BUT, each with the `research.md` facts it rests on;
- the visual anchor for each chapter.

Then a **fresh subagent reviews the outline** against both tests and the factual rules below, and against
`research.md` only. Fix what it finds before scripting.

## 4. Script (`episode.json` + `script-sources.md`)

**The opening** (format approved by the user, 2026-11-25; evidence in `research-deep-dive-intro.md`):
The first draft never said what the video was about until 0:45, and the user caught it. In our niche's outlier videos
the subject is named at a median of 5 s (research videos 16 s), and the source at 18 s.

| Slot | By | Job |
|---|---|---|
| 1. Scene | 0–10 s | The decisive moment on footage, outcome first. Narration starts in the first second; at most 2 sentences |
| 2. Nut graf | ≤ 20 s | **What** (the study or project, by name), **who** (lab or company), **when or where** (date, venue), and what they did, in 1–2 sentences |
| 3. Finding | ≤ 30 s | The headline finding the title promises, with its denominator |
| 4. Promise | ≤ 40 s | 2–3 questions in chapter order (≤ 8 s), unanswered. Footage under them where it exists |
| 5. Intro line | after 4 | Word for word: **"This is Pixels to Actions. We track the latest work in robotics and physical AI."** (user, 2026-11-24). Then the sting |

- No logo, greeting or "In this video, I…" before the scene.
- Time order: at most one jump, marked in words ("In the first round…", "This robot came later…"). A clip that isn't
  the event being described carries its label.
- Chapter 1 doesn't repeat the nut graf or the questions.
- **Gate:** before the full render, a fresh reviewer gets only the first-minute transcript (with times) and a stills
  sheet. By 30 s they must be able to say what the video is about and who did it, and by the end of slot 4 what
  they'll get from it.
- Say what happened plainly and early. Inform, don't sell.
- Dated framing: say when it was published ("last week", "on November 18th"), never "today" or "just announced".

**Teaching rules** (`research-deep-dive-teaching.md` section 2 has the evidence):
- **Jargon in two bins.**
  - Field-standard terms (policy, reward, sim-to-real, PID, transformer, URDF) are used without definition.
  - Paper-specific or neighbouring-field terms are defined once, at first use, in one clause, with the term on screen,
    then used identically every time.
  - Never open a section with a definition: the term arrives at the end of its example ("that loop is what the paper
    calls…").
- **Concrete before abstract.** Use one real design, failure or number first; the general rule after it; the name
  last. With two cases, compare them, then state the principle.
- **First-order model first,** then add one complication at a time, each with its reason.
- **Refute a sourced misconception** with evidence (d ≈ 0.8). Never a straw man.
- **Numbers:**
  - one per spoken sentence, always with its denominator and its comparison (baseline, prior best, chance, trials);
  - said as words in the narration and shown as digits on screen;
  - hedge once, precisely, at the claim ("in simulation only; 15 were built"), not with vague qualifiers.
- **Analogies:**
  - An engineer's best analogy is usually another technical system ("this is X, except…"); use an everyday analogy
    only when none exists.
  - Label it as ours ("think of it like…"), map its parts out loud, and say where it breaks.
  - Never use an analogy as evidence.
- **Voice:**
  - Talk like a colleague over lunch: second person, plain verbs, no cheerleading ("don't worry, this is easy") and no
    hype adjectives.
  - Don't narrate what the diagram already shows an engineer. Narrate the *why*.
  - Pauses, not a slower voice: keep 1.15×, and add a pause at chapter boundaries, after dense sentences and after
    each prediction prompt.
- **Signposts:** open each chapter with its question. Mark the boundary with a one-line "where we are", the chapter
  card and a pause. No re-explaining recaps mid-video.

**Factual rules for the story** (`research-deep-dive-storytelling.md` section 4). Story is the order and the joints,
never extra content.
- **May be framed** (our editorial choices, never presented as anyone's words or thoughts):
  - the central question, the opening moment, the chapter order and titles;
  - calling a documented failure an obstacle;
  - prediction prompts;
  - labelled analogies;
  - an order different from the paper's, as long as we don't call it chronological.
- **Must be sourced:**
  - every event and its order ("first they tried X" needs a source stating the sequence; otherwise say "the paper
    compares");
  - every stake;
  - every refuted expectation;
  - every "surprising";
  - the reversal;
  - every failure shown, with its N of M and the clip's speed (or "speed not stated");
  - every reaction (a named person, with a date).
- **Never:**
  - invented quotes, thoughts, feelings or motives ("the team was stunned");
  - wants or feelings for the robot or the agent: describe its logged actions literally;
  - invented conflict (rivals, races, villains, "nobody believed them");
  - manufactured twists or fake cliffhangers;
  - composite or reordered events presented as what happened;
  - dropping a caveat because it breaks the arc;
  - one vivid case presented as typical;
  - stakes beyond the source.
- **Pre-recording check:** tag every sentence in `script-sources.md` as *event*, *number*, *reaction* or *framing*.
  The first three point to a quote in `verify/<slug>.json`. Framing must read as our question or our structure.

**episode.json:**
- One segment per chapter: `{"id", "kind": "open" | "chapter" | "outro", "headline", "chapter": <n>, "source",
  "sourceDate", "beats": [...]}`.
- Top level: `"speed": 1.15`, `"design": "blueprint-light"`, `"title"`, `"date"`.
- One continuous TTS take per segment, cut at natural pauses (`voice.py`), never per-beat pieces.
- The visual schema is documented at the top of `video/src/deepdive/DeepDive.tsx`.

## 5. Visuals

**Match the visual to the sentence's job** (`research-deep-dive-teaching.md` section 3):

| Sentence job | Visual |
|---|---|
| Mechanism | Our diagram (`build`, `loop`, `timeline`), built step by step as the narration names each part, with labels on the parts and one highlight at a time (the rest dimmed) |
| Evidence | The real thing: footage (`clip`, `pair` for sim vs real) or the paper's figure (`figure`, with regions revealed in turn), credited |
| Number | `number`: the value, its comparison, its denominator |
| Limit | The actual failure clip, or the gap plot |
| Prediction | `predict`: the question and 2–3 options, a beat of silence, then the answer marked |
| Reaction | `quote`: a named person, their role, the source |

**Never use generic B-roll under an explanation.** No drift, no decorative motion. Every movement has a purpose.

**Footage first** (critic, 2026-11-25). The first RoboBuilder draft was 81% boxes and cards and 9% footage, with a
217 s stretch and no clip at all, even though the project had released 233 clips. The adversary rated it "plays as a
lecture" (S1). So:
- **Event beats show the event moving.** Every beat that describes an event (a robot failing, slipping, crossing, a
  bench test, a redesign) shows it moving where an official clip exists.
- **Footage targets:** footage is at least 30% of the runtime, and no stretch longer than 45 s goes without real
  footage or an animated mechanism.
- **Diagrams carry mechanisms only.** A box with a phrase in it is not a visual.
- **Plan before the script is final.** Write a per-beat visual plan from `assets.md` before scripting finishes, and
  count the footage share.
- **Sim-vs-real comparisons:** both panels at full opacity on the key word, the clips aligned so the divergence happens
  together, a freeze with a marker on the divergence point, and the robot at least 35% of its panel's width.
- **The payoff is seen.** When the narration says a robot "crosses", show it past the last obstacle, not still on it.
- **Numbers:** every spoken headline number is on screen on its word, with its n. A series of numbers belongs in a
  chart (`bars`), not in separate cards.
- **Audit every clip against its metadata before the final render** (critic round 3, 2026-11-25). Record in
  `qa/clip-audit.md` its agent, task, level, round, real or simulated, and fitted or unfitted simulator.
  - Round 3's adversary found two real errors in a near-final cut: "first fifteen" failure tiles that were the team's
    reference car, and a clip labelled "already fitted" playing over the unfitted simulator.
  - It also found a chart legend that called the real bars "sim". Check every legend and label against its data too.
  - The Shorts reuse the episode's clips, so check them against the same audit.

**Paper figures:** show the original briefly with a credit, so viewers see it's real. Then crop to the panel under
discussion, or redraw it and reveal one series at a time. Use only the one equation that carries the key idea, built
term by term. Screenshots and figures hold still: a straight cut-in, never a curved pan.

**Footage:**
- **Official footage first.** A few seconds per clip, muted, credited on screen and in the description.
  - Cut with `tools/clip.py`. It refuses shots under 1.5 s unless `--allow-short-shots "<why>"` is given.
  - Raw downloads are git-ignored and deleted after cutting.
- **Non-official footage** (a talk, another channel) is allowed **with credit, without asking per clip** (user,
  2026-11-24), only where no official footage shows the point. Credit the channel and URL on screen and in the
  description, and list every non-official clip in the hand-off.
- **AI-generated media only when nothing real can show the point.** Tell the user the credit cost before generating,
  and stamp it "AI-generated" on screen. That label is for media we generated, never for sources.
- **Tags** ("Simulation", "Sped up 8×", "Preprint") stay small and quiet. Every sped-up or simulated clip is tagged.
- The visual changes every 5–7 s. A diagram that builds a new part counts as a change; a static hold doesn't.

## 6. Voice, render and critic gates

1. `.venv/bin/python tools/voice.py <dd>`, unchanged. It writes `props.json`. Read its WARN lines about long static
   holds, and split those beats.
2. **Stills gate:** `tools/deepdive/render.sh <dd> --stills-only`, plus stills at the moment each diagram is fully
   built (`tools/deepdive/stills.sh`). Look at every one yourself. Then run the `video-critic` skill's stills packet: a
   fresh reviewer, then an adversary. Fix the severe findings.
3. **Draft gate:** run `SCALE=1 CRF=23 tools/deepdive/render.sh <dd> --skip-voice` for a 1080p draft. Run `video-critic`
   on it (the full packet: per-shot sheets with the narration under each, motion strips, seams, loudness, and the
   narration-vs-script transcript check). Fix, and re-run until no high-severity finding is left. Delete the draft
   afterwards.
   - **Shots file:** write `qa/shots.json` from `props.json`, one shot per beat (`t0`/`t1` from the beat durations;
     the 1.6 s sting after the open segment).
   - **Reviewers:** a 13-minute episode has about 90 shots, so split the 16 sheets across three fresh reviewers, each
     given the whole `packet.json`.
   - **Separate finding IDs:** give each reviewer its own range (F0xx, F2xx, F3xx). `merge_review.py` keys findings by
     ID, and in round 1 two reviews both used F001… and the merge silently dropped one review's findings for the first
     4 minutes.
   - **Regression lists:** later rounds get the previous `findings.md` and report a `prior` status for each item.
   - **The adversary matters most.** In round 1 it was the one that saw the film plays as a lecture: each reviewer saw
     a third of the film and noted it only in passing.
   - **Whisper:** the narration check needs Python with `faster_whisper`; none is installed (2026-11-25), so say the
     check was skipped.
   - **Allow for three rounds:** the 2026-11-25 draft took about 2.5 h to voice, render and review once.
4. **Final:** run `tools/deepdive/render_both.sh <dd>` for 4K light and dark (shared lock, about −14 LUFS, shot check).
   Check both with `ffprobe` (3840×2160, duration) and look at a frame sheet of the dark render.

## 7. Package (the full package, as in the digest)

Rules carry over from `youtube-thumbnails` and `promo-shorts` (read them; don't edit them). These are the deep-dive
differences.

**Titles:**
- At least 8, each tagged by angle, always including *current* (it names the lab, robot or model) and *plainer* (a
  newcomer gets it). Mark one ★, and give each the thumbnail it pairs with.
- The title states the central question's answer or its anomaly, concretely: a robot or system doing something, in
  plain words first and the name last. Statements, not questions; no hype words; no "Robotics this week:" prefix. An
  honest catch from the source is a good angle.
- Title and thumbnail words complement each other and never repeat.

**Thumbnails:**
- 8–10 candidates in the `moment` layout: one real frame of the decisive instant, 0–3 words, and a label naming the
  robot and its type. No receipts.
- Use the existing `Thumbnail` composition, unchanged. Render light and dark, and make one lettered contact sheet.

**A/B set (Test & Compare):**
- 3 pairs, each a different idea: outcome vs catch vs how. ★ goes first.
- Rendered to `out/ab-test/<n>-<slug>-{light,dark}.png`, and checked at 168×94.
- Logged in `docs/youtube-feedback.md`, marked as a deep dive.

**Shorts:**
- 3–4, each from a different chapter's decisive moment, in the promo-shorts newcomer format at 1.175×, with a moment
  cover (3–4 cover options, plus a sheet).
- The end card is the deep dive's thumbnail with "FULL EPISODE ↓". The user uploads the **dark** versions.
- Each description starts "Full episode: <EPISODE_URL>".

**`publish.md`:**
- the ★ title and all options with their angles;
- the description:
  - names what the video explains and when the work was published;
  - chapters with timestamps from `props.json` (one per segment, the first at 0:00; the first chapter's title is the
    central question);
  - every source URL, including the earlier works and background sources;
  - clip credits, official and non-official;
  - up to 15 hashtags, no tags;
- the A/B section ready to paste;
- the MP4 paths;
- each Short's options, description, cover, dark MP4 and Related video setting;
- a posting order.

## 8. Hand-off, then learnings

The hand-off message, in chat:
- every path the user uploads;
- the A/B set in full, and every title option with its angle;
- the contact sheets;
- the pick and why (the gate table and the panel means, one line each);
- the voice;
- every non-official clip (owner and URL);
- any AI-generated media and its cost;
- what was dropped or unverified, and why;
- the critic rounds and what they changed;
- any credits spent.
Tell them to start a Test & Compare test at publish, never edit mid-test, and act only on a "Winner" label.

Then append what this run taught (tool failures, workarounds, timing) to **this skill**, not to the digest's skills.
Commit code, skill and doc changes only.

## Rules that carry over (user decisions; keep them)

- **Never generate facts or opinions.** Every number in narration, on screen and in descriptions is on a captured page.
- **Never upload, never post.** Uploading is the user's action.
- **Keys** live only in the git-ignored `physical-ai-daily/.env`. Never print or commit them.
- **The repo is public:** never commit `.env`, `p2a_topics.json`, media, or episode working files.
- **Blocked sources:** follow `docs/blocked-sources.md`. Paywalls are never bypassed. Downloaded reference material is
  for private study and is deleted afterwards.
- **Episode format:** 4K, light (default) and dark. Captions on, compact, bottom band. No third text stream.
- **Permission refusals** from Claude Code are final: don't retry them another way.
- **Experiments** (new formats, A/B builds of this skill) live on a branch and in a separate episode folder until the
  user reviews them. Merge only after their review.
