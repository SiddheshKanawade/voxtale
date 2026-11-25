# Telling one research story without lecturing: evidence review

Compiled 2026-11-24 for the Pixels2Actions deep-dive format (one story per video, 12–20 min, engineer viewers, faceless TTS, Remotion visuals). It answers the user's feedback on the draft structure: "combine this with story telling skill. We don't want to bore users with lectures."

**Builds on, does not repeat:** `docs/research-deep-dive-teaching.md` (IDs L#, V#, P#, and its "teaching rules"), `docs/research-deep-dive-format.md` (IDs S#, E#, and the DRAFT structure in its section 4), and `docs/attention.md`.

**Method.** Web searches, then the primary pages where they could be reached (PMC, publisher pages, PDFs). Blocked pages (Taylor & Francis, Springer, MDPI, City repository) were read through Firecrawl. Two creator interviews were read from YouTube auto-captions pulled with yt-dlp (metadata and captions only, deleted afterwards). Where only an abstract or a second-hand description was read, the table says so. Nothing is quoted from memory.

**Strength labels** (as in the teaching doc): **strong** (meta-analysis or several consistent RCTs), **moderate** (small or off-setting RCTs, abstract-only meta-analyses), **weak** (one small or correlational study, or a second-hand report), **practitioner** (a creator's or expert's statement, untested).

**The big caveat.** No study tested story structure in a 12–20 minute explainer for working engineers. The learning evidence is mostly texts read by students; the persuasion evidence measures belief change, not understanding; every structural device (but/therefore, try-fail, midpoint reversal, open loops) is practitioner advice. Our own chapter retention will outrank all of it.

---

## 1. Sources

### 1a. What narrative does (evidence)

| # | Claim | Source | Evidence type | Strength |
|---|---|---|---|---|
| N1 | Narrative is defined by causality, temporality and character; it argues from a specific case up to a general claim. Cites Graesser and colleagues: narrative text read about twice as fast and recalled about twice as well. Ethical questions it raises: representativeness (a worst case is not a typical case) and persuasion vs comprehension as the goal | Dahlstrom 2014, *PNAS* 111(S4), [PMC4183170](https://pmc.ncbi.nlm.nih.gov/articles/PMC4183170) | review essay | moderate; the "twice" figures are second-hand (Graesser originals not read) |
| N2 | Narrative vs expository text: g = .55 (CI .31–.79), 37 articles, 150 effect sizes, N = 33,078. Memory .72, comprehension .48; adults .44 vs non-adults .67 (difference n.s.); listening .61. I² = 98%. Egger's test suggests publication bias in the memory studies. The authors do not recommend recasting all content as narrative. "stories were more easily understood and better recalled than essays" | Mar, Li, Nguyen & Ta 2021, *Psychon Bull Rev* 28, [PMC8219577](https://pmc.ncbi.nlm.nih.gov/articles/PMC8219577/) | meta-analysis | strong for direction; size uncertain |
| N3 | Science education, same content: narrative vs expository g = 0.16 (CI 0.03–0.30), high between-study variation. Narratives about scientists and their discoveries showed larger effects; narratives with fictional or humanized actors showed the opposite. Positive at university level | Tobler et al. 2024a, EdArXiv meta-analysis, as reported in N4 | meta-analysis (preprint), read second-hand | moderate |
| N4 | 109 undergraduate science students: narrative instruction gave higher transfer. Tentative: students with more biology background did as well or better with the expository text. "learning with narratives resulted in higher knowledge transfer" | Tobler et al. 2025, *Instructional Science*, [Springer](https://link.springer.com/article/10.1007/s11251-025-09723-w) | RCT (design-based) | moderate–weak (one study) |
| N5 | Recall tracked prior knowledge only for the expository version; expository readers built a weaker textbase but a marginally stronger situation model | Wolfe & Woodwyk 2010, *Br J Educ Psychol*, [doi](https://doi.org/10.1348/000709910X485700) (abstract via search excerpt; author PDF 404) | small experiments | weak |
| N6 | Discovery narratives (scientists' motives and obstacles) improved comprehension and retention over textbook text, most for low-prior-knowledge readers | Arya & Maul 2012, *J Educ Psychol*, described in later work ([escholarship](https://escholarship.org/uc/item/616549z3)) | RCT, read second-hand | weak (paper not read) |
| N7 | Transportation (absorption in a story) raised story-consistent beliefs (Exps. 1, 3, 4; N = 97–274). In Exp. 2 (N = 69), "highly transported readers found fewer false notes in a story". Labelling the story fact or fiction generally made no difference | Green & Brock 2000, *JPSP* 79(5), [scan](https://gwern.net/docs/www/pdfs.semanticscholar.org/39691cee32e52fbfc33f1a0316aa0d008cb699a4.pdf) (abstract read) | experiments | moderate |
| N8 | Transportation meta-analysis: 132 effect sizes from 76 articles; a model of its antecedents and consequences | van Laer et al. 2014, *J Consum Res* 40(5), [City repository](https://openaccess.city.ac.uk/id/eprint/6755/) (abstract only) | meta-analysis | moderate (no numbers read) |
| N9 | Narrative vs non-narrative controls: small persuasive effects, r ≈ .17 (beliefs), .19 (attitudes), .17 (intentions) | Braddock & Dillard 2016, *Commun Monogr*, [doi](https://doi.org/10.1080/03637751.2015.1128555) (403; numbers from secondary summaries) | meta-analysis | moderate; numbers unverified |
| N10 | Anecdotes beat statistics when emotional engagement is high (severe threat, health, oneself); statistics win when it is low. 61 papers | Freling et al. 2020, *OBHDP* 160, [abstract](https://econpapers.repec.org/RePEc:eee:jobhdp:v:160:y:2020:i:c:p:51-67) | meta-analysis (abstract) | moderate |
| N11 | Evidence in story order swayed mock jurors: guilty verdicts 78% when the prosecution's evidence came in story order and the defence's did not, 31% in the reverse; confidence was highest in story order | Pennington & Hastie 1988, *JEP:LMC* 14, via [OnFiction](https://www.onfiction.ca/2011/09/juries-and-stories.html) | experiment, read second-hand | weak (numbers unchecked) |
| N12 | 732 climate abstracts: more narrative abstracts were cited more, but narrativity explained about 5% of citation variance; journal alone explained 43%. "articles with more narrative abstracts are cited more often" | Hillier, Kelly & Klinger 2016, *PLoS ONE*, [PMC5158318](https://pmc.ncbi.nlm.nih.gov/articles/PMC5158318) | correlational; expert readers | weak |
| N13 | 867 viewers, one climate video, two narrations: infotainment lowered perceived seriousness in the younger, no-degree cluster. The authors' earlier study (described there) found degree holders preferred and believed the expository narration. "making light of a topic can reduce perceptions about its seriousness" | Davis et al. 2022, *Sustainability* 14, [MDPI](https://www.mdpi.com/2071-1050/14/17/10659) | randomized online | moderate (one topic) |
| N14 | Order sets the feeling: suspense = outcome postponed; curiosity = outcome shown before the events that led to it; surprise = an unexpected event | Hoeken & van Vliet 2000, *Poetics* 27 (abstract via search) | experiment | weak (abstract only) |
| N15 | No memory advantage for unfinished tasks (Zeigarnik); the tendency to resume interrupted tasks (Ovsiankina) holds | Ghibellini & Meier 2025, *Hum Soc Sci Commun* 12, [abstract](https://ideas.repec.org/a/pal/palcom/v12y2025i1d10.1057_s41599-025-05000-w.html) | meta-analysis | moderate |
| N16 | Most-viewed science videos: one central story (OR 3.16) and arousal (OR 3.0); dramatic question, insight and moments of change only bivariate | Huang & Grant 2020 (= format doc S9) | correlational, n = 306 | as rated there |

### 1b. Where storytelling misleads

| # | Claim | Source | Evidence type | Strength |
|---|---|---|---|---|
| M1 | Nearly all experiments allow several interpretations, and "storytelling actively denies this fact of science". Story-told papers bend figures to one line, push deviant data to the supplement, and penalise honest unexplained observations. Stories of discovery are the legitimate use | Katz 2013, *Nat Methods* 10:1045, [PDF](https://yarden.github.io/pdfs/YK_AgainstStorytelling.pdf) (read in full) | expert correspondence | practitioner |
| M2 | Kurzgesagt says its addiction video built a compelling story around one contested position and distorted the science; it was deleted. Its most successful videos were not necessarily its most accurate. Fix: books and papers, then two rounds of expert review | "Can You Trust Kurzgesagt Videos?" (2019), machine transcript at [Rosetta](https://rosetta.to/u/kurzgesagt/can-you-trust-kurzgesagt-videos) | creator statement | practitioner |
| M3 | Plot is events plus the causal "why?" (Forster's distinction); emotion matters; communicators must stay bound by the facts | Martinez-Conde & Macknik 2017, *PNAS* 114(31), [PMC5547661](https://pmc.ncbi.nlm.nih.gov/articles/PMC5547661) | opinion | practitioner |

### 1c. Story tools and creators (practitioner)

| # | Claim | Source | Evidence type | Strength |
|---|---|---|---|---|
| F1 | Beats should be joined by "but" or "therefore", never "and then"; they check outlines against this rule | Parker & Stone, NYU class (mtvU "Stand In", 2011), via [Cracked](https://cracked.com/article_43358_trey-parker-and-matt-stones-surprise-nyu-class-contains-some-great-writing-advice.html) | practitioner (secondary transcript) | practitioner |
| F2 | ABT: And (setup), But (problem), Therefore (consequence). Narrative Index = buts/ands × 100, from Olson's own text counts; FiveThirtyEight's editors were skeptical. No analysis of paper abstracts by Olson was found | [ABT Agenda](https://abtagenda.substack.com/p/the-two-narrative-metrics); [NCSE](https://ncse.ngo/and-therefore-randy-olson-and-art-science-storytelling-part-1) | practitioner | practitioner, unvalidated |
| F3 | Story spine: Once upon a time / Every day / But one day / Because of that (×3) / Until finally / Ever since then | Kenn Adams, [NPR 2026](https://www.npr.org/transcripts/nx-s1-5750619) | practitioner | practitioner |
| F4 | Try-fail cycles; each try ends "yes, but" or "no, and"; failures must arise from the plot; no fixed number of cycles | [Sanderson lecture 3 (2025)](https://www.brandonsanderson.com/blogs/blog/brandon-sandersons-2025-overview-of-story-structure-lecture-3); [Writing Excuses 10.29](https://writingexcuses.com/writing-excuses-10-29-why-should-my-characters-fail-spectacularly/) | practitioner | practitioner |
| F5 | Story circle: You, Need, Go, Search, Find, Take, Return, Change | [StudioBinder](https://studiobinder.com/blog/dan-harmon-story-circle) (Harmon's own page not reached) | practitioner | practitioner |
| F6 | Cialdini opens each lecture with a mystery common sense can't explain and solves it by the end: "Questions demand answers, but a mystery demands something more valuable – explanation." No data | [APS Observer](https://www.psychologicalscience.org/observer/those-who-can-teach) | practitioner | practitioner |
| C1 | Muller: no thesis statement first; create the question or mystery, then move through the logic to the solution; "introduce the scientific information right when you need it"; no intros; open with a mystery, a question or a cold-open story. In his teaching research, a demo without a forced prediction taught about as much as no demo ([SciAm](https://www.scientificamerican.com/article/how-youtube-star-derek-muller-of-veritasium-is-challenging-scientific/)) | Fireside Lisbon, 2026-02-27, [YouTube](https://www.youtube.com/watch?v=PEzwH50tyPw) (auto-captions) | interview | practitioner |
| C2 | Harris: "every story is effectively a promise"; confirm the thumbnail's promise within the first minute and pay it by the end; ground technical stretches in people and real tension | *How I Write* (David Perell), [YouTube](https://www.youtube.com/watch?v=zq4b96m1AvM) (auto-captions) | interview | practitioner |
| C3 | Asianometry: "A lot of the videos come from a story approach"; more interested in how technologies came about than in the technology itself | [ChinaTalk](https://www.chinatalk.media/p/semis-101-with-asianometry-and-fabricated) | interview | practitioner |
| C4 | Joss Fong (Howtown): hosts act as audience stand-ins who voice confusion; about half of a six-week feature is research (paraphrased by the interviewer) | [Storybench](https://www.storybench.org/?p=24033) | interview | practitioner |
| C5 | Welch Labs: the most compelling, clear question, carried by human narratives | teaching doc P8 | interview | practitioner |

**Not found:** primary method statements from Wendover / Half as Interesting, Lemmino and ColdFusion; Real Engineering's only on-record line found was that "the writing, in particular, is difficult" ([PorterShed](https://portershed.com/?p=2086)).

---

## 2. Findings

### Evidence

1. **Stories help, modestly.** Narrative text beats expository text on understanding and recall (N2, g = .55, with publication bias in the memory studies). For the same science content the estimate is small, g = 0.16 (N3). Expect a modest learning gain. Gains in interest were reported too, but by single studies (N4), and the link to popularity is correlational (N16).
2. **The kind of story matters.** Stories about real scientists and their discoveries did better; fictional or humanized actors did worse (N3, N6). So the protagonist is the real team and its real attempts, the problem is the opponent, and the robot is described, never personified.
3. **Expertise shrinks the benefit and may reverse it** (N4 tentative, N5; N2's adult estimate is smaller). This matches expertise reversal (teaching doc L1, L2). For engineers the story is the skeleton (order, motive, the next question), while each chapter's explanation stays dense and exact.
4. **Experts respond to story structure weakly, and to a jokey tone badly.** Narrative abstracts get slightly more citations (N12, about 5% of variance). Degree holders preferred and believed a plain expository narration over infotainment (N13). So: story structure, plain colleague tone, no gags.
5. **Order makes the feeling.** Showing the outcome first creates curiosity about how; postponing it creates suspense; an unexpected event creates surprise (N14). Curiosity peaks when viewers are about half-sure (L15), and a prediction before the reveal is what turns surprise into learning (L16).
6. **"Open loops" don't help memory** (N15). They may keep people watching (the resumption tendency), but that is untested for video.
7. **The power that holds attention also lowers scrutiny.** Absorbed readers caught fewer false notes (N7); narratives shift beliefs (N9); story order raised verdicts and confidence (N11); vivid anecdotes outweigh statistics when emotion is engaged (N10). For an expert channel this is the main risk, and why section 5 exists.

### Practitioner opinion

1. **Mystery first, science on demand.** Pose the question, then bring in each piece of science at the moment the story needs it, never thesis-then-evidence (C1, F6). This is the direct answer to "no lectures".
2. **A story is a promise.** Confirm the title's promise in the first minute and pay it off by the end (C2; format doc S19).
3. **Causal joints, not lists.** Join beats with but/therefore (F1, F2) and move by try-fail cycles whose failures come from the material (F4). Plot is "why?", not "and then?" (M3). These are fiction tools; we take only the causal skeleton.
4. **Technical story channels anchor in history and people** (C3, C5, C2). For a single paper, the "history" is the paper's own sequence of attempts plus the prior work it names.
5. **The documented way story-first science goes wrong** is building the story around one position and dropping what doesn't fit (M1, M2).

---

## 3. REVISED episode structure (replaces format doc section 4)

**Principle.** The video is one causal chain: question → obstacle → fix → test → new obstacle → … → answer. Every mechanism piece enters as the *therefore* to the *but* that came just before it (C1, F1). Nothing is explained before the viewer has a reason to want it.

| # | Chapter | 15 min [range] | Story job | Teaching job: by the end the viewer can… | Basis |
|---|---|---|---|---|---|
| 0 | **Cold open: the anomaly** | 0:00–0:30 [0:20–0:40] | **Hook.** Outcome first (curiosity order): the real decisive moment, the central question as a mystery, the title's promise shown to be true. No intro, no logo before it | …state the central question and the one number that makes it a question, with its denominator | C1, C2, N14; format E1 |
| 1 | **The goal and the stakes** | 0:30–1:30 [to 2:00] | **Protagonist and stakes.** Who the team is, what they set out to do, why it matters (sourced). Roadmap spoken as the 2–3 questions the story will answer, chapter chips on screen | …describe the task concretely (input, output, pass test) and know the 2–4 paper-specific terms | N3, C5; teaching rules 2, 12; L3 signalling |
| 2 | **The obvious approach hits the wall** | 1:30–4:00 [2–3 min] | **Obstacle and try-fail #1.** The approach a practitioner would reach for (the paper's first attempt, its baseline, or named prior work) and its documented failure. Prediction prompt before the reveal. "Before this" lineage enters here as earlier tries, only the ones the paper names | …explain why the obvious approach fails, from evidence; the tempting belief (sourced) is named and refuted | teaching rules 7–8 (L9, L11, L12); F4 |
| 3–4 | **Therefore: the fix, one piece per chapter** (2–3 chapters) | 4:00–8:00 [4–7 min] | **Try-fail turns.** Each chapter is one beat chain (below): the obstacle, the piece the team built, its test, and what is still wrong ("yes, but" / "no, and"). Each ends on the question the next answers | …draw one component of the mechanism, say which problem it solves and what it costs | C1, F1, F4; teaching rules 6, 9; visual rules 1–5 |
| 5 | **The turn** | 8:00–9:00 [0:45–1:30] | **Reveal or midpoint reversal.** A sourced result that changes what the viewer thought was doing the work (an ablation, a comparison, the authors' own diagnosis). If the sources hold no such result, cut the chapter; never manufacture one | …say the key insight, the one thing to remember, in one sentence | N16; teaching rule 2 |
| 6 | **The verdict** | 9:00–11:30 [2–3 min] | **Climax and payoff of the cold open.** The full test answers the central question. A prediction prompt before the headline number. Return to the cold-open footage where possible | …quote the headline results with baselines, denominators and scope (sim vs real, which subset) | teaching rules 10–11; L16; N10 |
| 7 | **What still breaks** | 11:30–13:30 [1.5–3 min] | **The conflict left open.** The authors' stated limits and the failures that remain. No tidy ending | …say where the result does not apply and what the evidence does not show | teaching rule 17; M1, M2 |
| 8 | **What you can use; the open question** | 13:30–14:40 [1–2 min] | **"Ever since then."** What changes for an engineer; the open question handed to the field (the authors' future work), not a cliffhanger of ours | …name the code, data or design lesson they can use | format D5; F3 |
| 9 | **Recap as questions** | 14:40–15:20 [0:30–0:45] | **Bookend.** The cold-open question answered in one line, then 2–3 "why" questions with one-line answers | …answer the central question and the why-questions without the video | teaching rule 19 |

**Scaling.** 12 min: two "therefore" chapters; fold the turn into the verdict; chapters 7–8 about 1 min each. 20 min: three "therefore" chapters, each with its own test, and longer verdict and limits. Add depth, not background.

**What changed from the draft.** "Stakes and roadmap" now names a protagonist and speaks the roadmap as questions. "Before this" is no longer a history lecture; earlier work appears in chapter 2 as earlier tries. "Mechanism chapters" become "therefore" chapters, each opened by a failure. "Results" is split: each fix is tested inside its own chapter, the reversal gets chapter 5, and the verdict pays off the cold open. "Where it breaks", "what you can use" and the recap keep their jobs.

### How the mechanism is taught inside the story: the beat chain

Every "therefore" chapter is built from this chain:

1. **BUT** — the obstacle, shown (real footage or the paper's figure), with the paper's numbers.
2. **ASK** — the question an engineer would ask now; on 3–4 key beats, a prediction prompt and a pause.
3. **THEREFORE** — the piece the team built: concrete instance first, then the general rule, then the paper's name for it, with the diagram built in sync with the words.
4. **TEST** — the sourced result of that piece, with its denominator.
5. **BUT / AND SO** — the next obstacle the result exposes, or the cost of the fix. This becomes the next chapter's BUT.

**Two checks on every outline.**
- *The "and then" test (F1).* Read the beat list joined only by "and then". Each joint that still reads fine is a lecture seam: find the real "but" or "therefore" in the source, or merge the beats.
- *The first-sentence test.* If a chapter could open with "Now let's look at module X", rewrite it to open with the obstacle module X answers. If the sources show no such obstacle, X goes into the verdict or a short aside; we never invent drama to motivate it.

**Story spine crosswalk (F3), as a sanity check:** Every day = how it's done (ch. 1) · But one day = the failure (ch. 2) · Because of that ×2–3 = the therefore chapters (3–4) · Until finally = the turn and the verdict (5–6) · Ever since then = the limits and the open question (7–8).

---

## 4. Rules for staying factual while telling a story

The channel rule stands: every fact and every reaction is sourced. Story is the *order and the joints*, never extra content.

**May be framed (our editorial choices; never presented as the team's words or thoughts)**
1. Which question is central, which moment opens the video, the chapter order and the chapter titles.
2. Calling a documented failure an "obstacle" and a documented fix a "turn".
3. Prediction prompts and "you might expect…" addressed to the viewer, with the real outcome shown right after.
4. Analogies, labelled as ours (teaching rules 15–16).
5. A presentation order that differs from the paper's, as long as we don't call it the chronological order.

**Must be sourced (in the script's source map; on screen where it is a number)**
1. **Every event and its order.** "First they tried X, then Y" needs a source that states the sequence. If the paper only reports a comparison, say "the paper compares", not "they tried". Papers usually present a logical order, not a diary.
2. **Every stake.** The paper's introduction or a named outside source, at the scale the source claims.
3. **Every expectation we refute.** The paper's framing, its related work, or a named researcher (teaching rule 8). Never a straw man.
4. **Every "surprising".** Only if a source says so. Otherwise show the number against the baseline and let it land.
5. **The reversal and any red herring.** A red herring must be a hypothesis someone documented (a baseline, an ablation, a prior method, an earlier version of the system), and it is resolved on screen.
6. **Every failure shown.** Its denominator ("N of M") next to the clip (N1, N10), and the clip's playback speed (if the source doesn't state it, say so).
7. **Every reaction.** A named person or account with a date, as in the digest.

**Never allowed**
1. Invented quotes, thoughts, feelings, motives or reactions ("the team was stunned").
2. Giving the robot or the AI agent wants or feelings. Describe logged actions literally ("the agent requested bench measurements"). Humanized actors also made science narratives worse (N3).
3. Invented conflict: rivals, races, deadlines, villains, "nobody believed them".
4. A manufactured twist, a fake cliffhanger, or withholding an answer past the chapter that promised it.
5. Composite events, or a reordered sequence presented as what happened.
6. Dropping a caveat or an unexplained result because it breaks the arc (M1, M2). Loose ends go into chapter 7.
7. One vivid failure or success presented as typical.
8. Stakes beyond the source ("this changes everything").

**Pre-recording check.** Tag every story sentence as *event*, *number*, *reaction* or *framing*. The first three need a source-map entry; framing must read as our question or our structure, not as a fact about anyone.

---

## 5. Worked example: RoboBuilder in this structure

Illustration only; the actual episode may tell a different story. Every fact below is from `episodes/2026-11-22-weekly/research.md` or `episodes/2026-11-22-deepdive/audit.md` (arXiv 2611.23855 v1, preprint). **[check]** marks what must be confirmed in the paper before scripting.

**Central question (cold open):** fifteen robots passed in simulation and none crossed the real course. Why, and what did it take to make simulator-approved robots work for real?

| # | Story beat | Facts used | [check] |
|---|---|---|---|
| 0 | **Anomaly.** A real failure beside its simulated twin. "Fifteen designs passed the simulator. On the real course: zero, in 44 tries." Promise: what fixed most of it, and what still breaks | 0 of 15 initial sim passes crossed (44 tries), under a simulator built from public part data; Fig. 6 pairs real failure frames with simulation | Whether any clip shows one of the *initial 15*; the loop-failure clips come from the later phase, so label honestly or use the Fig. 6 frame. Clip speed (only one clip is marked "half speed") |
| 1 | **Goal and stakes.** MIT MechE team (Nehme, Nam, Srinivasan, Ahmed; DeCoDE Lab) asks four coding agents to co-design a robot's hardware and program. Stake, in the authors' framing: a first step toward agents that turn requirements into working prototypes | Agents: Opus 5.5 and Sonnet 5.5 (Claude Code), GPT-6 Astra and GPT-6.1 Sol (Codex), medium effort. Task: reach the goal within a time limit and stay upright 5 s. Parts: brick-compatible parts, motors, one agent-written CadQuery part. Terms to define: open-loop timed motor program; "simulated pass" | Time limits per course; size of the parts inventory |
| 2 | **The wall (try-fail #1).** The obvious approach: design and iterate in MuJoCo, under a simulation budget, with a simulator from public data. Prediction prompt: how many sim passes cross for real? Reveal 0 of 15. Clues, in the paper's words: tyres slipped, joints pulled apart, gears walked out of mesh, loads shifted. Refuted belief, from the paper itself: "Simulation alone is not enough". Lineage: the paper names Chen et al. 2026 as reporting a sim-to-real gap too | research.md quotes; audit lineage list | The simulation budget; how the paper describes Chen et al. |
| 3 | **Therefore #1: the agent learns the simulator.** Opus 5.5 requests bench measurements (the bench clips), fits parameters and adds new effects (joint release, gear slip, supply sag, chain skip) to training videos, validated on held-out runs. Every sim pass is then built by hand and filmed up to three times. **Test:** 66% of learned-sim passes cross (77% / 68% / 42% at levels 1 / 2 / 3). **But:** a third still fail, level 3 most | 135 robots, about 13,000 pieces, an estimated 240 h; a design passes if one try reaches the goal. Authors' caveat: some parameters are not identifiable | Tabs. 25–27 (held-out fit); whether the paper says the public-data simulator lacked these effects (needed before saying so) |
| 4 | **Therefore #2: the hardware loop.** Each level-1/2 real failure goes back to its agent as video. The agent may change at most two simulator parameters, add one, and write up to 150 lines; the change is kept only if the old design now fails in simulation. The redesign must pass a 3-of-5 perturbed vote, then is rebuilt. Prediction prompt: 18 failures; how many survive one round? **Test:** 5 (72% removed); real crossing of sim passes 73% → 93%, 100% for Sonnet 5.5 and GPT-6 Astra | Tab. 12 | That 73% is the level-1/2 baseline; how the vote perturbs the design |
| 5 | **The turn.** The authors' own diagnosis: the loop's gains come mainly from redesign after watching the failure; the simulator is corrected only sometimes. A viewer who credited the simulator fix learns that the failure video did most of the work | Sec. 5 statement; Tab. 14 classifies the 18 redesigns by response type | Tab. 14 categories and counts, before writing the insight sentence |
| 6 | **Verdict.** Back to the opening question, across the whole benchmark: 4 agents, 10 obstacles × 3 levels, 16,427 simulations of 12,774 distinct robots. Agents solve nearly all easy obstacles and invent mechanisms the search baselines don't reach, but falter at harder levels and when one robot must cross all ten (at most six). Fig. 2 wall of built robots; Fig. 4 design map | Abstract, project page | Per-agent and per-level tables; random search and GA numbers (Tab. 17); single-call baseline (Tab. 2); what "invent mechanisms" refers to (App. C) |
| 7 | **Still breaks.** Recovery is measured on the same course, not transfer to new designs; each design built once with few tries, so real verdicts are coarse; Light and Fast designs not built; each agent ran each task once; small, slow robots on open-loop programs; search baselines not the strongest possible; preprint, no independent replication or expert reaction found; it measures capability, not safety | research.md, audit (stated limitations) | Run-to-run variance (App. F.5) |
| 8 | **Use and the open question.** Code (MIT licence: benchmark, learned simulator, harness adapters, real-world kit) and data (CC BY 4.0, 461 paired real/sim videos). Bookend: Fig. 7, one loop case per agent (real failure → redesign in sim → redesign real) closes the cold-open image. Open question, in the authors' framing: a robot in the loop that builds and tests designs; sensors and feedback control co-designed with the mechanism | audit, Code/data and future work | That each Fig. 7 case is the same design and course as its failure |
| 9 | **Recap questions.** Why did the first fifteen fail? What two steps closed most of the gap, and which one did most of the work? Where does the result not yet apply? | — | — |

**Not available in the notes, so not used:** a documented red herring (no hypothesis the team held and dropped is recorded), any quote or reaction from the authors (no author thread or talk found), and any named expert reaction.

---

## 6. Weak or missing evidence

- **No test of story structure in long YouTube explainers, and none for experts.** N16 is correlational popularity; the learning studies use texts and students (N2–N6). The case for story on this channel rests more on order and attention than on a proven learning gain for engineers.
- **Every structural device is practitioner-only:** but/therefore, ABT, the spine, try-fail, the midpoint reversal, the bookend (F1–F6, C1–C5). They come from fiction, comedy and classroom teaching, and nobody has tested them against a plain explainer on retention.
- **Open loops:** the memory claim fails (N15); the "keeps people watching" claim is untested for video. Re-hooks remain practitioner-only (format doc P3).
- **Prior knowledge as a moderator** rests on one tentative study and one abstract (N4, N5). Whether story still helps engineers *on the paper's new parts*, where they are novices (teaching doc L2), is unknown.
- **The persuasion evidence (N7–N11) measures belief change.** It shows the risk; it does not show our viewers will be misled.
- **Read second-hand or abstract only:** Graesser's originals (via N1); Braddock & Dillard (403, numbers from summaries); Arya & Maul and the Tobler 2024a meta-analysis (via later papers); Hoeken & van Vliet, Wolfe & Woodwyk, van Laer, Freling, Ghibellini & Meier (abstracts); Pennington & Hastie (blog summary).
- **Not read:** Martinez-Conde et al. 2019 (*J Neurosci*, "The Storytelling Brain"); Joss Fong's Open Notebook interview (403); the Stratechery interview with Jon Yu (paywall); a Medium interview with Muller (403). Harris and Muller are quoted from auto-captions; Kurzgesagt from a machine transcript, and whether that video is still online was not checked.
- **Olson's Narrative Index** is his own measure with no independent validation found; Hillier et al. (N12) is the nearest real test on scientific abstracts, and its effect is small.
- **TTS narration:** no narrative study used a synthetic narrator.
- **"No intros" (C1) vs "signal an outline" (teaching doc L3):** our compromise, a roadmap spoken as 2–3 questions in about 10 s, is untested.
