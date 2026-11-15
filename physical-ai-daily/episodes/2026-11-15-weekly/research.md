# Research 2026-11-15 (window: 2026-11-08 → 2026-11-15)

Supersedes `episodes/2026-11-14-weekly/research.md` (same week, run before web discovery and the judge panel).

Sources: arXiv ✓ (335 papers), X ✓ (901 posts; one batch of 10 accounts failed with a connection reset), News ✓ (266
saved items from 2 scheduled pulls, Nov 14 17:52 → Nov 15 02:30; the schedule was installed Nov 14, so earlier days
rely on web discovery), web discovery ✓ (79 events from 4 beats), topics synced ✓, credits: X OK (1.53M), NewsData OK
(582), ElevenLabs LOW (3,683 chars), triage batches: 8 (1,400 items: 54 keep, 468 maybe, 878 drop), judges: Opus,
Sonnet, Fable (+ Opus pairwise).

**Degraded:** the session's ~200 web searches ran out during web discovery. The global beat skipped Xiaomi, EngineAI,
Kepler, Fourier and Neura; the models beat skipped Physical Intelligence and Shanghai AI Lab. Verification used page
fetches only.

## Shortlist

Panel = mean overall score (1–10) from three judges on different models; the per-criterion columns are means.
Places 4–7 were settled pairwise (each pair asked in both orders; every win held both ways).

| # | Story | Type | Panel (A/B/C) | Sig | Pull | Evid | Show | Mom | Primary source (date) | Key fact, verbatim | Visual | Caveat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Gemini Robotics 2.5 at Mercedes-Benz | deployment | 9.33 (9/10/9) | 8.3 | 8.0 | 7.0 | 9.0 | 8.0 | [DeepMind blog](https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/) (Nov 10) | "96% success rate" (Apollo 2 kitting, proof of concept) | Official factory footage | Pilot; trial counts not disclosed; self-reported |
| 2 | Tesla sells Cybercab to individuals; Robotaxi 24/7 | release | 8.33 (8/9/8) | 7.7 | 10.0 | 5.3 | 8.0 | 9.3 | [InsideEVs](https://insideevs.com/news/815473/tesla-cybercab-pricing-sales-announced/) reporting Tesla's X post (Nov 13) | "$29,990" + required "$299 per month" | Tesla's Cybercab video | Secondary sources only; "supported service areas"; NHTSA questions |
| 3 | General AI models start driving robots (OpenAI folds robotics team; Claude Fable 5.2) | release | 8.0 (9/8/7) | 7.7 | 8.3 | 6.0 | 6.3 | 8.0 | [Fortune](https://fortune.com/2026/11/13/openai-disbands-robotics-team-as-its-general-models-start-running-robots/) (Nov 13); [Anthropic](https://www.anthropic.com/claude-fable-and-mythos-5-2) (Nov 9) | "remains a core part of OpenAI's long-term vision"; RoboDojo Sim 64.9% vs 29.8% | Anthropic robot video; LEASH figures | Single-outlet exclusive; vendor-run sim scores |
| 4 | NVIDIA GR00T 2 | release | 7.0 (8/7/6), 3 pairwise wins | 7.7 | 6.3 | 6.0 | 7.3 | 7.0 | [Project page](https://research.nvidia.com/labs/gear/gr00t-2/) (Nov 9) | "succeeds in 40% of trials" zero-shot; five demos "lift it to 95%" | Uncut 1x lab demo | Self-reported; tasks overlap training data; code in December |
| 5 | SuperMimic (UC Berkeley) | paper | 6.0 (7/5/6), 2 pairwise wins | 6.3 | 5.3 | 5.7 | 8.7 | 6.7 | [Paper](https://www.alphaxiv.org/abs/2610.28480), [project](https://supermimic.github.io) | "1 GPU · 3 h to surpass SONIC's return" vs "7 days on 128 GPUs" | Real G1 flips, web demo | Best Paper only per the author; unmatched compute comparison; paper from October |

**Editorial rules applied after the panel:** none moved a story. No company has more than 2; there is a research story
(SuperMimic); the topics vary (factory, consumer car, AI labs, open model, research); GR00T 2 and SuperMimic aren't
front-page news.

### 1. Gemini Robotics 2.5: humanoids sort car parts on Mercedes-Benz lines

- **Summary (27 words):** Google DeepMind's new robot brain, Gemini Robotics 2.5, ran Apptronik Apollo 2 humanoids that
  sorted car parts on live Mercedes-Benz production lines with a 96% success rate.
- **Facts:** proof of concept on active lines in Kecskemét (Hungary) and Sindelfingen (Germany). Versus the previous
  model: 30%–100% higher success depending on task type, 20%–30% faster, tested on 200 distinct tasks; packing pastries
  with tongs went from 5% to 100%. New open ASIMOV-Jailbreak benchmark (jailbreak resistance 99.5% for Gemini 4 Argon vs
  82%). Rolling out to robotics partners, not generally available.
- **Explain it:** a VLA is one model that sees (vision), understands an instruction (language) and moves (action).
  Analogy: a new hire who's read the manual and watched hundreds of colleagues, then gets put on the real line.
- **What people say:** DeepMind's launch post got 4,645 likes, Google's 1,911. A DeepMind researcher says Gemini 4
  Argon "strongly accelerated" building it (AI helping build AI). Pushback: it's a pilot with no trial counts, and a top
  YouTube comment predicts German unions will resist. YouTube explainers are small (≈1–2.5k views), but DeepMind's
  earlier Gemini Robotics 2 video has 6M, so the audience is there.
- **Judges:** all three ranked it first. "The clearest 'what changed' story of the week" (A); "first credible
  factory-line number for a general robot model" (B, C).
- **URLs:** DeepMind blog (above); GXO–Apptronik release (reserve 3) for the Apptronik context.
- **Highlight:** `--find "96% success rate"`.

### 2. Tesla starts selling Cybercab to individuals; Robotaxi goes 24/7

- **Summary (26 words):** Tesla is inviting select customers to buy Cybercab, its two-seat robot car with no steering
  wheel, for $29,990 plus a required $299-a-month autonomy subscription.
- **Facts:** first month free; operates only in "supported service areas"; deliveries "as soon as next month" (the
  Dec 20 date some outlets give isn't in the source). Same week: unsupervised Robotaxi launched in San Antonio on Nov 9
  (8th city, ~274–275 sq mi), and rides now run 24/7 in all eight unsupervised cities (not the Bay Area, which is still
  supervised).
- **Explain it:** you own the car, but it only drives itself where Tesla has mapped and approved; the subscription is
  the "driver".
- **What people say:** @robotaxi's post got 25,067 likes. Fans see a robot car you can own. Sceptics: $300 a month for
  life makes it over $30k, it only works in geofences, and on HN "if a Tesla Cybercab fleet were profitable, Tesla
  wouldn't sell you one". NHTSA is questioning the no-wheel, no-pedal design, and noncompliant deployments are capped at
  2,500 a year.
- **Judges:** highest viewer pull and momentum (10 and 9.3) but evidence 5.3: everything is secondary reporting of
  Tesla's X posts. **Before scripting, capture Tesla's own post or order page.**
- **URLs:** InsideEVs (above); [Not a Tesla App, 24/7](https://www.notateslaapp.com/news/4991/tesla-robotaxi-now-runs-247-in-every-city);
  [Teslarati, San Antonio](https://www.teslarati.com/tesla-robotaxi-enters-newest-city-san-antonio/).
- **Highlight:** `--find "$29,990"`.

### 3. General AI models start driving robots

- **Summary (28 words):** OpenAI folded its stand-alone robotics team into its main research and product groups, saying
  its general model can drive robots directly, while Anthropic's new Claude Fable 5.2 more than doubled a robot-control
  benchmark.
- **Facts:** per Fortune, robotics research moves under Chief Research Officer Mark Chen and applied/hardware under
  President Greg Brockman. Researchers saw "surprising performance" connecting GPT-6 Astra directly to robots. Fortune
  cites an Nvidia study: an Astra agent at 74.8% vs 51.0% for the best specialised robot models, in simulation. Fable
  5.2 scored 64.9% on RoboDojo Sim vs Opus 5.5 39.4%, Gemini 4 Argon 34.2%, GPT-6 Astra 29.8%, Fable 5.1 27.6%
  (Anthropic's internal runs).
- **Reality check (the story's second half):**
  - Embodied First Exam (Nov 9, 30 authors incl. Abbeel and Fei-Fei Li): the best coding agent, GPT-6 Astra, succeeds
    on 34% of 1,156 robot tasks.
  - LEASH (Nov 10): frontier models' harm rates controlling robots run 9.06%–76.92% in scenarios built to tempt unsafe
    actions, and a gating layer cuts most to near 0 (real-robot test: 13 tasks, one run each).
  - Apple's AMP1 (Nov 6, just before the window) has Claude Opus 5.5 win 104/108 real trials vs 30 for π0.5. But its
    authors say agentic policies are "not yet practical": about 211 s of model reasoning per success.
- **Explain it:** instead of a special robot brain trained on robot data, you hand the robot to a general AI that
  reasons, writes code and calls skills, like hiring a brilliant generalist instead of a trained specialist: capable, but
  slow to think.
- **What people say:**
  - "Robotics may be no longer a special problem" (Guanya Shi, in Fortune).
  - Pushback: cloud models are slow and need connectivity (Shi, Pete Florence); robotics companies keep a real-world
    data advantage (Russ Tedrake).
  - On X, Andrew Curran (2,131 likes) and Ethan Mollick amplified the Fable jump, and Artificial Analysis has Fable 5.2
    leading AA-Robotics at 75 vs 69.
- **Judges:** A ranked it 2nd ("big-name actors and a genuine shift"). B noted it's "a bit multi-part to explain".
  **Script it as one claim, general models drive robots, with OpenAI as the hook and the benchmarks as evidence.**
- **Note:** Anthropic's launch is ours (this research ran on Claude). Treat it like any vendor claim: say it's
  Anthropic's own runs.
- **URLs:** Fortune; Anthropic; [Embodied First Exam](https://arxiv.org/abs/2611.12177);
  [LEASH](https://arxiv.org/abs/2611.14115); [Apple AMP1](https://arxiv.org/abs/2611.09892).
- **Highlight:** Fortune `--find "remains a core part of OpenAI's long-term vision"`; Anthropic `--find "substantially improved skills in controlling physical robots"`.

### 4. NVIDIA GR00T 2: an open robot model that learns from watching people

- **Summary (25 words):** NVIDIA's GR00T 2 learned from 402,730 hours of video of people working, and with five
  demonstrations per task reached 95% success on a two-handed robot.
- **Facts:**
  - Zero-shot "succeeds in 40% of trials".
  - One demonstration gives 71%; five ("under 3 minutes of robot data") give 95%, on 50 tasks that appear in other
    robots' data or human video.
  - "40 of 50 tasks at least once" is a separate count; don't merge the two.
  - Pretraining also includes 23,500 hours of teleoperation. GR00T 2 Turbo runs on a Unitree G1 itself.
  - Code is "available in December", and Jim Fan's talk says weights then too. Nothing is released yet.
- **Explain it:** a "world action model" imagines the next few seconds of video, then picks the moves that make that
  future happen: daydream, then do.
- **What people say:**
  - @nvidia got 3,017 likes, @DrJimFan 1,911 and @yukez 1,474.
  - Post-Hype Tech calls the zero-shot claim overstated, because the tasks overlap the training data.
  - **Context line:** it launched three days before Jensen Huang resigned (see reserves). Two judges suggested folding
    the CEO news in here as one sentence.
- **URLs:** project page; [NVIDIA blog](https://blogs.nvidia.com/blog/robotics-simulation-advancements-corl/).
- **Highlight:** `--find "lift it to 95%"`.

### 5. SuperMimic: a humanoid's whole-body skills on one GPU in 3 hours

- **Summary (27 words):** UC Berkeley's SuperMimic taught a humanoid 129,514 motions inside a learned simulator on one
  GPU in three hours. The previous best took seven days on 128 GPUs.
- **Facts:**
  - Its policy failed on 25 of 129,514 benchmark motions after 3 h on 1 GPU, against 179 failures for SONIC's released
    checkpoint, and on none after 9.8 h on 2 GPUs.
  - It transfers to a real Unitree G1, including flips and a side roll, and was open-sourced at CoRL.
  - It's an oral at CoRL 2026. The **Best Paper award is claimed by the lead author** on X (1,209 likes), but CoRL's
    awards page lists no winners yet.
  - The compute comparison isn't matched: 1 H100 on a 264 h data subset vs 128 L40s on 611 h.
- **Explain it:** normal RL is learning to skateboard by falling thousands of times. SuperMimic learns a smooth,
  differentiable model of its own body, so it can calculate which way to adjust, like reading the slope instead of
  rolling down blind.
- **What people say:** researchers are impressed by the compute drop, and XMimic already claims faster on a smaller
  motion set. The lab's video has 12k views.
- **Judges:** highest showability (8.7) but the lowest viewer pull of the five (5.3). It won its pairwise matches against
  the Haiti and Huang stories. This is the research slot.
- **URLs:** paper, [project](https://supermimic.github.io).
- **Highlight:** `--find "1 GPU · 3 h"`. Say "won CoRL's best paper award, according to its lead author" or "a
  headline paper at CoRL" until CoRL posts it.

## Reserves

1. **Armed robot dogs in Haiti (panel 6.67, viewer pull 9, evidence 3.7).** El Salvador's forces used "four-legged
   headless robots, with guns mounted upside down on their backs" to help retake Port-au-Prince's national stadium. It
   lost every pairwise match except against Huang, because the only source is Bukele's promotional video: no model, no
   date, no role confirmed. On-topic YouTube coverage is ~1.09M views. **Best Short candidate** if framed as "what we
   know and don't".
2. **Jensen Huang resigns (panel 5.33, judges split 5/4/7).**
   - The 8-K and release (Nov 12) say "Jensen Huang has resigned as President, Chief Executive Officer, and a member of
     the Board, effective immediately"; CFO Colette Kress is interim CEO.
   - It's the week's most-watched story (HN 4,043 points, CNN's video at 1.8M views).
   - It lost every pairwise match: nothing changes for robots this week, and there's no robot footage.
   - The spy link is alleged by NYT reporting, not confirmed. Use one line in story 4.
3. **Agile Robots $500M at $8.5B (5.33).** It has capacity for "up to 10,000 legged and wheeled humanoid robots
   annually", but no shipment numbers.
4. **Boston Dynamics' wheeled Atlas (5.33, showability 9.3).** B-roll for a humanoid segment.

## Snippets (unverified)

- Foundation is in preliminary talks to sell up to 1,000 Phantom MK1s to the IDF for tunnel work (FT via the Daily
  Caller). Israel hasn't confirmed.
- South Korea "440.2 billion won ($330 million)" for humanoid components and "80% localization" by 2030 (KED Global,
  via a beat agent). **Not in the cited Korea Herald article**, so don't use it.
- Joyson Electronics delivered a "robot brain backpack" compute module to Unitree (Nov 10). Snippet-level.
- HeymanDex WA100 underwater dexterous hand rated to 300 m (TechNode, Nov 13). Snippet-level.
- X-Humanoid's "Wise Kaiwu" platform and Hi-WBC controller (YouTube listing, Nov 10). Snippet-level.

## Dropped (worth knowing)

- **Outside the window:**
  - Caltech's ASTRA built one truss bay of an antenna in orbit, "the first time that a structure has been autonomously
    constructed in space" (release Nov 5; widely re-reported Nov 8–13).
  - Apple AMP1 (Nov 6) and RoboColosseum (Nov 2), now evidence inside story 3.
  - Also outside: MuJoCo 4.0 (Nov 6), LeRobot 0.7 (Nov 5), Agility's Nasdaq debut (Nov 5), China blocking SoftBank's
    ABB Robotics deal (Nov 5), Waymo's NHTSA recall query (Nov 6).
- **Judged, not picked** (panel 1.7–5.0): LingBot-Touch, Runway Praxis-1, Q3 humanoid shipments (16,500, +347%,
  ~97% Chinese vendors; one firm's estimate), GXO–Apptronik LOI for up to 1,000 Apollos, Path Robotics $300M,
  Foundation's missed payroll, ASUS Gallop, the D.C. robotaxi bill (failed 3-3), government plans (Beijing's 2028
  targets), India's MeitY consultation on in-home worker video.
- **Corrections caught by verification:**
  - D.C.'s vote was 3-3, not 3-2.
  - The Beijing plan has no production-capacity target.
  - "China 87%" is demand; Chinese vendors are ~97% of shipments.
  - MeitY held a consultation, not a probe.
  - GR00T 2's 40% and "40 of 50 tasks" are different metrics.
  - Path Robotics' round was announced Nov 10.

## Notes for next run

- **Web discovery found what the harvest missed.** The Huang resignation was confirmed only through the beat agents and
  the 8-K. Yesterday's run had filed it as speculation, because the first viral posts came before the filing.
- **Budget searches:** give each beat agent 35 web searches (now in the skill).
- **YouTube views:** chatter's search returns off-topic videos (for example, "Praxis 1" is also a teacher exam). Count
  only on-topic views.
- **chatter.py:** it now skips whisper for non-English videos, after a Hindi video produced garbage.
