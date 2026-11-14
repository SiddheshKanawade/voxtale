# Research 2026-11-14 (window: 2026-11-07 → 2026-11-14)

Sources: arXiv ✓ (350 papers, 350 on alphaXiv, 20 on HF Daily Papers), X ✓ (992 posts), News ✓ (149 + 1 scheduled pull; the
pull job started today, so news older than ~36 h came from web search), topics synced ✓ (6 profiles).
Credits: X OK (~1.5M), NewsData OK (561 today), ElevenLabs **LOW (3,683 chars, resets 2026-12-14)**, Render OK.
Triage: 8 subagent batches, 1,468 items read in full → 76 keep / 345 maybe / 1,047 drop. Web searches: ~10 (main) plus
the 4 verification agents' searches. Chatter files: `chatter/*.json` (X, HN, YouTube).

## Shortlist

| # | Story | Type | S | NV | A | C | Q | D | Primary source (date) | Key fact, verbatim | Visual | Caveat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | OpenAI folds its robotics team: general AI models now drive robots | strategy + research | 79.0 | 0.80 | 0.89 | 0.65 | 1.0 | 0.94 | [Fortune](https://fortune.com/2026/11/13/openai-disbands-robotics-team-as-its-general-models-start-running-robots/) (Nov 13) + arXiv papers (Nov 6–12) | "The company confirmed the reorganization when asked about it by Fortune." | Fortune headline; Embodied First Exam figures (CC BY); AMP1 video page | Reassigned, not laid off. The best general agent manages only 34% on the 456 core tasks of Embodied First Exam |
| 2 | Tesla opens personal Cybercab orders | release | 78.8 | 0.80 | 1.00 | 0.86 | 0.7 | 0.94 | [@robotaxi](https://x.com/robotaxi/status/2121384435241402737) (Nov 13) + [Electrek](https://electrek.co/2026/11/13/tesla-cybercab-personal-use-orders-299-subscription-geofence/) | "Select customers are now being invited to purchase Cybercab for personal use" | Order-page image in Tesla's post; 24/7 video | Invite-only, $1,000 non-refundable deposit. The $299/mo and the geofence come from Electrek's screenshots. NHTSA's ruling on the missing brake pedal is pending |
| 3 | Gemini Robotics 2.5: Apollo 2 at 96% on Mercedes lines | release (pilot) | 71.2 | 0.80 | 0.99 | 1.00 | 1.0 | 0.76 | [DeepMind blog](https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/) (Nov 10) | "our models helped Apptronik’s Apollo 2 humanoid pick and pack automotive parts with a 96% success rate." | Mercedes kitting video, 3×3 grids | Proof of concept. No trial count. Partners only. The 99.5% jailbreak figure belongs to Gemini 4 Argon, not GR 2.5 |
| 4 | SuperMimic: humanoid motion skills in 3 GPU-hours | paper | 67.3 | 0.60 | 1.00 | 0.70 | 1.0 | 0.82 | [arXiv 2610.28480](https://arxiv.org/abs/2610.28480) v2 (Nov 9); award: [author post](https://x.com/qiayuanliao) (Nov 12) | "In 3 hours on a single GPU, SuperMimic surpasses the return of a state-of-the-art motion tracker baseline trained for 7 days on 128 GPUs." | Project-page videos (G1 flips, rolls) | Paper v1 is from Oct 20. The award is confirmed only by the author's post (CoRL's page lists it as a finalist). GPUs differ (H100 vs L40). The arXiv license is non-exclusive, so don't reuse figures |
| 5 | Runway Praxis-1 goes open weights | release | 63.7 | 0.53 | 0.95 | 0.65 | 1.0 | 0.82 | [Runway](https://runway.com/news/praxis-1-is-now-available) (Nov 11) | "Praxis-1, our state-of-the-art world action model, is now available to everyone with open weights under the Apache 2.0 license." | Runway video; [NVIDIA RoboLab leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html) | The model was announced Sept 30; only the open weights are new. 46.2% is leaderboard-verified but ranks #5 overall. 80.0% on SimplerEnv is Runway-only |

**Why this order:** strictly by S. Constraints check:
- No company has more than 2 stories.
- Research stories: #1 is research-heavy and #4 is a paper.
- Variety: strategy, autonomous vehicles, factory humanoid, research, open models.
- Off the mainstream front page: #4 and #5.
- **Close call:** #6–#8 (GXO 62.6, Agile Robots 62.1, GR00T 2 61.3) are within 2.5 points of #5. GR00T 2 lost mainly on freshness (5 days) and because its weights aren't out yet. GR00T 2 and Praxis-1 share a theme ("robot brains trained on human video"), so they could become one story if the user prefers.

### 1. OpenAI folds its robotics team: general models now drive robots
- **Summary:** OpenAI moved its robotics researchers into its main research team, saying GPT-6 Astra-class models "can be directly applied to robotics". This week's papers back the bet, put limits on it, and flag safety risks.
- **How we'll explain it:**
  - Analogy: a brilliant general contractor who has never held a wrench, hired instead of specialist tradesmen. He can read any blueprint, but he's slow to react and needs a site foreman. And anyone who slips a fake work order into his inbox can steer the saw.
  - The evidence beat uses three numbers:
    - **Apple AMP1** ([2611.09892](https://arxiv.org/abs/2611.09892)): "The agent succeeds zero-shot in 104 of 108 real-world trials spanning twelve tasks and three robot instances". Apple's own numbers, and the brain was Claude Opus 5.5.
    - **MIT Reins** ([2611.16632](https://arxiv.org/abs/2611.16632)): "GPT-6.1 Sol succeeds in 88% of real episodes on five tasks". The license is NC, so capture the page only.
    - **Embodied First Exam** ([2611.12177](https://arxiv.org/abs/2611.12177), CC BY): "On the 456 core tasks, the strongest of eight agents, GPT-6 Astra in Codex, achieves a success rate of 34% in the Standard Setting".
- **Reality-check beat:**
  - **Safety:**
    - [RoboRisk/LEASH](https://arxiv.org/abs/2611.14115): "These models generally show high harm rates."
    - [ArmedAgent](https://arxiv.org/abs/2611.11438): prompt injection "can carry over into the physical world".
  - **Benchmark caveat:** [kNN paper](https://arxiv.org/abs/2611.13160): "retrieving nearby demonstrations and reusing their action chunks achieves a success rate of 91.7 % on LIBERO without policy training."
- **What people say:**
  - HN thread at 174 points / 72 comments. Consensus: "the Bitter Lesson comes for everything".
  - Pushback: jacquesm, "General models can run robots. They are very bad at it." Cloud latency is a theme ("getting a 429 rate limit halfway down the stairs").
  - In Fortune: Guanya Shi (CMU) says small on-board models are still needed. Russ Tedrake says deployment data is still a moat.
  - YouTube: essentially uncovered (6 videos, under 20 views each). Open lane.
- **URLs:** Fortune; arXiv 2611.09892, 2611.16632, 2611.12177, 2611.14115, 2611.11438, 2611.13160; HN item 50487184; `chatter/openai-robotics.json`, `chatter/embodied-first-exam.json`, `chatter/vla-knn.json`.
- **Highlight:** "The company confirmed the reorganization when asked about it by Fortune." (Fortune is paywalled; check that capture works.)

### 2. Tesla opens personal Cybercab orders
- **Summary:** Tesla started inviting select customers to buy a Cybercab for personal use: $29,990, delivery estimated for December in Austin. It needs a $299/month subscription and only works inside robotaxi zones.
- **How we'll explain it:** a phone that only works on one carrier's towers in a few cities, and only while you pay the plan.
- **Context beat (same week):**
  - Robotaxi "now runs 24/7" in every city it serves ([@robotaxi](https://x.com/robotaxi/status/2120736143566344324), Nov 12). The Bay Area is still supervised.
  - Texas registrations: "1,302 vehicles registered to its Robotaxi service against Waymo's 1,297". These are registrations, not cars in service ([EVwire](https://evwire.com/p/tesla-overtakes-waymo-texas-robotaxi-registrations)).
- **Reality check:**
  - Electrek's Fred Lambert: the subscription adds "$3,588 a year", and usable zones are only 59–78% of the advertised geofences.
  - NHTSA's ruling on the missing brake pedal is pending.
- **What people say:**
  - Order post: 24.0K likes, 5.43M views. Meet Kevin's video: 105K views.
  - The mainstream reading is "under $30K, but there's a catch".
  - Don't use the "Musk: monthly fee too high" quote: no source.
- **URLs:** Tesla posts; Electrek; Not a Tesla App; EVwire; `chatter/cybercab-personal.json`.
- **Highlight:** Electrek, "The order page lists the Cybercab at $29,990 before taxes, with an estimated delivery of December 2026 in Austin, Texas."

### 3. Gemini Robotics 2.5 in a Mercedes-Benz pilot
- **Summary:** Google DeepMind's Gemini Robotics 2.5 helped Apptronik's Apollo 2 humanoid sort metal brackets into kitting trays on live Mercedes-Benz lines in Hungary and Germany, at a 96% success rate. Proof of concept.
- **How we'll explain it:** an intern laying out a surgeon's tray in the right order, getting 24 of 25 right. Then: is that good enough for a production line?
- **Lab numbers:** "30% to 100% higher success rate depending on the type of task", across 200 tasks (vendor-reported).
- **What people say:**
  - Sundar Pichai's roundup: 12.7K likes. @GoogleDeepMind: 4.6K likes.
  - Pushback: all company-reported; no architecture or size details (HN). An Ars Technica argument that "96% isn't enough" could not be fetched, so it's unverified.
  - YouTube: thin (2.4K views max).
- **URLs:** DeepMind blog; @GoogleDeepMind and @carolina_parada posts; `chatter/gemini-robotics-2-5.json`.
- **Highlight:** "During a real-world proof-of-concept on Mercedes-Benz production lines, our models helped Apptronik’s Apollo 2 humanoid pick and pack automotive parts with a 96% success rate."

### 4. SuperMimic (UC Berkeley)
- **Summary:** UC Berkeley's SuperMimic teaches a humanoid to imitate human motions with one controller. In 3 hours on one GPU it beat a baseline trained for 7 days on 128 GPUs. It fails 25 of 129,514 motions. It won CoRL 2026 Best Paper, per its lead author.
- **How we'll explain it:** instead of 128 students each memorising their own dance, one student learns how a body moves and picks up any dance fast.
- **Verbatim:** "It reduces GPU-hours by more than $1000\times$ and fails on only 25 of 129,514 benchmark motions, versus 179 failures for the baseline." (In narration: "more than a thousand times".)
- **What people say:**
  - The author's launch post got 1,840 likes; the award post 1,206. The lab's YouTube video has 12.3K views.
  - Follow-ups are already chasing it: Caltech MorphMimic; XMimic claims "85x faster" on a different benchmark.
- **URLs:** arXiv 2610.28480; supermimic.github.io; `chatter/supermimic.json`.
- **Highlight:** the abstract sentence above (arXiv abs page).

### 5. Runway Praxis-1 open weights
- **Summary:** Runway, the AI video company, released Praxis-1, a 27B world action model, under Apache 2.0. On NVIDIA's independent RoboLab-120 leaderboard it scores 46.2%, the top open and top world action model there.
- **How we'll explain it:** a cook who learned from thousands of cooking videos needs only a short apprenticeship in your kitchen. Apache 2.0 is a free recipe book anyone can use commercially.
- **Verbatim:** "On RoboLab-120, Praxis-1 reaches 46.2% success across 1,200 episodes, compared with 42.9% for FLUX 3 Action, 36.8% for Cosmos3 Nano and 28.0% for π0.5."
- **What people say:**
  - Quiet: @LeRobotHF got 184 likes.
  - Praised for a "real Apache 2.0" license. Pushback: "Self-reported and narrow"; the leaderboard check partly answers that.
- **URLs:** Runway news + research page; HF model card; GitHub runwayml/praxis-1-oss; `chatter/runway-praxis-1.json`.
- **Highlight:** the Apache 2.0 sentence.

## Reserves
- **GXO–Apptronik LOI, up to 1,000 Apollo humanoids (62.6):**
  - [GXO, Nov 12](https://gxo.com/news_article/gxo-apptronik-expand-humanoid-robotics-partnership/). It's a conditional LOI, with "Subject to the success of the deployments…" in the release.
  - Very low social engagement. A good 20-second add-on to story 3 (the Apollo connection).
- **Agile Robots $500M at $8.5B (62.1):**
  - [company release, Nov 12](https://www.agile-robots.com/en/news/detail/physical-ai-leader-agile-robots-raises-500-million-series-f-funding/); Path Robotics $300M at $2.45B ([Nov 10](https://www.path-robotics.com/news/path-robotics-announces-300-million-series-e-led-by-georgian)).
  - Funding ranks low on news value. Could be a "money" montage line.
- **NVIDIA GR00T 2 (61.3):**
  - [GEAR page](https://research.nvidia.com/labs/gear/gr00t-2/), [blog Nov 9](https://blogs.nvidia.com/blog/robotics-simulation-advancements-corl/): "Five demonstrations per task, under 3 minutes of robot data, raise that to 95% after a 30-minute fine-tune".
  - Weights are "Code Available in December". It would pair with Praxis-1.

## Snippets (unverified)
- An Ars Technica argument (via search snippets) that 96% falls short of human kitting accuracy and that the model runs in the cloud. Not fetched.
- Brad Templeton (Forbes, Nov 13), slug "…coming soon to invitees, $20/hr". The page returned 403.
- Emulate (DeepMind spinout) is "reportedly in talks" for up to $700M at about $3.7–4B, with NVIDIA and Unity reportedly joining (Sifted, Nov 11; page 403). Not closed.
- Embodied First Exam lead Manling Li (Nov 12): "Fable 5.2 gets a ~13% improvement". The report is unpublished.

## Dropped (worth knowing)
- **"Jensen Huang resigned / Kress interim":**
  - A big cluster on X, dropped. No primary source; Nvidia told Business Insider Huang "is committed to continuing to lead Nvidia as CEO".
  - It's succession speculation tied to a board investigation, and off-anchor anyway.
- **NHTSA recall query RQ26006 into 2,895 Waymo vehicles (44.0):**
  - Opened 11/06, before the window. Strong primary source ([ODI resume](https://static.nhtsa.gov/odi/inv/2026/INOA-RQ26006-19477.pdf)).
  - Score cut by freshness and low attention. Keep it for next week if Waymo responds.
- **VLA kNN + "What are we benchmarking" (43.0):**
  - Folded into story 1 as the reality check rather than standing alone.
- **MuJoCo 4.0 (42.6):** released Nov 6, announced Nov 7. Matters only to specialists.
- **India MeitY and head-cam training data (41.8):**
  - Interesting new angle (gig workers as robot-data camera crews), with zero social attention. Sources are anonymous, via ET.
  - Candidate for a Short.
- **Agility Digit expansion at GXO:** announced Oct 26, outside the window.
- **Off-anchor big-lab news:** the Anthropic IPO, Gemini 4 Carbon and the Claude 5.2 releases.

## Notes for next run
- The scheduled news pull starts tonight, so next week's harvest will have a full week of NewsData.
- ElevenLabs has 3,683 characters left. A 5-story episode needs about 6,000, so voicing this episode needs a top-up or the reset on Dec 14.
