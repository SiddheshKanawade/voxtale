# Deep-dive format: which stories, and how to structure them (research, 2026-11-24)

**Purpose.** This doc supports a new series that runs independently of the weekly 3–5 story digest. Each episode is one
story, 12–20 minutes long, with full technical depth explained so it is easy to follow. The viewer is an engineer
working in robotics or AI. Each week the episode picks one story from that week's verified digest research
(`episodes/<week>/research.md`). The doc answers two questions:

1. Which stories can carry a deep dive?
2. How should a 12–20 minute episode be built?

The evidence ends in a DRAFT set of selection criteria and a DRAFT episode structure, for the user to validate.

**Date.** 2026-11-24. All channel data was pulled on this date.

**Method.**
1. **Channel pull (metadata only).** For 15 channels, yt-dlp read the `/videos` tab, which excludes Shorts:
   `--flat-playlist -J --playlist-end 50`, plus `--extractor-args youtubetab:approximate_date` for upload dates. Calls
   were paced 8 s apart.
2. **Per-video pull.** `--dump-json --skip-download` ran on 45 videos (each channel's top 2 outliers plus 1 median
   video) for exact dates and chapters, paced 3 s apart. No video, audio or subtitles were downloaded.
3. **Outlier ratio.** A video's views divided by its channel's median views. The median uses only videos at least 30
   days old.
   - Many channels' latest 50 uploads span years, and older videos have had longer to collect views.
   - So each video also gets a **local ratio**: its views divided by the median of its 14 nearest uploads. This
     controls roughly for age and channel growth.
4. **Topic coding.** One coder (this agent) coded each channel's top 3 outliers and its 3 videos nearest the median,
   from titles and chapter titles only.
5. **Literature.** YouTube Help and blog pages, peer-reviewed studies of science YouTube and video engagement,
   multimedia-learning reviews, and practitioner writing. Papers were read in full where possible; where only the
   abstract was read, the table says so.
6. **Not repeated here.** The repo docs were read first: `research-reach.md`, `packaging.md`, `attention.md`,
   `competitors.md`, `judge-evidence.md`, `research-tuning.md`, and the digest rubric
   (`.claude/skills/physical-ai-research/SKILL.md` §6).

**Strength labels.**

| Label | Meaning |
|---|---|
| OFF | YouTube's or Google's own statement |
| RCT / EXP | Randomized trial / experiment |
| LD | Large observational dataset |
| PEER | Peer-reviewed and correlational, with its n given |
| PRAC | Practitioner opinion, no data |
| DESC | Our own descriptive pull |

## Sources

| # | Claim | Source | Evidence type | Strength |
|---|---|---|---|---|
| S1 | "Intro" is the share of viewers still watching at 30 s; a high share suggests the opening matched the title and thumbnail. Audiences shrink as a video goes on, so move compelling moments earlier. Studio compares each video with your 10 latest of similar length, and with all of YouTube | [YouTube Help: key moments for audience retention](https://support.google.com/youtube/answer/9314415) | Official guidance | OFF, no data |
| S2 | Chapters need 3+ timestamps, the first at 00:00, each at least 10 s. Stated benefits: rewatching and context per section. The page makes no claim about discovery | [YouTube Help: Video chapters](https://support.google.com/youtube/answer/9884579) | Official guidance | OFF |
| S3 | "There's no universal “ideal” length for YouTube videos." Avoid filler, use retention curves. Viewers repeatedly stopping a channel's recommended videos hurts it | [YouTube Help: recommendations](https://support.google.com/youtube/answer/16559651?hl=en) | Official guidance | OFF |
| S4 | Google Search may create "key moments" from timestamps in a YouTube description | [Google Search Central: video](https://developers.google.com/search/docs/appearance/structured-data/video) | Official docs | OFF |
| S5 | Video essayists are "typically proving out a thesis", unlike a classic explainer. Video-essay videos drew 200M+ views in 2024 | [YouTube Culture & Trends, 2025-04-16](https://blog.youtube/culture-and-trends/the-joy-of-video-essays/) | Official, descriptive | OFF |
| S6 | 55.2% of sessions left before the end, and 36.6 points of that came in the first 3% of the video. Predicted dropout: 53% for a 5-min video vs 71% for a 20-min one. 61% of rewatch/play peaks fell on visual transitions | [Kim et al., L@S 2014](https://iis.seas.harvard.edu/papers/kim14video.pdf) (862 edX videos; edX autoplays) | Second-by-second logs | LD, peer-reviewed; MOOC, not YouTube |
| S7 | Median engagement time is at most 6 min regardless of video length. Planning before production had the largest impact. Fast, enthusiastic speakers held viewers better | [Guo, Kim & Rubin 2014](https://www.cs.rochester.edu/hci/pubs/pdfs/edX-MOOC-video-production-and-engagement_LAS-2014.pdf) (6.9M sessions) | Observational | LD; MOOC |
| S8 | Duration alone explains over 58% of the variance in average watch percentage. Watch percentage ranked within its duration bin ("relative engagement") stays stable within a channel. Engagement can be predicted from context, topic and channel (R²=0.77) | [Wu, Rizoiu & Xie, ICWSM 2018](https://arxiv.org/abs/1709.02541) (5.3M videos) | Observational | LD |
| S9 | In 306 science videos, emotional arousal (OR 2.998) and being built on one central story (OR 3.163) predicted being most-viewed. Moments of change and dramatic question mattered only bivariately. Length did not differ (p=0.44). Topic was not analysed. "Emotional arousal, in particular, showed a strong association with popularity." | [Huang & Grant 2020](https://www.frontiersin.org/articles/10.3389/fcomm.2020.581349/full) | Extreme-case sample, coded | PEER, n=306, correlational |
| S10 | A regular communicator went with more views (d=1.35, user-made channels), and so did pace (r=0.21–0.25). No correlation with length (r=−0.03), but the videos were short (median 196 s for professional channels). **Topic was not analysed** | [Welbourne & Grant 2016](https://openresearch-repository.anu.edu.au/bitstream/1885/107245/2/01%20Dustin%20and%20Grant%20Science%20communication%20on%202016.pdf) | Coded sample | PEER, n=390 |
| S11 | Shorter videos were more likely to be viewed. Likes went with every type of engagement | [Yang et al. 2022, PLOS ONE](https://doi.org/10.1371/journal.pone.0267697) (abstract only) | One channel (ACS Reactions) | PEER, narrow |
| S12 | Only 32 of the 391 most popular STEM channels had female hosts. Female-hosted channels got more comments per view, with more hostile and sexist ones. **Says nothing about topic choice** | [Amarasekara & Grant 2019](https://doi.org/10.1177/0963662518786654) (abstract only) | Sentiment analysis | PEER |
| S13 | In 622 French channels (70,795 videos), individual creators usually focus on one or two fields. Institutions cover more subjects and get fewer subscribers and less engagement | [Debove et al. 2021](https://doi.org/10.3389/fcomm.2021.612667) (abstract only) | Dataset + survey | PEER, LD |
| S14 | Stating and then refuting common misconceptions raised learning (d=0.79; dialogue form d=0.83) vs a plain exposition. Low-prior-knowledge students gained most; high-prior students were not disadvantaged | [Muller et al. 2008, JCAL](https://doi.org/10.1111/j.1365-2729.2007.00248.x) (abstract via Crossref) | Randomized, n=364 | RCT |
| S15 | A video that explained force through misconceptions gave the same feeling of understanding as a correct one, but more misconceptions | [Kulgemeyer & Wittwer 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC8932681/) (abstract via Europe PMC; PMC page was bot-checked) | Experiment, N=149 | EXP |
| S16 | Effect sizes for multimedia design: segmenting d=0.70, signalling 0.46, pre-training 0.46, coherence 0.70. Some principles are stronger for low-knowledge learners | [Mayer 2017, JCAL](https://doi.org/10.1111/jcal.12197) (abstract) | Review of experiments | Strong for learning; lab settings |
| S17 | Expertise reversal: guidance that helps novices can lose its value, or hurt, for experienced learners | [Kalyuga 2007](https://doi.org/10.1007/s10648-007-9054-3) | Review | Strong in learning research; not tested on YouTube |
| S18 | "Topic choice matters way more than production quality." Give examples before general frameworks; treat definitions as an end point, not a start; be niche early | [3Blue1Brown FAQ](https://www.3blue1brown.com/faq/) | Creator advice | PRAC |
| S19 | The first minute must show that the thumbnail's promise will be kept; minutes 1–3, 3–6 and 6–end each have defined jobs | [MrBeast production doc via Simon Willison, 2024](https://simonwillison.net/2024/Sep/15/how-to-succeed-in-mrbeast-production/) | Leaked internal doc; the blog paraphrases it | PRAC; authenticity unconfirmed; entertainment, not science |
| S20 | Channel outliers, lengths and chapters (section 1) | yt-dlp pull, 2026-11-24 (15 channels, 660 long videos; 45 per-video) | Descriptive | DESC |
| S21 | Curiosity peaks at intermediate knowledge (Kang 2009). Practical value predicts sharing among viewers (Kim 2015). Videos keep earning views for months (Borghol 2012; Beaupré 2026) | `docs/research-reach.md`, `docs/packaging.md` | Secondhand via repo | As rated there |

## 1. Channel outliers (pulled 2026-11-24)

**Table 1. Channel summary.**
- Long uploads only; n = videos with views and length.
- "≥2×" counts videos at least 30 days old with ratio ≥2.
- ρ is the Spearman correlation of length and views within the channel.
- "Span" is the age of the oldest video pulled, from approximate dates.
- View counts in flat listings are rounded by YouTube ("2.5M views").

| Channel | Subs | n | Span | Median views | Median length | ≥2× | ρ length–views | Top outlier, exact upload date |
|---|---|---|---|---|---|---|---|---|
| Welch Labs | 915K | 50 | ~9 y | 216,000 | 13.2 min | 15 (31%) | 0.56 | 2023-07-16 |
| 3Blue1Brown | 8.78M | 45 | ~4 y | 2,300,000 | 27.2 | 10 (23%) | 0.15 | 2024-04-01 |
| Branch Education | 2.78M | 50 | ~8 y | 1,200,000 | 11.0 | 16 (33%) | 0.64 | 2024-05-17 |
| Real Engineering | 5.12M | 47 | ~3 y | 1,900,000 | 23.0 | 8 (18%) | 0.19 | 2023-11-04 |
| Asianometry | 971K | 50 | ~0.5 y | 209,000 | 23.9 | 4 (10%) | 0.07 | 2026-07-05 |
| AI Explained | 460K | 50 | ~1 y | 96,000 | 19.1 | 6 (13%) | 0.49 | 2026-09-04 |
| bycloud | 239K | 50 | ~1 y | 75,000 | 16.0 | 9 (20%) | 0.38 | 2026-09-02 |
| Yannic Kilcher | 333K | 50 | ~2 y | 33,500 | 37.1 | 4 (8%) | 0.14 | 2025-01-26 |
| Steve Brunton | 567K | 50 | ~1 y | 17,500 | 16.9 | 8 (17%) | 0.38 | 2026-07-02 |
| Gonkee | 137K | 18 | ~5 y | 403,000 | 12.4 | 1 (6%) | −0.25 | 2021-10-10 |
| Two Minute Papers (contrast) | 1.84M | 50 | ~0.5 y | 131,000 | 5.9 | 8 (20%) | −0.01 | 2026-09-08 |
| Aaed Musa (robot builds) | 277K | 39 | ~7 y | 21,000 | 8.0 | 17 (44%) | 0.73 | 2024-05-31 |
| Articulated Robotics (ROS) | 79.8K | 50 | ~4 y | 34,000 | 16.7 | 14 (28%) | 0.64 | 2022-05-19 |
| Kevin Wood (robotics) | 82.8K | 46 | ~0.5 y | 5,000 | 8.4 | 10 (25%) | 0.44 | 2026-07-29 |
| Ilia (VLA explainers) | 8.5K | 15 | ~2 y | 10,000 | 66.0 | 4 (27%) | 0.00 | 2025-03-27 |

The robotics channels were found by four YouTube searches, e.g. "robot learning explained VLA diffusion policy" and
"robot actuator design deep dive". PRO ROBOTS and AI Revolution are already profiled in `competitors.md`.

**Table 2. Top 3 outliers (ratio / local ratio, length) and each channel's typical videos (ratio ≈ 1×).**

Type codes (coded from titles and chapter titles):

| Code | Type |
|---|---|
| M | Mechanism reveal |
| N | Famous name or thing in the title |
| S | Surprising result |
| F | Failure or problem |
| T | Practical tool, tutorial or build |
| O | Foundational overview |
| C | Controversy or risk |
| H | History |
| R | Release analysis |

| Channel | Top 3 outliers | Types | Typical videos (≈1×) |
|---|---|---|---|
| Welch Labs | Oppenheimer's Apocalypse Math 19.0×/19.6×, 6 min; The Most Dangerous Rock in the World 15.3×/50.0×, 13; AI can't cross this line and we don't know why 11.6×/4.5×, 24 | H+N, H, S+M | Inside the World's Smartest Robot Brain [VLA] 1.00×, 35 min; Subliminal Learning; Backpropagation Part 2 |
| 3Blue1Brown | Transformers 4.8×/5.2×, 27; Solving Wordle using information theory 4.8×/2.8×, 31; LLMs explained briefly 3.8×/2.9×, 8 | M+N ×3 | Escher's print gallery; How might LLMs store facts; Laplace Transform |
| Branch Education | How are Microchips Made? 10.0×/4.1×, 28; How does Bluetooth Work? 8.3×/10.4×, 22; How does Starlink…Work? 8.2×/2.7×, 28 | M+N ×3 | Noise-canceling headphones; wireless earbuds; smoke detectors |
| Real Engineering | The Insane Engineering of the F-16 4.0×/4.9×, 41; …of the F-117 3.1×/2.6×, 28; The Problem with Wind Energy 2.6×/2.1×, 17 | M+N, M+N, F | The Questionable Engineering of the 737 Max; Golden Dome; Why America's Tank Failed in Ukraine |
| Asianometry | The Debt-Fueled Collapse of China's Top Machine Tool Maker 2.6×/2.9×, 37; Sweden Brought Back DC Transmission 2.5×/3.2×, 26; China's Fiery Incinerator Boom 2.1×/1.9×, 28 | F+H, S+H, H | IGBTs; The Dark Fiber Listening Network; The EU Chips Act is a Failure |
| AI Explained | GPT 6 Astra, so good even OpenAI are worried 8.9×/5.9×, 29; Opus 5.5: How Close Are We to Automated AI Research? 3.2×/2.0×, 33; OpenAI Security: Controlling Models is Now 'Hell' 3.1×/1.9×, 38 | R+N+C, R+N, C+N | Two roundups ("AI Accelerates: New Gemini Model + AI Unemployment…", "Google Takes No Prisoners Amid Torrent of AI Announcements"); Claude 4 breakdown |
| bycloud | Robots Just Had Their GPT-3 Moment 9.9×/10.3×, 16; How Did DeepSeek Make V4 So Cheap? 4.5×/5.2×, 21; NVIDIA Made 4-Bit AI Work. It's a Nightmare. 4.2×/3.3×, 20 | S, M+N, M+N | The RL Irony in LLMs; The New AI Open Source Trifecta (roundup); Decision Models |
| Yannic Kilcher | [GRPO Explained] DeepSeekMath 5.3×/5.5×, 69; AGI is not coming! 4.8×/6.4×, 7; Flow Matching (Paper Explained) 3.4×/2.9×, 56 | M+N, C, M+O | Context Rot; Energy-Based Transformers; [ML News] roundup |
| Steve Brunton | Optimization: A Bootcamp… 6.2×/6.1×, 22; Maximum Likelihood Estimation 4.4×/4.8×, 24; Bayesian Inference: Overview 4.3×/5.6×, 30 | O ×3 | Bayesian Linear Regression; Bayesian Updates and Conjugate Priors; Bayesian MAP Estimation |
| Gonkee | I'm Coding an Entire Physics Engine from Scratch 4.2×/3.9×, 9 (the only one ≥2×) | T | AI learns to fly like Ironman; fluid simulations ×2 |
| Two Minute Papers | GPT-6 Astra Changes Everything 4.4×/3.3×, 5; I Never Thought I'd See This Happen 3.0×/2.1×, 9; Claude Opus 5.5…Leap Forward 2.9×/1.4×, 5 | R+N, S, R+N | Single-paper "just changed…" episodes |
| Aaed Musa | High Precision Speed Reducer Using Rope 142.9×/8.8×, 20; I Built an Internal Cycloidal Robotic Actuator 61.9×/3.9×, 19; I Built a Ball Balancing Robot 57.1×/19.0×, 9 | T+S, T+M, T | Fusion 360 gear tutorial; "Ball Balancing Robot pt.2"; actuator part-video |
| Articulated Robotics | The Ultimate Guide to using Motors in Robotics 15.8×/5.8×, 26; If you're not developing with this… (dev containers) 10.2×/18.3×, 14; Easy SLAM with ROS 8.5×/6.6×, 26 | O+T, T, T | ROS release update; phone control; Gazebo upgrade |
| Kevin Wood | Robotics Course on Kinematics & Dynamics 14.4×/27.2×, 138; Microduck RL policy training 9.6×/8.1×, 8; AGIBOT A3 Humanoid Robot Does Kung Fu… But That's Not the Impressive Part 7.6×/5.4×, 11 | O+T, T, N+S | Cosmos 3 on Jetson; a humanoid model demo; a robot-hand product |
| Ilia | Running AI robotics experiments at home with LeRobot 8.5×/8.2×, 40; What Are Vision-Language-Action Models? (VLA Series Ep.1) 7.5×/8.0×, 35; 3D-printable arm 3.9×/5.0×, 4 | T, O+M, T | VLA Ep.4 GR00T 1.0×; competition write-ups |

### What the outliers share (only what the data shows)

1. **One subject, never a roundup.**
   - None of the 45 top-3 outliers covers several stories.
   - Four of the 45 near-median videos are roundups (two from AI Explained, one each from bycloud and Yannic).
   - This matches `packaging.md` finding 2.
2. **In mechanism channels, the outlier explains how something famous works.**
   - 3Blue1Brown (transformers, LLMs, Wordle), Branch (microchips, Bluetooth, Starlink) and Real Engineering (F-16,
     F-117): every video is a mechanism explainer. The outliers' subjects are names most viewers already know.
   - Their typical videos explain narrower subjects (the Laplace transform, smoke detectors, earbuds).
   - A famous name is not enough on its own: Real Engineering's 737 Max video sits at 1.00×.
3. **In AI-analysis channels, the outliers are the period's biggest single releases or surprising results, often
   framed by what is at stake.**
   - Examples: "GPT 6 Astra, so good even OpenAI are worried" (8.9×), and GRPO/DeepSeekMath (5.3×, uploaded
     2025-01-26).
   - bycloud's pattern is the mechanism behind a famous result: "How did DeepSeek make V4 so cheap?", "How did a 27M
     model even beat ChatGPT?" (about 4×).
4. **In teaching channels, broad foundations beat narrow follow-ups.**
   - Brunton's three overviews run 4.3–6.2×; his narrower Bayesian follow-ups sit at about 1.03×.
   - Ilia's VLA series decays from episode to episode: Ep.1 (what VLAs are) 7.5×, Ep.2 (pi0) 3.0×, Ep.3 (pi0.5)
     1.3×, Ep.4 (GR00T) 1.0×.
   - Complete guides and courses top Articulated (15.8×) and Kevin Wood (14.4×).
5. **Robotics specifically: few data points, mostly positive.**
   - bycloud's biggest video in the pull is a robotics one: Generalist's Gen-1.5 in-context learning, 9.9×, local
     10.3×. Its title is an analogy to a famous LLM moment.
   - Welch Labs' 35-min pi0 VLA explainer sits at its channel median (1.00×, about 217K views).
   - Kevin Wood's AGIBOT video with a withheld catch: 7.6×.
   - Asianometry's two robot videos: 1.58× and 0.94×.
   - AI Explained's autonomous-weapons episode is among its 3 lowest (0.44×).
6. **Title framing.**
   - Statements that withhold a catch recur among outliers: "…and we don't know why", "It's a Nightmare.", "But That's
     Not the Impressive Part".
   - "How…?" questions are 5 of the 45 outliers (Branch 3, bycloud 1, AI Explained 1). So `packaging.md`'s
     "statements over questions" may not hold for "how" titles in this format (small n).
7. **Length.**
   - In 11 of 15 channels, the ≥2× outliers run longer (median) than the channel's other videos. Examples: Welch
     23.7 vs 10.8 min, Branch 21.3 vs 9.5, AI Explained 27.0 vs 18.4, bycloud 19.9 vs 15.3, Brunton 24.3 vs 15.6,
     Articulated 25.2 vs 13.7.
   - Exceptions: 3Blue1Brown, Gonkee, Two Minute Papers, Ilia.
   - Within-channel ρ(length, views) ranges from −0.25 to 0.73 (median 0.38).
   - This is confounded: early uploads are shorter, and bigger productions run longer. It does not show that length
     causes views. It does show that nothing in these channels penalises 15–30 minute videos.
8. **Chapters.**
   - 32 of the 37 sampled videos of 10+ min have chapters. They have a median of 9 chapters (range 5–22). The first
     chapter (the intro) runs a median 64.5 s (IQR 43–140 s), and chapters average 2.5 min.
   - For the 14 videos of 12–25 min with chapters: a median of 8 chapters, a 57 s intro and 2.3 min per chapter.
   - Two of Aaed Musa's biggest videos have no chapters, so chapters are not required for an outlier.
   - Chapter titles show a recurring order: prior work or context → the core idea → one component at a time →
     results → limits. Two examples:
     - Welch Labs' VLA video: Intro / SayCan / RT-1 / PaLM-E / RT-2 / Pi0 Overview / PaliGemma / Action Expert
       Overview / … Deep Dive / "Just a Demo?" / World Models & Yann LeCun.
     - Real Engineering's F-16: Vietnam war and the MiG-21 / Energy maneuverability theory / turn performance /
       engine inlet / wing / …

## 2. Structure and retention: findings

Evidence first, then practitioner opinion. Each finding cites its sources.

**Evidence**

- **E1. Most leaving happens at the very start.**
  - Two-thirds of all dropouts (36.6 of 55.2 points) came in the first 3% of the video (S6).
  - For a 15-min episode, 3% is 27 s; YouTube's Intro metric is measured at 30 s (S1).
  - So the first 30 s must pay off the title, not warm up. edX autoplays, which inflates early exits there.
- **E2. Longer videos lose a larger share of viewers, so judge them against videos of the same length.**
  - Predicted dropout is 53% at 5 min vs 71% at 20 min (S6).
  - Duration explains over 58% of the variance in watch percentage (S8).
  - YouTube names no ideal length and warns against filler (S3). Studio's typical-retention comparison and Wu's
    relative engagement both control for length (S1, S8).
  - So a 20-min deep dive must earn each minute, and should be judged against similar-length videos, not against our
    4-min digests.
- **E3. Attention comes in chunks of a few minutes; segmenting helps learning.**
  - Median engagement time is at most 6 min regardless of length (S7).
  - Segmenting (d=0.70) and signalling (d=0.46) improve learning (S16).
  - Peer channels use chapters of about 2.3–2.5 min (section 1, item 8).
- **E4. Viewers act at visual transitions.** 61% of interaction peaks fell on visual transitions (S6). Chapter
  boundaries and key claims should get a clear visual change. The 5–7 s visual rule in `attention.md` already fits.
- **E5. A central story and emotional arousal predict which science videos are most watched.**
  - Odds ratios: central story 3.16, arousal 3.0 (S9).
  - The authors describe the typical popular video this way: a question posed at the start, a twist in the middle and
    a revelation at the end.
  - YouTube describes video essays as proving a thesis (S5).
  - Correlational, with a non-random sample.
- **E6. Refuting a misconception teaches better, and a wrong simple story feels just as understood.**
  - Refutation beat plain exposition by d≈0.8 (RCT, S14).
  - A misconception-based video gave equal felt understanding but worse knowledge (S15).
  - So a "where intuition fails" section helps, as long as both the intuition and the correction come from sources.
- **E7. Engineers are not novices.**
  - Guidance that helps novices can hurt experienced learners (S17).
  - Several multimedia principles are stronger for low-knowledge learners (S16).
  - Inference, untested on YouTube: keep basics short and spend the time on the system's specific design choices.
- **E8. The channel and a consistent presenter matter.**
  - A regular communicator went with more views (d=1.35, S10).
  - The channel alone explains much of engagement (S8; R²=0.42 in `research-reach.md`).
  - So keep one voice, one format and one opening pattern for the series.
- **E9. Planning before production matters.** Pre-production had the largest effect on engagement in edX (S7). The
  script and its source map are the main lever, not effects.

**Practitioner opinion (no data)**

- **P1.** The first minute should confirm the thumbnail's promise, and later sections have their own jobs (S19).
  Leaked, unverified, and about entertainment.
- **P2.** Give a concrete example before the general framework, and end on the definition rather than opening with
  it. Topic choice matters more than polish (S18).
- **P3.** "Re-hooks" (a fresh open question at each section boundary) are common creator advice. No primary source
  or data was found; a Paddy Galloway episode lists "re-engagement hooks", but only its chapter list was visible.

## 3. DRAFT story-selection criteria (for the user to validate)

> **DRAFT.** Built from the findings above and the digest rubric. Not yet tested against our own retention data.
> Weights are deliberately left out: rank with the existing panel method, then calibrate after 6 episodes, as in
> `research-tuning.md`.

### Hard gates (every one must pass)

| Gate | Definition | What decides it | Supported by |
|---|---|---|---|
| G1. Technical primary source | **Learning:** a paper or technical report with method, data and evaluation sections, or a project page plus code. **News:** a primary document with technical substance (filing, recall report, regulator decision, company engineering post). A press release alone fails | The verification record's `primary_url`, plus whether its method or detail sections exist | Sourcing rule; S15 (a wrong explanation feels as clear as a right one, so gaps can't be filled by inference) |
| G2. Verified | Digest step 8 status is confirmed or corrected; no quote marked "not found" | `verify/*.json`, `check_quotes.py` | Same gate as the digest |
| G3. Enough to show | Real footage or figures of the system itself, plus enough source figures to anchor every planned chapter (about 4+ distinct visual sources). The mechanism can be drawn as our own diagram | The `visuals` field; a list of figures and clips | E4 (S6); Mayer's multimedia principle d=1.67 (S16); the 5–7 s visual rule |
| G4. One answerable central question | One sentence ("How did X do Y?" / "Why did X fail?") that the sources actually answer | Write the question and point to the answering passage | E5 (S9 central story; S5 thesis); S1 (opening matches the title) |
| G5. Fills 12 min without padding | At least 5 distinct sourced sub-topics: prior work, two or more mechanism parts, results, failure modes or limits, implications | A section outline with a source per section | E2 (S3 no filler; S6 longer means more dropout); peer chapter counts (item 8) |

### Scored criteria (1–10 each, reasoning first, as in the digest panel)

| Criterion | One-line definition | Evidence that decides it | Supported by |
|---|---|---|---|
| D1. Mechanism depth | How much real "how" there is: architecture, data, training recipe, control loop, hardware, and ablations showing what matters | Paper sections, ablation tables, appendix, code | Item 8 (peer deep dives run about 8–9 chapters); E7 (engineers want the specifics) |
| D2. Surprise or misconception | A sourced result that contradicts what a practitioner would predict, or a common belief the story corrects | Results vs baselines, ablations, failure counts, the authors' own statements | E5 (arousal); E6 (S14); outliers "…we don't know why", "How did a 27M model beat ChatGPT?" |
| D3. Known subject, unknown mechanism | Viewers know the name (lab, robot, product, a widely shared demo, or the problem itself), but the "how" isn't explained elsewhere | Digest attention signals, plus `chatter.py`: do existing YouTube videos explain the method, or only show the demo? | Item 2; item 3; S21 (curiosity peaks at intermediate knowledge) |
| D4. Story arc and stakes | Problem → earlier attempts → turn(s) → result → what's still open, with stakes for the field | Related-work lineage, failure history, the source's own framing | E5 (S9); Welch Labs' VLA lineage chapters; Asianometry's failure and reversal outliers |
| D5. Practical transfer | Something an engineer can use: released code, weights or data, a reproducible recipe, a design rule, cost or hardware numbers | Repo and licence, dataset card, bill of materials | Item 4 (guides and courses top Articulated, Kevin Wood, Ilia); S21 (practical value drives sharing) |
| D6. Durability | Still worth watching in 6–12 months: a new method class, a foundational idea, or the first explanation of a system line | Is this the defining instance? Is it a successor in a line we have already explained? | Item 4 (Ilia's decay from 7.5× to 1.0×; Brunton's overviews beat follow-ups); S21 (views accrue for months) |
| D7. Visual depth | Beyond the gate: success and failure clips, sim-and-real pairs, figures that animate into our diagrams. Decorative footage doesn't count | Verification `visuals`, project page | E4; `attention.md` (seductive details) |
| D8. Momentum (lowest priority; tie-breaker) | Attention this week | Digest signals | Item 3 (AI outliers are the biggest launches), but views accrue over months, so it should not decide on its own |

### How this differs from the digest rubric, and why

| Digest criterion | Deep dive | Why |
|---|---|---|
| Significance | Split into **D6 durability** and **D4 stakes** | A deep dive is watched for months and must teach something lasting. The headline importance of a week's news is not enough |
| Viewer pull | Split into **D3 known subject** and **D2 surprise** | The outliers pair a recognised subject with a hidden "how" or a twist |
| Evidence | Becomes **gates G1–G2**, plus depth in **D1** | Reliable isn't enough: the source must contain 12 minutes of method |
| Showability | **Gate G3** plus **D7** | The episode needs visuals for every chapter, not one good clip |
| Momentum | Kept, lowest priority | Short-term attention matters less for an evergreen format |
| (new) | **D1 mechanism depth**, **D5 practical transfer** | The engineer viewer and the teaching-channel outliers |
| Lineup rules (≥2 learning stories, ≤2 per company, variety) | Don't apply (one story). Instead: avoid a successor in a system line we already explained unless its mechanism is new | Item 4 (series decay) |

**Kinds.**
- **Learning** stories pass G1 most easily.
- **News** stories qualify only when a technical primary document exists. Asianometry's and Real Engineering's failure
  and industry videos show that the form works when the mechanism is documented.

**Illustration (gates only, using facts already in `episodes/2026-11-22-weekly/research.md`).**
- **Waymo (digest #1) fails G1 for now.** The notes say there is "no DMV or Waymo primary document", and the recall
  filings are not yet on NHTSA.
- **RoboBuilder passes G1, G3 and G4.**
  - G1: arXiv preprint, 135 physical robots.
  - G3: paired real and simulation videos.
  - G4: a central question about why designs that pass in simulation fail on the real course. The research notes
    record that 66% of simulated passes held in the real world.
- **Heave-Ho passes G1 and G3** (arXiv paper, official clips, a real four-G1 carry at 8/10).
- **ForceHarness: G3 needs a check.** Its per-task clips are sped up 8–64×.
- **UCSD soft robot: G1 needs a check.** The Science page returned 403; only Crossref metadata was read.

## 4. DRAFT episode structure (12–20 min)

> **DRAFT.**
> - Times are for a 15-min episode, with the 12–20 min range in brackets.
> - Length follows the material (S3). Cut a section rather than pad it.
> - Every episode gets description chapters: 3+, starting at 00:00 (S2). The first chapter title is the central
>   question.

| # | Section | 15-min target [range] | What it does | Tied to |
|---|---|---|---|---|
| 0 | Cold open | 0:00–0:30 [0:20–0:40] | The decisive real moment (footage), the central question in one sentence, and the title's promise shown to be true. No brand sting before it | E1 (S6, S1); P1 |
| 1 | Stakes and roadmap | 0:30–1:15 [to 1:30] | Why an engineer should care; what they'll be able to explain by the end; the chapter list on screen | E3 signalling (S16); peer intro chapters ~57–65 s (item 8) |
| 2 | Before this | 1:15–3:00 [2–3 min] | The lineage: what was tried and why it fell short. Define only the 2–4 terms this story needs | Pre-training d=0.46 (S16) but E7 (keep basics short); D4; Welch Labs' VLA lineage |
| 3–5 | The mechanism, in 2–4 chapters | 3:00–9:00 [5–8 min] | One component per chapter (e.g. data → model → control loop); a concrete example first, then the general rule; each chapter ends on the question the next one answers | E3 (2–3 min chunks); E4 (visual change at each boundary); P2; P3 |
| 6 | Results | 9:00–11:30 [2–3 min] | Numbers tied to the exact system and subset, baselines, ablations; the surprise lands here (the mid-video twist) | E5 (twist in the middle); verification scope rules |
| 7 | Where it breaks | 11:30–13:30 [1.5–3 min] | Failure modes, sim vs real, what the source does not show (its `gaps`); refute the tempting misconception with sourced facts | E6 (S14, S15) |
| 8 | What you can use and what's open | 13:30–15:00 [1–2 min] | Code, weights, data, design lessons, open questions | D5, D6; S21 (practical value) |
| 9 | Recap and end screen | 15:00–15:40 [0:30–0:45] | A three-line recap of the answer to the central question; point to the weekly digest | Signalling (S16); S1 (compelling content earlier, so the tail stays short) |

**Scaling.**
- **12 min:** two mechanism chapters; sections 7 and 8 at about 1 minute each.
- **20 min:** up to four mechanism chapters and longer results and limits sections. Add depth, not background.

**Check after publishing.**
- Compare retention per chapter with Studio's typical retention for similar-length videos (S1, S8). Never compare
  with the 4-min digest.
- Log each episode's criterion scores next to its retention, as in `research-tuning.md`.

## 5. Weak or missing evidence

- **No study measures retention for 12–20 min YouTube explainers.** The dropout and engagement numbers (S6, S7) come
  from edX MOOCs, where autoplay and course assignment differ from a viewer choosing a video.
- **Re-hooks, pattern interrupts and the "first minute" rule are practitioner-only** (S19, P3). The MrBeast doc's
  authenticity is unconfirmed and it covers entertainment.
- **Chapters' effect on retention is unknown.** No official data was found. Third-party claims (e.g. "30% longer")
  had no traceable source, and some practitioners argue chapters can lower average view duration by letting viewers
  skip.
- **No study tests which science topics work as long explainers.** Welbourne & Grant (S10) and Huang & Grant (S9)
  didn't analyse topic. Amarasekara & Grant (S12) is about presenter gender and comments, not topics.
- **Our outlier coding is weak evidence.**
  - One coder, titles only.
  - Views, not click-through or retention.
  - Rounded view counts and approximate dates.
  - Age confounds in channels whose pull spans years (local ratios help, but don't remove it).
  - 15 channels chosen partly by name, partly by search.
- **Robotics-specific demand** rests on a handful of videos (bycloud 9.9×, Welch Labs 1.00×, Ilia's series, Kevin
  Wood).
- **The expert audience is barely studied.** Most findings come from novices or the general public (S14–S16). The
  expertise-reversal inference (E7) has not been tested on YouTube.
- **Faceless TTS narration and a consistent presenter.** The communicator effect (S10) involved human hosts; whether a
  consistent TTS voice gets the same effect is unknown.
- **Length.** Yang et al. (S11) found shorter videos more likely to be viewed, on one channel. Our pull found the
  opposite pattern within most channels. Both are correlational.
- **Not used:** vendor retention benchmarks by length (e.g. Retention Rabbit, via a third-party page with
  inconsistent numbers). The primary report was not found.
- **Not fetched:**
  - Kulgemeyer & Wittwer's PMC page (bot check; the abstract came from Europe PMC instead).
  - Springer's copy (login redirect).
  - The Rey et al. 2019 segmenting meta-analysis (MPI page 403 and a certificate error), so its effect sizes aren't
    cited.
- **Our own data.** No channel retention data exists yet. The first deep dives' chapter retention will outrank
  everything above.
