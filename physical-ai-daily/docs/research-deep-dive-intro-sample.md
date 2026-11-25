# How technical deep-dive videos open: a measured sample (2026-11-25)

**What this is.** Coded openings (first ~90 s of transcript plus chapter list) of 25 niche videos (robotics, physical AI, AI/ML research explainers) from 15 channels, plus a 3-video general-science reference group reported separately. Every video is its channel's outlier (views >= 2x the channel median of recent uploads), 10-30 min long and about one topic. The raw table is `sample.json` in this folder.

**Caveats.**
- Times are approximate. Auto-caption word timings are +-2 s. The 4 transcripts fetched with youtube-transcript-api (AI Explained GPT-6, bycloud Robots, both Kevin Wood) have segment start times, +-3-5 s.
- One coder, from speech only. Visuals were not watched, so "footage/demo" means the speech refers to on-screen footage ("this is...", "look at this").
- Null means not found within 90 s. Medians are given two ways: among the videos that have the element, and with misses counted as >90 s.
- 25 + 3 videos is a small, purposive sample of outliers. It describes what successful videos do. It does not show that the openings cause the views.
- No video was dropped: all 28 had English captions. 4 were fetched with youtube-transcript-api after yt-dlp hit HTTP 429.

## Key findings (niche sample, n=25; all times approximate)

1. **The subject is named almost at once.**
   - 23/25 name it within 90 s, at a median of 5 s (IQR 2.5–12.5 s). 17/25 name it within 10 s.
   - In 17/25 the subject is in the first thing said.
   - Only bycloud's two videos hold it past 45 s: they name it at ~110 s and at ~123 s or later.
2. **The source comes later, and is often missing.**
   - 16 videos are about someone else's work. 12/16 name the lab, company or paper within 90 s, at a median of 18 s (IQR 5–39 s).
   - With misses counted as >90 s, the median is 30.5 s. 7/16 name the source after 45 s or not at all within 90 s.
3. **Stakes come early, and the question comes later.**
   - A "why it matters" claim appears in 22/25, at a median of 13 s (IQR 8.5–25.5 s).
   - A central question or problem is stated in 18/25, at a median of 36 s (IQR 13.5–51.5 s).
   - The subject comes before the question in 13 of 16 videos that have both, plus 1 tie.
4. **A promise is common; a roadmap is not.**
   - Explicit promise ("in this video", "let's build", "today we look at"): 8/25 within 30 s, 14/25 within 60 s, 15/25 within 90 s. Median 29 s.
   - Roadmap of 2 or more parts: 3/25 within 30 s, 7/25 within 60 s, 9/25 within 90 s. Median 2 items.
   - 9/25 have neither within 90 s. Several of those put a series or section plan just after 90 s; for example, Mutual Information's "six part series" comes at ~169 s.
5. **Cold opens are mostly a plain statement of the topic** (12/25). The rest: story or anecdote 4, footage/demo 3, other 3, mystery or question 2, surprising number 1.
6. **Hardly any channel introduces itself.**
   - 2/25 have a channel or host identity line: Two Minute Papers' "Fellow Scholars" at ~2 s, and Mutual Information's host saying where he works at ~27 s.
   - 3/25 have any sponsor mention within 90 s, and each is one line. Full sponsor reads come after 90 s.
7. **Research and release videos open differently from builds and concept videos.**
   - In 6 of 11 research/release videos, the problem or stakes come before the subject. That happens in 0 of 8 builds and tutorials, and 1 of 6 concept, history or industry videos.
   - The research/release group also names its subject later (median 16 s, against 4 s and 3 s). Its question comes later too (median 48 s), and it has fewer roadmaps within 60 s (2/11, against 4/8 for builds).
8. **The first chapter is long.** 15/25 have creator chapters, and their first chapter runs a median of 80 s (IQR 54.5–126 s). This is longer than the 64.5 s median in the earlier 37-video pull, which included general-science channels.
9. **Dominant order.**
   - Name-first, in 18/25: [hook] → subject (≤10 s) → stakes → promise (~30 s) → question (~35–50 s) → source wherever it fits.
   - Problem-first, in 7/25, mostly research videos: [hook] → problem or stakes → subject (~15–40 s) → promise, if any.
   - The four-element sequences are scattered: 17 different orders across 25 videos. So the reliable rule is only that the subject comes first, not one fixed script.
   - The reference group (3Blue1Brown, Real Engineering, Veritasium) follows name-first: subject 0–16 s, stakes 16–25 s, source 17–74 s.


## Method

1. Channel pull: yt-dlp `--flat-playlist -J --playlist-end 80` on each channel's /videos tab (22 channels). Median views = the latest 80 long uploads minus the newest 4.
2. Niche channels: the 15 of `docs/research-deep-dive-format.md` section 1, plus yt-dlp searches ("robot learning explained", "VLA explained", "sim to real explained", "humanoid robot deep dive", "physical AI explained", "robotics paper explained", "reinforcement learning robot explained", and others). The searches added James Bruton, Julia Turc, Foundation Robotics, Art of the Problem and Mutual Information.
3. Per channel: the 1-2 most-viewed or clearest >=2x outliers, 10-30 min, one topic. Dropped: Ilia (no 10-30 min video), Gonkee (no >=2x video in range), Branch and Kurzgesagt (general science; reference group capped at 3).
4. Per video: info JSON (views, chapters) and English captions only. yt-dlp `--skip-download --write-subs --write-auto-subs`, with youtube-transcript-api as the fallback. No video or audio was downloaded.
5. Coding rules:
   - **subject** = the specific thing named in speech (a model, paper, machine or method), not a field.
   - **source** = the lab, company, author or paper.
   - **question** = a question or problem stated as such, not one embedded in a promise.
   - **why** = an explicit claim of importance or stakes.
   - **promise** = an explicit statement of what the video will cover or give ("in this video", "today we look at", "let's build", a thesis "the point I want to make").
   - **roadmap** = 2 or more upcoming parts listed. "Structural" means sections; "teaser" means story beats.

## Sample

| # | Channel | Video | Focus | Views | Ratio | Min | Captions |
|---|---|---|---|---|---|---|---|
| 1 | Welch Labs | [AI can't cross this line and we don't know why.](https://www.youtube.com/watch?v=5eqRuVp65eY) | research | 2,531,843 | 9.5x | 24 | auto captions |
| 2 | Welch Labs | [The moment we stopped understanding AI [AlexNet]](https://www.youtube.com/watch?v=UZDiGooFs54) | research | 2,564,816 | 9.7x | 17 | auto captions |
| 3 | AI Explained | [GPT 6 Astra, so good even OpenAI are worried](https://www.youtube.com/watch?v=Spuza-KwTJ4) | research | 859,178 | 8.3x | 29 | youtube-transcript-api |
| 4 | AI Explained | [o3 - wow](https://www.youtube.com/watch?v=YAgIh4aFawU) | research | 289,488 | 2.8x | 22 | auto captions |
| 5 | bycloud | [Robots Just Had Their GPT-3 Moment](https://www.youtube.com/watch?v=cqwKceUSZ5Q) | research | 743,033 | 11.3x | 15 | youtube-transcript-api |
| 6 | bycloud | [1-Bit LLM: The Most Efficient LLM Possible?](https://www.youtube.com/watch?v=7hMoz9q4zv0) | research | 403,330 | 6.1x | 14 | auto captions |
| 7 | Two Minute Papers | [New DeepSeek Research - The Future Is Here!](https://www.youtube.com/watch?v=fFL7la73RO4) | research | 294,991 | 2.3x | 12 | manual captions |
| 8 | Yannic Kilcher | [Tree of Thoughts: Deliberate Problem Solving with Large La](https://www.youtube.com/watch?v=ut5kp56wW_4) | research | 128,478 | 3.6x | 29 | auto captions |
| 9 | Yannic Kilcher | [Retentive Network: A Successor to Transformer for Large La](https://www.youtube.com/watch?v=ec56a8wmfRk) | research | 103,444 | 2.9x | 28 | auto captions |
| 10 | Articulated Robotics | [The Ultimate Guide to using Motors in Robotics (including ](https://www.youtube.com/watch?v=-PCuDnpgiew) | build | 537,626 | 11.7x | 25 | auto captions |
| 11 | Articulated Robotics | [Easy SLAM with ROS using slam_toolbox](https://www.youtube.com/watch?v=ZaiA3hWaRzE) | build | 289,410 | 6.3x | 25 | auto captions |
| 12 | Aaed Musa | [High Precision Speed Reducer Using Rope](https://www.youtube.com/watch?v=MwIBTbumd1Q) | build | 3,058,560 | 61.8x | 20 | auto captions |
| 13 | Aaed Musa | [I Built an Internal Cycloidal Robotic Actuator](https://www.youtube.com/watch?v=GakFB8Tdd98) | build | 1,318,141 | 26.6x | 19 | auto captions |
| 14 | Steve Brunton | [Reinforcement Learning: Machine Learning Meets Control The](https://www.youtube.com/watch?v=0MNVhXEX9to) | concept | 434,953 | 21.7x | 26 | manual captions |
| 15 | Kevin Wood / Robotics & AI | [AGIBOT A3 Humanoid Robot Does Kung Fu… But That’s Not the ](https://www.youtube.com/watch?v=ES8atEmxBeI) | research | 38,758 | 8.1x | 11 | youtube-transcript-api |
| 16 | Kevin Wood / Robotics & AI | [Reinforcement Learning for Robotics — Simulation to Real-W](https://www.youtube.com/watch?v=bPtiN8MS-LE) | build | 29,488 | 6.1x | 19 | youtube-transcript-api |
| 17 | James Bruton | [I built an Omni-Directional Ball-Wheeled Bike](https://www.youtube.com/watch?v=ZVFB2g25OkM) | build | 10,834,580 | 55.8x | 27 | auto captions |
| 18 | James Bruton | [How Robots Use Maths to Move](https://www.youtube.com/watch?v=IN8tjTk8ExI) | build | 547,693 | 2.8x | 15 | auto captions |
| 19 | Asianometry | [Silicon Valley Thinks TSMC is Braking the AI Boom](https://www.youtube.com/watch?v=2lLFBun1qR0) | concept | 458,922 | 2.3x | 20 | manual captions |
| 20 | Art of the Problem | [I Gave ChatGPT a Body](https://www.youtube.com/watch?v=S67z2aekBrI) | build | 5,710,452 | 57.1x | 27 | auto captions |
| 21 | Art of the Problem | [The 35 Year History of ChatGPT](https://www.youtube.com/watch?v=OFS90-FX6pg) | concept | 1,329,123 | 13.3x | 26 | manual captions |
| 22 | Mutual Information | [Reinforcement Learning, by the Book](https://www.youtube.com/watch?v=NFo9v_yKQXA) | concept | 233,363 | 2.9x | 18 | auto captions |
| 23 | Julia Turc | [It’s von Neumann’s fault / GPU · TPU · LPU](https://www.youtube.com/watch?v=yTXHZxXXjPI) | concept | 319,541 | 7.3x | 29 | manual captions |
| 24 | Julia Turc | [How far are we from "Her"?](https://www.youtube.com/watch?v=kOD9rMhn4f4) | research | 300,822 | 6.9x | 21 | auto captions |
| 25 | Foundation Robotics | [The Crazy History of the First Humanoid Robot](https://www.youtube.com/watch?v=IailkUfSUdo) | concept | 883,995 | 74.9x | 23 | auto captions |
| 26 | 3Blue1Brown | [Transformers, the tech behind LLMs / Deep Learning Chapter](https://www.youtube.com/watch?v=wjZofJX0v4M) | reference | 11,759,752 | 5.0x | 27 | manual captions |
| 27 | Real Engineering | [The Insane Engineering of the F-35B](https://www.youtube.com/watch?v=1lCOgFPtaZ4) | reference | 10,928,428 | 5.0x | 25 | manual captions |
| 28 | Veritasium | [Can You Keep Zooming In Infinitely?](https://www.youtube.com/watch?v=88bMVbx1dzM) | reference | 31,962,723 | 2.9x | 21 | manual captions |

Ratio = exact views / median of the channel's recent long uploads. Foundation Robotics' median rests on only 8 videos.

## 1. Timing of subject, source, question and stakes (niche, n=25)

| Element | Result |
|---|---|
| Subject named | 23/25 within 90 s; median 5 s (IQR 2.5-12.5) among those; median with misses counted as >90 s: 5 |
| Source (lab/company/paper) named | 13/25 within 90 s; median 20 s (IQR 6-46) among those; median with misses counted as >90 s: 87 |
| Central question/problem stated | 18/25 within 90 s; median 36 s (IQR 13.5-51.5) among those; median with misses counted as >90 s: 50 |
| Why it matters (stakes) | 22/25 within 90 s; median 13 s (IQR 8.5-25.5) among those; median with misses counted as >90 s: 16 |
| Source, only the 16 videos about outside work | 12/16 within 90 s; median 18 s (IQR 5.25-39.25) among those; median with misses counted as >90 s: 30.5 |
| Promise | 15/25 within 90 s; median 29 s (IQR 7-43) among those; median with misses counted as >90 s: 60 |
| Roadmap | 9/25 within 90 s; median 34 s (IQR 25-60) among those; median with misses counted as >90 s: >90 |
| First spoken word | median 0.1 s; max 8.5 s; 1/25 start speaking after 3 s |

9 niche videos are own builds, tutorials or a lecture, so they have no outside source. The source row for the 16 videos about outside work is the fairer one.
First spoken element after the hook (merged within 2 s): subject 17, question 4, why 2, channel-line 1, source 1.
Named later than 45 s or not within 90 s: subject 2/25; source 7/16 of the videos about outside work.

## 2. Promise and roadmap (niche, n=25)

| | within 30 s | within 60 s | within 90 s |
|---|---|---|---|
| Explicit promise | 8/25 (32%) | 14/25 (56%) | 15/25 (60%) |
| Roadmap (any) | 3/25 (12%) | 7/25 (28%) | 9/25 (36%) |
| Roadmap (structural only) | 2/25 | 6/25 | 8/25 |

Roadmap length: [2, 2, 2, 2, 2, 3, 3, 4, 5] items (median 2). A promise with no roadmap: 7/25. Neither within 90 s: 9/25.

## 3. Cold-open types (niche, n=25)

| Type | n |
|---|---|
| direct statement of topic | 12 |
| story or anecdote | 4 |
| footage/demo | 3 |
| other | 3 |
| mystery or question | 2 |
| surprising fact or number | 1 |

## 4. Channel line and sponsors (niche, n=25)

- **Channel or host identifies itself by name or role within 90 s: 2/25.** Two Minute Papers ~2 s ('Fellow Scholars' signature line; names the channel ~95 s); Mutual Information ~27 s (host identifies himself (works at Lyft)).
- Other channel mentions are greetings ("hi there", "welcome back"), plugs for earlier builds, or a membership plug. None of them introduces the channel.
- **Sponsor within 90 s: 3/25.** Welch Labs ~32 s (KiwiCo one-line mention ('more on them later')); Aaed Musa ~0 s (PCBWay one-liner as the first words); James Bruton ~82 s (thanks a filament supplier). The full sponsor reads in this sample come after 90 s: bycloud at ~125 s, Art of the Problem at 231 s, others later.

## 5. Order of elements within 90 s

Pairwise precedence (niche, among videos that have both elements):

- subject before question: 13/16 (+1 tie)
- subject before why: 13/20 (+3 tie)
- why before question: 8/15
- subject at or before promise: 13/15

Sequence of the four core elements only (S subject, W stakes, Q question, P promise; ties by speech order S>W>Q>P):

| Pattern | n |
|---|---|
| SW | 3 |
| SWPQ | 2 |
| QW | 2 |
| WSQ | 2 |
| SQP | 2 |
| SPQW | 2 |
| SWP | 2 |
| QSW | 1 |
| PSWQ | 1 |
| WSPQ | 1 |
| PSW | 1 |
| SPW | 1 |
| QWSP | 1 |
| SPQ | 1 |
| SWQP | 1 |
| SQW | 1 |
| SWQ | 1 |

Per-video sequence (elements within 2 s merged with "+"):

| Channel | Video | Focus | Sequence |
|---|---|---|---|
| Welch Labs | AI can't cross this line and we don't know why. | research | [mystery] question -> subject -> why -> source |
| Welch Labs | The moment we stopped understanding AI [AlexNet] | research | [footage] subject+source -> why |
| AI Explained | GPT 6 Astra, so good even OpenAI are worried | research | [direct-statement] subject -> why -> source -> promise+roadmap -> question |
| AI Explained | o3 - wow | research | [direct-statement] source+subject -> why -> promise -> roadmap -> question |
| bycloud | Robots Just Had Their GPT-3 Moment | research | [other] question -> why |
| bycloud | 1-Bit LLM: The Most Efficient LLM Possible? | research | [mystery] question -> why |
| Two Minute Papers | New DeepSeek Research - The Future Is Here! | research | [direct-statement] channel-line -> source -> why -> subject -> question |
| Yannic Kilcher | Tree of Thoughts: Deliberate Problem Solving with Large Lang | research | [direct-statement] promise+subject+source -> why -> question |
| Yannic Kilcher | Retentive Network: A Successor to Transformer for Large Lang | research | [other] why -> subject -> promise -> question -> source |
| Articulated Robotics | The Ultimate Guide to using Motors in Robotics (including RO | build | [story] subject -> question -> promise -> roadmap |
| Articulated Robotics | Easy SLAM with ROS using slam_toolbox | build | [direct-statement] promise+subject -> why -> roadmap |
| Aaed Musa | High Precision Speed Reducer Using Rope | build | [footage] subject -> promise -> question -> why |
| Aaed Musa | I Built an Internal Cycloidal Robotic Actuator | build | [direct-statement] subject -> why -> promise -> source |
| Steve Brunton | Reinforcement Learning: Machine Learning Meets Control Theor | concept | [direct-statement] subject+promise -> why |
| Kevin Wood / Robotics & AI | AGIBOT A3 Humanoid Robot Does Kung Fu… But That’s Not the Im | research | [footage] question -> why+subject+source -> promise+roadmap |
| Kevin Wood / Robotics & AI | Reinforcement Learning for Robotics — Simulation to Real-Wor | build | [direct-statement] subject+promise -> roadmap -> question -> why |
| James Bruton | I built an Omni-Directional Ball-Wheeled Bike | build | [direct-statement] subject -> promise -> question |
| James Bruton | How Robots Use Maths to Move | build | [direct-statement] subject -> why -> promise -> roadmap |
| Asianometry | Silicon Valley Thinks TSMC is Braking the AI Boom | concept | [other] source -> subject -> why -> question -> promise |
| Art of the Problem | I Gave ChatGPT a Body | build | [story] subject -> question -> why |
| Art of the Problem | The 35 Year History of ChatGPT | concept | [direct-statement] subject+why -> question |
| Mutual Information | Reinforcement Learning, by the Book | concept | [direct-statement] subject+why -> channel-line |
| Julia Turc | It’s von Neumann’s fault / GPU · TPU · LPU | concept | [fact] why -> subject+question -> source |
| Julia Turc | How far are we from "Her"? | research | [story] subject+source -> question -> promise -> roadmap |
| Foundation Robotics | The Crazy History of the First Humanoid Robot | concept | [story] subject+why -> roadmap -> source |
| 3Blue1Brown | Transformers, the tech behind LLMs / Deep Learning Chapter 5 | reference | [direct-statement] subject -> why -> promise -> source |
| Real Engineering | The Insane Engineering of the F-35B | reference | [direct-statement] subject+why -> source |
| Veritasium | Can You Keep Zooming In Infinitely? | reference | [footage] subject -> source -> why -> question |

## 6. Research/release videos vs other niche videos

| Group | n | Subject named | Source named | Question | Stakes | Promise <=60 s | Roadmap <=60 s |
|---|---|---|---|---|---|---|---|
| niche (all) | 25 | 23/25; med 5 (2.5-12.5) | 13/25; med 20 (6-46) | 18/25; med 36 (13.5-51.5) | 22/25; med 13 (8.5-25.5) | 14/25 (56%) | 7/25 (28%) |
| research/release | 11 | 9/11; med 16 (3-20) | 9/11; med 16 (6-31) | 10/11; med 48 (18.75-51.5) | 10/11; med 17 (10.5-50.5) | 6/11 (55%) | 2/11 (18%) |
| build/tutorial | 8 | 8/8; med 4 (2-8) | 1/8; med 46 (46-46) | 5/8; med 22 (8-36) | 6/8; med 15 (12.5-28.75) | 7/8 (88%) | 4/8 (50%) |
| concept/history/industry | 6 | 6/6; med 3 (0.75-4.5) | 3/6; med 30 (16-58.5) | 3/6; med 28 (21.5-43.5) | 6/6; med 1.5 (0-6.75) | 1/6 (17%) | 1/6 (17%) |
| reference (general science) | 3 | 3/3; med 14 (7-15) | 3/3; med 37 (27-55.5) | 1/3; med 42 (42-42) | 3/3; med 22 (19-23.5) | 1/3 (33%) | 0/3 (0%) |

Cell format: number within 90 s / n; median (IQR) in seconds among those.

## 7. Chapters

- niche: creator chapters 15/25, first chapter median 80 s (IQR 54.5-126, range 27-214); auto-only chapters 8; none 2.
- reference: creator chapters 2/3, first chapter median 157.5 s (IQR 144.75-170.25, range 132-183); auto-only chapters 1; none 0.

## Illustrative examples (approximate times; paraphrased unless quoted)

1. **Kevin Wood, "AGIBOT A3 Humanoid Robot Does Kung Fu… But That's Not the Impressive Part"** (8.1×, 11 min). This is the closest match to the draft's open: robot footage, then the problem.
   - 0 s: footage line. It looks like a kung-fu film, but "the important part is not the choreography".
   - ~13 s: the problem is named ("That is a sim-to-real problem").
   - ~18 s: stakes. It is "one of the biggest challenges in robotics".
   - ~20 s: the subject and source together (AGIBOT's A3 performance, filmed without CGI).
   - ~25 s: promise plus a 3-item roadmap: why RL is hard, why simulation never matches reality, and what the moves say about deployment.
2. **Welch Labs, "AI can't cross this line and we don't know why."** (9.5×, 24 min). Mystery first, source late, no promise.
   - 0–2 s: "AI models can't cross this boundary and we don't know why".
   - ~24 s: the line is named as the compute-optimal frontier, and ~29 s as one of three neural scaling laws.
   - ~45 s: the question. Is this a law of nature, or an artefact of today's methods?
   - ~60 s: stakes asked as questions. How powerful can models get? Can error reach zero?
   - ~78 s: the source, OpenAI's 2020 paper, as the history begins.
3. **AI Explained, "GPT 6 Astra, so good even OpenAI are worried"** (8.3×, 29 min). Name first, then a two-part roadmap.
   - 0 s: the subject ("GPT-6 is here at last").
   - ~10 s: stakes. People who haven't checked in lately hold stale views.
   - ~31 s: the source. Its silent reasoning unsettles people at OpenAI.
   - ~34 s: promise and roadmap. The first half covers why it is so good; the second half covers the concerns.
   - ~48 s: the question ("why do people say it's so good?").
4. **Yannic Kilcher, "Tree of Thoughts (Full Paper Review)"** (3.6×, 29 min). Paper named and summarised, result included, within the first minute.
   - ~1–3 s: promise and subject ("a quick look at this paper", read by title).
   - ~6–55 s: summary of the whole paper: tree search over model outputs, new tasks, and that it "works really well" on them.
   - ~58 s: his take, as stakes. It is a small step toward mixing language models with algorithms.
   - ~75 s: the institutions (Princeton, Google DeepMind).
   - ~85 s: the problem. Plain prompting falls short, however well it is phrased.
5. **bycloud, "Robots Just Had Their GPT-3 Moment"** (11.3×, 15 min). The counter-example: the subject is held back for nearly 2 minutes.
   - 0 s: an analogy. LLMs' big milestone was in-context learning.
   - ~48 s: the question ("Can this behavior also emerge in other modalities?").
   - ~60 s: stakes. Nothing was more anticipated than this moment in robotics.
   - ~66–105 s: why robotics is harder.
   - ~110 s: the subject and source (Generalist AI's GEN-1.5).
   - ~125 s: an ad for the creator's own course platform.
6. **Aaed Musa, "High Precision Speed Reducer Using Rope"** (61.8×, 20 min). A build, opening on the outcome.
   - 0 s: three finished robots that are "each driven by rope".
   - ~26 s: promise. He'll share the whole journey of making "the most underrated reducer".
   - ~36 s: the question ("so why rope?"), answered with gear backlash.
   - ~52 s: stakes. Precision matters when building robots.
   - ~66 s: the specific name, capstan drive.

## Comparison with the draft opening (data only)

The draft runs 27 s of robot failure, then a question, then names MIT's RoboBuilder preprint at ~45 s. It never states what the video covers. Against the niche sample:

- **Subject at ~45 s.** 23/25 niche videos name the subject sooner. The only later ones are bycloud's two, which name it at ~110 s and later. Research/release videos: median 16 s, IQR 3–20 s.
- **Source at ~45 s.** Of the 16 videos about outside work, 9 name the source by 45 s, 3 name it between 46 and 90 s (Welch ~78 s, Yannic RetNet ~64 s, Foundation Robotics ~87 s), and 4 not within 90 s.
- **Question at ~27 s.** This falls inside the niche range (median 36 s, IQR 13.5–51.5 s).
- **No promise and no roadmap.** 9/25 niche videos also have neither within 90 s. 15/25 make a promise (median 29 s), and 9/25 give a roadmap (median 34 s).
- **27 s before speech.** This only applies if the 27 s carries no narration. In the niche sample, the first spoken word comes at a median of 0.1 s; the latest is 8.5 s (James Bruton). In the reference group, the latest is ~16 s (Real Engineering).
- **Closest structural analogue.** Kevin Wood's AGIBOT video also opens on robot footage plus a problem. It names the subject and source at ~20 s and gives a promise with a 3-item roadmap at ~25 s.

