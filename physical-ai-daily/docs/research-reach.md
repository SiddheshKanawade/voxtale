# What content gets reach on YouTube (research, 2026-11-22)

The user asked: what kind of content gets better reach, whether "dopamine" content wins, and what else works
(curiosity, amazement, raised eyebrows), with studies preferred. Two literature reviews were run: emotion and virality
psychology, and YouTube's own system plus YouTube video studies. Packaging (titles and thumbnails) is in
`docs/packaging.md`.

Strength labels: **OFF** = YouTube's own statements or papers; **RCT** = large randomized tests; **LD** = large
observational dataset; **EXP** = experiment; **SMALL** = n under about 60; **V** = vendor data, weak.

## 1. What YouTube rewards

| Finding | Source | Strength |
|---|---|---|
| Ranking uses predicted watch time that viewers are glad of, not clicks. CTR ranking "often promotes deceptive videos". "Valued watchtime" counts only views rated 4–5 stars in surveys | [Covington et al. 2016](https://research.google/pubs/pub45530); [Zhao et al. 2019](https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/); [Goodrow 2021](https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/) | OFF |
| More than 70% of watch time comes from recommendations. Subscribers click about 10% of uploads | Mohan 2018 (via Tubefilter); Beaupré, Creator Insider 2026 (via ppc.land) | OFF, secondhand |
| The channel is the strongest predictor of engagement (R²=0.42 for relative engagement from channel alone). In science videos, the channel correlates with views at 0.746 | [Wu, Rizoiu & Xie 2018](https://arxiv.org/abs/1709.02541) (5.3M videos); [Velho et al. 2020](https://www.frontiersin.org/articles/10.3389/fcomm.2020.567606/full) | LD / peer-reviewed |
| Science channels with one regular communicator get more views (d=1.35). User-generated channels beat corporate ones. Length does not correlate with views | [Welbourne & Grant 2016](https://openresearch-repository.anu.edu.au/bitstream/1885/107245/2/01%20Dustin%20and%20Grant%20Science%20communication%20on%202016.pdf) (390 videos) | Peer-reviewed, correlational |
| **Story structure and emotional arousal predict which science videos are most viewed:** arousal OR ≈ 3.0, "story status" OR ≈ 3.16, top videos have 3+ moments of change. An explicit "dramatic question" adds little | [Huang & Grant 2020](https://www.frontiersin.org/articles/10.3389/fcomm.2020.581349/full) (306 videos) | Peer-reviewed, correlational, small |
| Narrative raises non-experts' comprehension, interest and engagement | [Dahlstrom 2014, PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC4183170) | Review |
| Animation helps views per day; talk and interview formats hurt. Continuous visual flow and fast, enthusiastic speech hold viewers | Velho 2020; [Guo, Kim & Rubin 2014](https://www.cs.rochester.edu/hci/pubs/pdfs/edX-MOOC-video-production-and-engagement_LAS-2014.pdf) (6.9M MOOC sessions) | Peer-reviewed |
| For News, views and watch time diverge (ρ = −0.34; Music +0.80) | Wu 2018 | LD |
| Early views compound. For young videos the uploader matters most. Videos can keep earning for months, so judge them over 90+ days | [Borghol et al. 2012](https://arxiv.org/abs/1311.6526); Beaupré 2026 | Peer-reviewed / OFF |
| Shorts run on their own system (a seed audience, then decided in the swipe). In Science & Tech, Shorts lead long-form by less | Sherman 2023 (via TechCrunch); [Violot et al. 2024](https://arxiv.org/abs/2403.00454) (9.9M Shorts) | OFF / LD |

## 2. Which emotions and qualities make content spread

| # | Quality | Effect | Source | Strength |
|---|---|---|---|---|
| 1 | **High-arousal emotion** | Per +1 SD in reaching NYT's most-emailed list: anger +34%, **awe +30%**, anxiety +21%, surprise +14%; **sadness −16%** (low arousal) | [Berger & Milkman 2012](https://journals.sagepub.com/doi/10.1509/jmr.10.0353) (6,956 articles) | LD + EXP |
| 2 | **Positive high-arousal emotion** (amusement, excitement, inspiration, warmth, plus drama or surprise) | Raises sharing of YouTube videos. Only 7% of ads evoked these | [Tellis et al. 2019](https://doi.org/10.1177/0022242919841034); [Nelson-Field et al. 2013](https://ideas.repec.org/a/eee/aumajo/v21y2013i4p205-211.html) | LD |
| 3 | **Practical value** | The strongest predictor of sharing among people who already viewed, but it doesn't raise views | [Kim 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4591750); Berger & Milkman | LD |
| 4 | **Curiosity / information gap** | Engages reward circuits; people pay to resolve it and remember the answer better. In headline tests, open references ("this") raised clicks, while explicit "you'll learn X" promises lowered them | [Kang et al. 2009](https://pubmed.ncbi.nlm.nih.gov/19619181/); [Gruber et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4252494); [Banerjee & Urminsky](https://home.uchicago.edu/ourminsky/Banerjee_Urminsky_Headlines.pdf) | EXP (small) + RCT |
| 5 | **Surprise / novelty** | Draws attention (72% of gaze shifts go to surprising spots). Surprise focuses attention, and rising joy stops viewers zapping away. But novelty lowered views of health news | [Itti & Baldi 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2860069); [Teixeira et al. 2012](https://research.tilburguniversity.edu/en/publications/emotion-induced-engagement-in-internet-video-ads/); Kim 2015 | EXP / SMALL |
| 6 | **Awe / amazement** | Shared more (row 1). Raises awareness of knowledge gaps, which goes with more interest in science. Can backfire with some audiences | [McPhetres 2019](https://doi.org/10.1080/02699931.2019.1585331) (N=1,518); Valdesolo 2016 | EXP |
| 7 | **Humour** | Raises sharing; science jokes led to more relevant comments | Berger & Milkman 2a; [Cacciatore et al. 2024](https://jcom.sissa.it/archive/23/01/JCOM_2301_2024_A02) | EXP |
| 8 | Outrage, moral-emotional words, out-group animosity | +12% shares per moral-emotional word in the replication; out-group words +67% odds. Spreads misinformation equally well, and users rate engagement-ranked anger feeds lower | [Brady et al. 2025](https://doi.org/10.1093/pnasnexus/pgaf327); [Rathje et al. 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8256037/); [Milli et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11894805/) | LD; **not for us** |
| 9 | Negative words | +2.3% CTR per word in headlines (clicks, not shares) | [Robertson et al. 2023](https://www.nature.com/articles/s41562-023-01538-4) | RCT |

**The first seconds predict reach.** Nucleus accumbens activity in the first seconds of Discovery and Animal Planet
YouTube clips forecast their real views and percent watched. Viewers' own ratings of emotion did not ([Tong et al.
2020, PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC7104008), n=36, SMALL).

## 3. "Dopamine content": verdict

**Pop-science shorthand, not a scientific category.**
- Dopamine signals reward prediction error, novelty, and the value of advance information (Schultz 1997; Bunzeck &
  Düzel 2006; Bromberg-Martin & Hikosaka 2009; Kobayashi & Hsu 2019).
- It drives "wanting", not pleasure (Berridge & Robinson 2016).
- No virality study measures dopamine; fMRI measures blood flow.

The defensible version: curiosity (anticipating information), surprise and novelty (prediction errors) and early
positive arousal engage reward circuits and predict attention and memory. Those are the levers; "dopamine" adds
nothing actionable.

## Weak or not supported

- Arousal *by itself* causing sharing: the Berger 2011 jogging study failed two replications (Prowten et al. 2024).
- Question or "mystery" framing raising YouTube views: Huang & Grant found a small, non-significant effect, and
  question clickbait titles got fewer views across 16,215 videos (Cui et al. 2024).
- Peer-reviewed tests of hooks, pacing (shot length), faceless/TTS narration and thumbnails on YouTube: none found.
- Short-video "addiction" or "brain rot" as causal: correlational only (a 2025 meta-analysis, plus one n=30 fMRI
  study).

## What it implies for Pixels2Actions (inference, to test with our own analytics)

1. **Lead with amazement and curiosity, never outrage.** Awe, surprise and positive high-arousal emotion (a robot
   doing something nobody expected) are the high-arousal emotions that fit an honest science channel. Low-arousal
   delivery is what hurts.
2. **Tell each story as a story, not a digest item:** a problem, turns (3+ moments of change), stakes and a payoff.
   This is the strongest content signal for science videos, and news-digest framing risks views without watch time.
3. **Put the most surprising moment in the first seconds**, and pay the title's promise off in the first 30 s (the
   Intro metric; Tong 2020).
4. **Keep one recognisable format and voice in one lane.** Channel effects dominate.
5. **Prefer open-ended curiosity over "you'll learn X" promises.** Add practical value only when there's a real reason
   for the viewer to tell someone.
6. **Judge over 90 days, weekly cadence is fine, and length needn't change.**
