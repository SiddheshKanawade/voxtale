# Competitor and reference channels

Snapshot of 2026-11-15. Method: 12 YouTube searches (20 results each, via the `youtube-search` skill's yt-dlp
commands) across robotics news, physical-AI explainers, AI research explainers and weekly AI news; company and
mainstream-news channels set aside (Tesla, Unitree, DeepMind, NVIDIA, CNBC, Bloomberg, BBC…); well-known AI
explainers added by name. For each channel: subscribers, and the median views and length of its last 30 long videos
(the `/videos` tab, so Shorts are excluded), and cadence from the dates of its 1st and 10th latest videos.

**Limits:** the stats are numbers only. Format notes come from titles and lengths, not from watching the videos.
Median views reflect each channel's own audience size, not quality.

## Closest to us: explaining research so people learn

| Channel | Subs | Median views | Median length | Uploads | What it does (from titles) |
|---|---|---|---|---|---|
| Two Minute Papers | 1.84M | 182,500 | 5.4 min | ~2/wk | One paper or result per video, short, enthusiastic ("DeepMind's AI Just Did The Impossible") |
| Welch Labs | 914K | 443,500 | 23.7 min | rare (last Sept) | Long animated deep explainers ("Yann LeCun's $1B Bet Against LLMs") |
| AI Explained | 459K | 109,500 | 22 min | ~0.6/wk | Careful analysis of model releases and lab claims |
| bycloud | 239K | 70,000 | 17.8 min | ~1/wk | Technical explainers of new methods ("The Next Scaling Axis of LLM Is… Sleeping?") |
| Yannic Kilcher | 333K | 33,000 | 37.5 min | dormant since March | Paper read-throughs |

## Same topic: robotics and physical AI

| Channel | Subs | Median views | Median length | Uploads | What it does |
|---|---|---|---|---|---|
| AI Revolution | 578K | 69,000 | 14.1 min | ~6/wk | Faceless, high-volume AI and robot news, hype titles ("SUPERHUMAN AI Is Almost…") |
| PRO ROBOTS | 373K | 27,500 | 21 min | ~1/wk | Robotics news and explainers ("This Robot Hand Walks on Its Own Fingers") |
| Kevin Wood, Robotics & AI | 82K | 5,000 | 8.7 min | ~2.4/wk | Hands-on: demos, tutorials, first tries ("I FINALLY Tried the Cybercab Robotaxi!", CoRL demo highlights) |
| AI News | 94K | 3,200 | 8.1 min | ~3.5/wk | AI news roundups, covered Gemini Robotics 2.5 this week |

## Same format: weekly AI news roundups

| Channel | Subs | Median views | Median length | Uploads | Notes |
|---|---|---|---|---|---|
| Matt Wolfe | 1.03M | 138,000 | 28.3 min | ~2/wk | "AI News: …" weekly roundup plus tutorials |
| AI Search | 774K | 370,000 | 33.2 min | ~2/wk | Long roundups of new releases; covered Fable 5.2 and LingBot-Touch this week |
| Matthew Berman | 646K | 137,000 | 19.1 min | ~7/wk | News, interviews, tests |
| Paul J Lipsky | 531K | 133,500 | 16.9 min | ~4.5/wk | Hype-heavy news ("The Craziest AI Week of The Year…") |
| TheAIGRID | 407K | 63,500 | 26 min | ~5/wk | Hype-heavy news ("…Is Absolutely RIDICULOUS") |
| Wes Roth | 335K | 171,500 | 26.9 min | ~2.7/wk | News commentary |

## Craft references (not competitors): the bar for faceless explainers

| Channel | Subs | Median views | Median length | Why it's a reference |
|---|---|---|---|---|
| 3Blue1Brown | 8.76M | 2.3M | 29.7 min | Animation carries the explanation |
| ColdFusion | 5.26M | 962,500 | 19.1 min | Faceless narrated documentary style |
| Fireship | 4.33M | 1.95M | 5.8 min | Fast, dense, short news ("Code Report") with jokes and visuals every few seconds |

## What the numbers suggest (our reading, to test, not settled)

1. **Nobody owns "learn what's new in physical AI".** The robotics channels are either hype-led and high-volume
   (AI Revolution) or small (PRO ROBOTS ~27K median, Kevin Wood ~5K). The research explainers that do teach (Two
   Minute Papers, bycloud, AI Explained) cover AI broadly and touch robots only now and then.
2. **Our format sits between two proven ones.** A weekly roundup like Matt Wolfe or AI Search, but teaching like Two
   Minute Papers. The roundups run 17–33 min; we plan 5–9 min, nearer Two Minute Papers and Fireship.
3. **Hype titles dominate the news channels**; the teaching channels use curiosity without shouting. Our
   evidence-led packaging (`docs/thumbnails.md`) is closer to the teaching channels, which is a deliberate bet.
4. **This week's overlap:** Kevin Wood (CoRL demos, Cybercab), AI News (Gemini Robotics 2.5), AI Search (Fable 5.2,
   LingBot-Touch) covered our stories. `chatter.py` already counts their views as each story's momentum.

## Next, if useful

Watch (via transcripts) 2–3 recent videos each from Two Minute Papers, AI Explained, bycloud, PRO ROBOTS and Kevin
Wood to compare hooks, structure and pacing with ours, then re-check `docs/attention.md`. Refresh this snapshot
monthly; the commands are in the `youtube-search` skill's local notes.
