# What holds viewers on short-form vertical video: evidence review

Compiled 2026-11-15 for the Pixels2Actions Shorts skill (one story per Short, which sends viewers to the weekly episode).

**Strength labels**
- **strong**: peer-reviewed, or a large sample with a disclosed method and a measured outcome close to what we care about.
- **moderate**: peer-reviewed but small or indirect (wrong platform, wrong outcome, or correlational), or platform-published with a stated sample.
- **weak**: platform or industry claim with no disclosed method, or a secondary report of one.
- **practitioner**: vendor blog or creator opinion with no method.

**What the evidence base looks like.** No peer-reviewed study measures first-seconds *retention* on Shorts or TikTok against creative features. Platforms do not publish that data. The best direct evidence is:
- a swipe-timing user study (Dashlet);
- two large YouTube channel datasets (Violot et al.; Rajendran et al.);
- MOOC video engagement at scale (Guo et al.);
- platform-commissioned *ad* research from Facebook/Nielsen, TikTok and Google/Kantar. That research measures recall, not retention.

Most of the precise percentages circulating online are untraceable (see "Unverified and debunked numbers" at the end).

---

## 1. The opening (first 1–3 seconds)

| Source | Setting / N | Finding | Strength |
|---|---|---|---|
| Li et al., "Dashlet: Taming Swipe Uncertainty for Robust Short Video Streaming" (NSDI 2024), https://arxiv.org/abs/2204.12954 | IRB user studies in a TikTok-like interface, 500 popular TikTok videos, 20 min sessions. 25 students (3,069 swipes) and 133 MTurk workers (15,344 swipes). | Swipes are bimodal: "users are most likely to swipe either soon after the video playback begins or at the tail end of the video." For MTurk users, "29% and 42% of swipes … are within the first 20% or last 20% of videos". Only 6% of campus swipes fell in the 60–80% band. Swipe profiles are a property of the *video* and stable across user groups (KL 0.2 median). One video had 60% of swipes in its first 20%, another had 80% in its last seconds. | moderate (lab, small N, but measures real swipe timing) |
| Facebook + Nielsen (2015–16), reported by AdExchanger, https://www.adexchanger.com/digital-tv/facebook-adds-viewable-video-options-automated-captions-autoplay/ and Marketing Dive, https://marketingdive.com/news/brand-lift-happens-in-less-than-1-second-of-video-study-finds/377333 | Facebook feed *video ads*. Method and N not published. | "47% of a video's value in the news feed" (awareness and conversions) comes "in the first three seconds", and 74% by 10 s (Smart Insights summary). Ads watched for under 3 s still gave +47% ad recall and +32% brand awareness. | weak (platform-commissioned, no method, ads, 2015) |
| TikTok Marketing Science, internal 2021 meta-analysis, cited in TikTok for Business blog "Understanding attention on TikTok" (MediaScience), https://ads.tiktok.com/business/en/blog/mediascience-study-brands-memorable-tiktok | TikTok *ads*. The MediaScience lab part was n=343. The page was blocked (HTTP 451), so this comes from a search snippet. | "50% of the impact from a TikTok ad is realized in the first 2 seconds". The first 6 s capture "90% cumulative impact on Ad Recall". Ads on screen ≤6 s still gave 38% of the recall of 20 s+ views. | weak (platform, ads, recall not retention, not verified first-hand) |
| Google/Kantar ABCD playbook, https://thinkwithgoogle.com/_qs/documents/15987/ABCDs_PDFPlaybook_April2022_Final.pdf. Kantar "Short and Long of ABCDs" (2021) | YouTube *ads*, Kantar n≈11,000 ads. | Ads following the ABCDs (Attention, Branding, Connection, Direction) saw +30% short-term sales likelihood. Attention guidance includes "two or more shots in the first five seconds", tight framing, and opening on people. | moderate for the package effect, weak for any single tactic (not isolated) |
| Google, YouTube Shorts ABCDs one-sheet, https://services.google.com/fh/files/misc/formarketingshortsabcdsonesheeters.pdf | Guidance only, no data. | "Announce yourself with audio", "Be unavoidable", "Use tight framing", "Hook them with the benefits that matter most", "Anchor with supers" (on-screen text). | practitioner (platform guidance, no numbers) |
| PaperTok (arXiv 2601.18218), https://arxiv.org/pdf/2601.18218 | Small formative study with researchers and creators. | Defines the hook as the opening 2–5 s, built from "questions, surprising facts, or relatable pain points". | practitioner/weak (definition, not outcome) |
| OpusClip blog, https://www.opus.pro/blog/tiktok-length-format-retention-data | "500 videos". Method not disclosed. | Question hooks are "+23% retention". Skipping intros gives "30–40%" better results. Payoff in the first 15 s gives "~20% higher retention". | practitioner |

**Takeaway.** The one direct measurement (Dashlet) shows the early drop-off is real and specific to each video: some videos lose most viewers in the first fifth and others almost none. Nothing peer-reviewed isolates *which* opening features cause the difference (motion onset, text on frame 1, starting mid-action). The "first 2–3 seconds" emphasis rests on platform ad-recall research, not on retention data.

## 2. Length vs completion and retention

| Source | Setting / N | Finding | Strength |
|---|---|---|---|
| Guo, Kim & Rubin, "How video production affects student engagement" (ACM L@S 2014), https://pg.ucsd.edu/publications/edX-MOOC-video-production-and-engagement_LAS-2014.pdf | 6.9M video sessions, 4 edX courses. | "Shorter videos are much more engaging." "The median engagement time is at most 6 minutes" regardless of length. Recommendation: "chunks shorter than 6 minutes". | strong for educational video. The setting is long-form MOOC, not feed video. |
| Habibi & Salim, PLOS ONE 2021, https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0248507 | 1 science creator, 40 TikToks: 15 s quick-facts vs 40–60 s lecture vs 40–60 s experiment. | Average % watched was "significantly higher for the short lecture-style video at 69%". Long lecture videos drew more comments (0.4% of reach). Experiment videos drew the most views. | moderate/weak (n=40, one account, 2020) |
| Violot et al., "Shorts vs. Regular Videos on YouTube" (ACM WebSci 2024), https://arxiv.org/abs/2403.00454 | 70,712 channels, 9.88M Shorts and 6.86M regular videos (2021–22). | Shorts get ~4× mean views (~6× by late 2022), and 1.4× likes/view by late 2022, but fewer comments/view. In **Science & Technology** the Shorts/regular view ratio (2.30–4.15) stayed below the platform-wide ratio in every period. **Education** started above it (5.28) and then fell below it. | strong (large, descriptive. Reach, not retention.) |
| Zhan et al., "Deconfounding Duration Bias in Watch-time Prediction" (Kuaishou), https://arxiv.org/abs/2206.06003 | Kuaishou production recommender. | Duration "concurrently affects video exposure and watch-time prediction". Raw watch time and completion are confounded by length, so a shorter video completes more by construction. | moderate (methodological, but explains why completion-vs-length charts mislead) |
| OpusClip blog, https://www.opus.pro/blog/tiktok-length-format-retention-data | Method not disclosed. | "21–34 sec … highest completion rates, averaging 62%" vs 48% for videos over 60 s. | practitioner |
| Rewatch-rate calculators (e.g. https://insights.ttsvibes.com/tiktok-rewatch-rate-statistics) | Modeled estimates. | Loop rates fall from about 42% at 7 s to near 0 at 10 min. The site says these are estimates, not platform data. | practitioner |

**Takeaway.** No study gives a retention-optimal length for short-form explainers. The robust facts are these:
- Completion falls mechanically with length (Kuaishou).
- Learners' engagement caps at roughly 6 min even on long-form (Guo).
- In a tiny TikTok science test, 15 s quick facts kept about 69% watched while longer pieces earned more comments.
- Shorts are relatively weaker on reach in Sci/Tech than in other categories (Violot).

The "21–34 s" figure is vendor lore.

## 3. Pacing, cuts, on-screen text, captions, voice vs music

| Source | Setting / N | Finding | Strength |
|---|---|---|---|
| Guo et al. 2014 (above) | 6.9M sessions. | "Videos where instructors speak fairly fast and with high enthusiasm are more engaging". Khan-style drawing beats slides: "Introduce motion and continuous visual flow". Talking head interleaved with slides is more engaging than slides alone. | strong (education) |
| Lang et al., "The effects of production pacing and arousing content on the information processing of television messages", J. Broadcasting & Electronic Media 1999, https://doi.org/10.1080/08838159909364508 | Lab experiments (LC4MP tradition). | Faster pacing (more cuts) raises arousal and orienting attention. Fast pacing combined with arousing or dense content overloads processing and lowers recognition memory. | moderate (lab, TV, but directly about cut rate × learning) |
| Verizon Media + Publicis Media survey (May 2019), https://www.nexttv.com/news/mobile-videos-often-watched-without-audio-study-finds. Survey size reported as ~5,600 US consumers by MarketingCharts. | US consumer survey (self-report). | 69% watch with sound off in public. 80% are "more likely to watch an entire video when captions are available". 37% say captions make them turn the sound on. | moderate (large survey, self-report, pre-Shorts) |
| Facebook internal tests, https://www.facebook.com/business/news/updated-features-for-video-ads (2016) | Feed video ads, method not disclosed. | "Captioned video ads increase video view time by an average of 12%." "41% of videos were basically meaningless without sound." | weak (platform, no method) |
| Adesope & Nesbit, "Verbal redundancy in multimedia learning environments: a meta-analysis", J. Educational Psychology 2012, https://doi.org/10.1037/a0027696. Mayer's redundancy principle. | Meta-analysis of lab learning studies. | Spoken plus written text gave a small learning advantage over spoken-only overall. Mayer's redundancy principle warns against adding full on-screen text to narrated *graphics* for novices. Short keyword captions with highlighted terms sit between the two findings. | moderate (learning outcomes, lab) |
| Lu & Shen, "Unpacking Multimodal Fact-Checking" (Social Media + Society 2023), https://doi.org/10.1177/20563051221150406 | 4,309 Douyin fact-check videos, automated and manual coding. | Fact-check videos have faster tempo and higher brightness than same-account and trending videos. Several audiovisual features and persuasive strategies (clickbait, humor) are associated with likes, comments and shares. Three types: long storytelling cartoons, short stimulating, short authoritative. | moderate (engagement, not retention. Abstract-level detail only.) |
| Douyin thinspiration computational study (ScienceDirect 2026), https://www.sciencedirect.com/science/article/abs/pii/S1740144526000240 | Douyin videos (abstract only). | More emphasis on overlay text vs pictures goes with *fewer* likes and comments. Text-heavy posts do better when the text is "centered, informative … congruent with the pictorial content". | weak/moderate (one niche, engagement) |
| OpusClip blog (above) | No method. | "Pattern interrupts every 4 seconds average 58% retention compared to 41%" for static talking heads. Accurate captions gave "~12% higher retention". | practitioner |

**Takeaway.**
- Captions are well supported for reach and comprehension: many viewers are sound-off, and captions add to learning when kept short.
- Fast speech and continuous visual motion help engagement.
- Cut rate has an inverted-U shape for learning: more cuts raise attention until content density overloads viewers.
- There is no peer-reviewed evidence on voiceover vs music-only Shorts. Google's Shorts guidance says "announce yourself with audio".

## 4. Loops and rewatch

| Source | Setting / N | Finding | Strength |
|---|---|---|---|
| YouTube Help, Shorts views change (31 Mar 2025), https://support.google.com/youtube/answer/10059070. Coverage: https://techcrunch.com/2025/03/26/youtube-is-changing-how-youtube-shorts-views-are-counted | Platform policy. | "Views will count the number of times a Short starts to play or replay, with no minimum watch time." The old metric was renamed "engaged views". YPP and revenue use engaged views. | strong as a fact about metrics |
| Dashlet (above) | User study. | 42% of MTurk swipes come in the last 20% of the video, so many viewers watch to the end and then leave or auto-advance. | moderate (context only) |
| Vendor blogs (OpusClip, https://www.opus.pro/blog/ideal-youtube-shorts-length-format-retention. GoFaceless, https://www.gofaceless.ai/en/blog/youtube-shorts-retention-rate-benchmarks) | No method. | Seamless loops earn replays and push "% viewed" above 100%. | practitioner |

**Takeaway.** No study isolates a looping ending. Since March 2025 a replay inflates the public view count but not engaged views. A loop's real value can only be judged from our own "% viewed" and engaged-view data. For a promo Short, a loop also competes with the call to action (CTA) to the long video.

## 5. Educational and explainer Shorts

| Source | Setting / N | Finding | Strength |
|---|---|---|---|
| Loewenstein, "The psychology of curiosity" (Psychological Bulletin 1994), https://doi.org/10.1037/0033-2909.116.1.75 | Theory review. | Curiosity is an information-gap state. It needs a gap the viewer can see and believes is closable, and enough prior knowledge to notice it. | strong (theory, widely replicated) |
| Kang et al., "The wick in the candle of learning" (Psychological Science 2009), https://doi.org/10.1111/j.1467-9280.2009.02402.x | Trivia paradigm, fMRI and behavioural, small N. | Higher curiosity before an answer predicts better memory for that answer, and people "pay" (time or tokens) to resolve it. | moderate |
| Gruber, Gelman & Ranganath (Neuron 2014), https://doi.org/10.1016/j.neuron.2014.08.060 | Lab, small N. | Curiosity states improve memory even for incidental material shown during the gap. | moderate |
| Guo et al. 2014 (above) | 6.9M sessions. | Enthusiasm, faster speech, motion and a personal feel beat polished studio slides. | strong (education) |
| Habibi & Salim 2021 (above) | 40 videos. | Demonstrations and experiments got the most views. 15 s quick facts kept the highest % watched. | moderate/weak |
| Single-creator science TikTok case study (ScienceDirect 2023), https://www.sciencedirect.com/science/article/abs/pii/S0306452223003731 | 150 videos, one creator. | Videos that summarize research papers got the most views and shares. Career and life advice got the least. | weak |
| Memetic science-TikTok study (PubMed 42533308), https://pubmed.ncbi.nlm.nih.gov/42533308/ | 1,025 videos, 113 creators (snippet only). | Imitation and humor positively predict engagement. Reappropriation predicts it negatively. | moderate/weak |
| PaperTok (above) | Formative. | Effective research Shorts open with a question, surprising fact or pain point, then pay it off. | practitioner/weak |

**Takeaway.** The best-grounded mechanism for explainers is the curiosity gap. Open a specific, answerable question in the first line and resolve it. Curiosity improves memory as well as viewing. A demonstration (watching the thing happen) beats lecture. Paper-summary content does well for science audiences (weak evidence).

## 6. Converting Short viewers to long-form

| Source | Setting / N | Finding | Strength |
|---|---|---|---|
| Rajendran et al., "Shorts on the Rise" (arXiv 2402.18208), https://arxiv.org/html/2402.18208v2 | 250 creators with ≥100k subs, before/after first Short, Wilcoxon tests. | Long-form views fell after Shorts adoption (mean −743,589/channel, p=0.011), driven by long-form-first creators (p=0.005) and channels with >5M subs. Information/education channels were "not significantly affected". | moderate (before/after, no control, confounded by platform trends) |
| Violot et al. 2024 (above) | 70k channels. | Same-channel Shorts get ~110× the views of regular videos. The paper does not measure transfer to long-form or subscribers. | strong (descriptive) |
| YouTube VP Tara Walpert Levy (June 2022), via TechCrunch, https://techcrunch.com/2022/06/15/youtube-shorts-tops-1-5b-logged-in-users-monthly-users-touted-as-feeder-to-long-form-content | Platform claim. | "Channels uploading both short and long-form content are seeing better overall watch time and subscriber growth." YouTube "didn't share specific figures". | weak |
| AIR Media-Tech, 18,000 channels (Mar 2025–Mar 2026), https://air.io/en/audience-growth/do-youtube-shorts-help-your-long-form-videos-grow-data-from-18000-channels | 18k English channels, median subscribers by Shorts share. Vendor, correlational. | Mixed-format channels had more subscribers in education (+37%), but long-only led in **science (+16% at mid-size)** and gaming. A 25–40% Shorts share was the "sweet spot". Above 55%, long-form drops. Claims Shorts-to-long conversion of 0.04% → 0.5% after setting the **Related Video** field plus a verbal CTA (measurement method not explained). | weak (large N, but vendor and correlational) |
| Metricool 2026 (via Tubefilter), https://www.tubefilter.com/2026/08/03/youtube-shorts-long-form-study-metricool/ | 799,718 videos, 71,177 accounts. Method not described. | 61% of views come from the Shorts feed. Long-form views +76% YoY but watch time only +11%. No conversion data. | weak |

**Takeaway.** Short-to-long conversion is small, unmeasured in public data, and possibly negative for big long-form channels. Education and information channels were the least harmed. The only lever with even anecdotal conversion data is the Related Video link plus an explicit spoken CTA. YouTube's own claim has no numbers.

## 7. Faceless vs presenter

| Source | Setting / N | Finding | Strength |
|---|---|---|---|
| Guo et al. 2014 (above) | 6.9M sessions. | Talking head interleaved with slides is more engaging than slides alone. 46% vs 33% of students attempted the following problem. Tight framing and eye contact matter more than studio quality. | strong (MOOC, not feed) |
| Bakhshi, Shamma & Gilbert, "Faces engage us" (CHI 2014), https://doi.org/10.1145/2556288.2557403 | 1.1M Instagram *photos*. | Photos with faces got about 38% more likes and 32% more comments. Age and gender of the face did not matter, and the number of faces did not either. | strong for still images, indirect for video |
| Google ABCD / Shorts ABCDs (above) | Ads. | "Open on people", "Give real people a real stage", tight framing. A 2015 Google study linked smiling characters in the first 5 s with brand lift. | weak/practitioner |
| Google "short-form video advertising tips" (Jul 2024), https://business.google.com/in/think/search-and-video/short-form-video-advertising-tips/ | Talk Shoppe survey n=10,003. Creative Works review of 2,000+ Shorts ads. | "73% … trust influencers more than big celebrities". No face-vs-faceless retention figure. | weak |

**Takeaway.** No study compares faceless and presenter *Shorts* on retention. The indirect evidence (MOOC, Instagram, Google ads guidance) consistently favors a human face or person on screen. A faceless channel should replace that with another salient "subject": the robot itself, ideally acting, in tight framing from frame 1. This is an inference, not a tested result.

---

## Unverified and debunked numbers

| Claim | Status |
|---|---|
| "Humans have an 8-second attention span, shorter than a goldfish" | **Debunked.** It traces to a 2015 Microsoft Canada consumer report that cited "Statistic Brain", which has no traceable study. BBC *More or Less* (2017) could find no source: https://www.bbc.com/news/health-38896790. That page could not be re-fetched here; the debunk is widely documented. |
| "You have 3 seconds" | **Unverified for retention.** It comes from Facebook's 3-second view metric and the Facebook/Nielsen "47% of value in 3 seconds" *ad-recall* claim (2015–16, no method). It is not a retention finding for Shorts. |
| "65% of viewers decide in the first 3 seconds" | Unverified. No primary source found. |
| "63% of top-CTR TikTok ads show the key message in the first 3 s" | Unverified. Repeated by blogs, and no TikTok primary source was found. |
| "85% of Facebook video is watched without sound" (2016) | Unverified. It came from anonymous publishers quoted in trade press, not a Facebook study. The documented figure is the Verizon/Publicis 69% in public (self-report). |
| "80% more likely to finish with captions" | Self-report survey item (Verizon/Publicis 2019), not measured behaviour. |
| "21–34 s is the optimal TikTok length", "pattern interrupt every 4 s", "+23% for question hooks" | Vendor numbers (OpusClip) with no method. Treat as practitioner. |
| "89% 1-second retention for pattern-interrupt hooks" (attributed to Later) | Primary report not found. Unverified. |

---

## What this means for our Shorts (INTERPRETATION, ranked by strength of evidence)

The ranking is by how well the evidence supports the design move, not by how much the move matters.

1. **Design for the early drop.** It is real and differs a lot between videos (Dashlet). We will learn ours only from YouTube Studio's "viewed vs swiped away" metric and the retention curve, so the skill should log the Studio first-seconds metric per Short and compare hook variants. *Evidence: moderate, direct.*
2. **Open a specific, answerable question (curiosity gap) in the first line, and pay it off inside the Short.** Then point to the episode for the "how/why". Don't withhold the payoff entirely: curiosity theory says the gap must look closable, and memory gains come from resolution. *Evidence: strong theory and moderate lab work, not tested on Shorts.*
3. **Captions always.** Use short burned-in keyword captions, not full transcripts over busy graphics. A large share of viewers are sound-off, captions help comprehension, and the redundancy principle warns against walls of text. *Evidence: moderate.*
4. **Fast, enthusiastic voiceover with continuous visual motion.** No dead frames and no static slides. Guo shows speech rate, enthusiasm and motion raise engagement at 6.9M-session scale. *Evidence: strong (education), indirect for Shorts.*
5. **Keep it short, but don't trust "completion %" across lengths.** Completion falls mechanically with duration (Kuaishou). The tiny TikTok science test favored about 15 s for % watched, while longer videos got more comments. Start around 20–40 s for one story and A/B on *engaged views and average view duration*, not completion. *Evidence: moderate/weak. No optimum is established.*
6. **Cut pace: change something visually every few seconds, but lower the density when the content is hard.** Lang's work suggests fast cuts with dense information hurt memory. This fits the channel's existing 5–7 s visual cap. *Evidence: moderate (TV/lab).*
7. **Show a "face" from frame 1.** Without a presenter, lead with the robot acting, in tight framing. Human-presence evidence (MOOC, Instagram, Google ads guidance) points this way, but faceless vs presenter on Shorts is untested. *Evidence: weak–moderate, indirect.*
8. **Conversion: set the Related Video link on every Short and say the CTA out loud.** Expect low conversion, well under 1%. Measure it in Studio ("traffic from Shorts" on the episode). Shorts are a discovery and reach tool, and education channels were not harmed by adding them. Keep Shorts at roughly a minority share of output (vendor heuristic, about 25–40%). *Evidence: weak.*
9. **Loops: optional, low priority.** A loop inflates the public view count since March 2025 but not engaged views, and it competes with the CTA ending. Only use it if our own data shows a "% viewed" gain. *Evidence: practitioner only.*
