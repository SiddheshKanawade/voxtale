# How a new viewer learns who we are: research (2026-11-15)

**Question from the user:** the episode opens with a cold open teaser ("This week, …"), then the 1.6 s brand sting, then
the stories. The channel's name is first spoken in the outro ("That's Pixels to Actions…"). A new viewer never hears
what Pixels to Actions does. How do comparable channels handle this, and what does the evidence say?

**Rule for this doc:** sections 1 and 2 report only what was observed or what a source says. Our own reading is
labelled as such. The proposals are all in section 4, "Options for us", and each one points back to the evidence.

**Our current state** (from `episodes/2026-11-{01,08,15}-weekly/episode.json` and the skill's Format section):

- Intro (about 20 s): story teasers. In 2026-11-15 the last line is "Five stories, in plain English."
- Then the 1.6 s sting with the logo and sonic logo. No words.
- The top-bar lockup (logo plus "Pixels2Actions") is on screen in every beat. The name is therefore *visible* from
  the start, just never *said*.
- Outro, identical in all three episodes: "That's Pixels to Actions. Every source is linked in the description. If
  this was useful, subscribe, and I'll break down next week's stories the same way."

---

## 1. Competitor observation

**Method.** We took the latest long-form uploads from the `/videos` tab of 15 channels in `channels.md`,
`competitors.md` and `faceless-channels.md`, between 2026-09-01 and 2026-11-15, 1–3 videos each: 35 videos in all.

- **Getting the words.** YouTube auto-captions returned HTTP 429 after 3 videos. For the other 32, we downloaded only
  the first 80 s of audio and transcribed it locally with whisper.cpp (`ggml-base.en`), the same fallback
  `chatter.py` uses. Timings are ±2 s.
- **Checking what's on screen.** For 6 videos we pulled frames every 3 s from the first 60 s at 240p, to see whether
  the name appears on screen. All audio and video was deleted after analysis.
- **Limits.**
  - "Not said in the first 80 s" means exactly that. The channel may say its name later.
  - On-screen names were checked only in the 6 videos marked 👁.
  - Whisper misspells names ("Dr. Kohaa" is Dr. Károly Zsolnai-Fehér).
  - What channels do is not proof that it works. These are popular channels, and we only see their choices, not the
    effect of those choices.

| Channel | Video (upload date) | Name spoken at | Intro line or promise, quoted (≤15 words) | Format of the first ~60 s | Who the opening addresses |
|---|---|---|---|---|---|
| Two Minute Papers 👁 | [gPNiuyYirfw](https://www.youtube.com/watch?v=gPNiuyYirfw) (Nov 12) | **101 s** | "Dear fellow scholars, this is Two Minute Papers with Dr. [Károly]" | Cold open with the story's problem for 100 s, then the catchphrase. No on-screen name in the first 60 s | Fixed greeting; neutral |
| Two Minute Papers | [vug73goPJaM](https://www.youtube.com/watch?v=vug73goPJaM) (Nov 10) | Not said (captions, full video) | At 109 s: "hold on to your papers, fellow scholars" | Straight into the topic | Catchphrase only |
| Two Minute Papers | [x7iwI6WfWS0](https://www.youtube.com/watch?v=x7iwI6WfWS0) (Nov 14) | Not said | No speech; a song from 55 s | Music video (atypical) | n/a |
| AI Explained 👁 | [583ThJIqQ7k](https://www.youtube.com/watch?v=583ThJIqQ7k) (Nov 10) | Not said in 80 s | "if you only have a spare few seconds, the TLDR is this" | Hook (what I read) → TL;DR → "let's get started" at ~49 s. Sources on screen, no name | Neutral |
| AI Explained | [0Kw6_DTZPrY](https://www.youtube.com/watch?v=0Kw6_DTZPrY) (Nov 2) | Not said in 80 s | "this counts as a quiet few days by AI standards" | Spoken overview of the week's topics, about 60 s | Neutral |
| bycloud | [rAxTxqIHh9E](https://www.youtube.com/watch?v=rAxTxqIHh9E) (Nov 10) | Not said in 80 s | none | Straight into the topic | Neutral |
| bycloud | [2K_Rm8cho1I](https://www.youtube.com/watch?v=2K_Rm8cho1I) (Nov 3) | Not said in 80 s | none | Straight into the topic | Neutral |
| AI Revolution (faceless) | [ua8OVSw2wYU](https://www.youtube.com/watch?v=ua8OVSw2wYU) (Nov 14) | Not said in 80 s | "Stick around till the end … let's talk about it" | Story teasers ~46 s, then the stories. Closest to our structure | Neutral |
| AI Revolution | [HPCayerH1Jw](https://www.youtube.com/watch?v=HPCayerH1Jw) (Nov 12) | Not said in 80 s | "the one we already took apart in a full video" | Straight into the topic | Nods to returning viewers |
| PRO ROBOTS (faceless) 👁 | [DEIEYZqRAxo](https://www.youtube.com/watch?v=DEIEYZqRAxo) (Nov 11) | Not said in 80 s | "This story sounds like a spy scandal, but the truth is way more interesting" | Cold-open questions ~55 s. **Corner logo watermark on every frame** | Neutral |
| PRO ROBOTS | [8_DJd_YAqrk](https://www.youtube.com/watch?v=8_DJd_YAqrk) (Oct 31) | Not said in 80 s | none | Straight into the topic | Neutral |
| PRO ROBOTS | [C8ZrIDdG6b4](https://www.youtube.com/watch?v=C8ZrIDdG6b4) (Oct 24) | Not said in 80 s | none | Straight into the topic | Neutral |
| AI News (faceless) 👁 | [_TqaR5k6QEQ](https://www.youtube.com/watch?v=_TqaR5k6QEQ) (Nov 12) | **3 s** | "Today on AI News, we'll break down Google's new robotics model" | One-line hook → name + agenda → "But first" at 14 s. No on-screen name; a subscribe animation at ~48 s | Neutral |
| AI News | [2XHmajAeb0o](https://www.youtube.com/watch?v=2XHmajAeb0o) (Nov 13) | **45 s** | "So today on AI News, we'll show you how to make…" | ~44 s demo cold open + sponsor line, then name + agenda | Neutral |
| AI News | [_OsrV4aHMOo](https://www.youtube.com/watch?v=_OsrV4aHMOo) (Nov 11) | **60 s** | "Today on AI News, we'll walk you through it step by step." | Demo cold open + sponsor, then name + agenda | Neutral |
| Kevin Wood | [K09496DPiBE](https://www.youtube.com/watch?v=K09496DPiBE) (Nov 13) | Not said in 80 s | none | First-person vlog, straight in | Neutral |
| Kevin Wood | [0xWFDAxSCVM](https://www.youtube.com/watch?v=0xWFDAxSCVM) (Nov 12) | Not said | none (music only) | Demo reel | n/a |
| Matt Wolfe | [G_Jrexy_5Bg](https://www.youtube.com/watch?v=G_Jrexy_5Bg) (Nov 13, weekly "AI News") | Not said in 80 s | "This week has to be the wildest week I've ever experienced" | Hook → "my quick recap" at ~28 s | Neutral |
| Matt Wolfe | [2TM8W66NI_8](https://www.youtube.com/watch?v=2TM8W66NI_8) (Nov 9) | Not said in 80 s | "let's go ahead and dig right in here" | Hook + context, ~52 s | Neutral |
| AI Search | [woWo80dPkVk](https://www.youtube.com/watch?v=woWo80dPkVk) (Nov 15, weekly "AI NEWS") | Not said in 80 s | "AI never sleeps. And this week has been absolutely insane." | Rapid list of the week's stories, ~55 s, then "let's jump right in" | Neutral |
| AI Search | [S7lpCyZSdJQ](https://www.youtube.com/watch?v=S7lpCyZSdJQ) (Nov 11) | Not said in 80 s | "In this video, we're going to go over all the ridiculous things" | Hook + agenda ~25 s | Neutral |
| AI Search | [Er0740eVIEw](https://www.youtube.com/watch?v=Er0740eVIEw) (Nov 12) | Not said in 80 s | "So in this video I'm going to go over…" | Hook + agenda ~25 s | Neutral |
| Matthew Berman | [FCmLuTENOp0](https://www.youtube.com/watch?v=FCmLuTENOp0) (Nov 11, "AI NEWS") | Not said in 80 s | none | Straight into the first story | Neutral |
| Matthew Berman | [Sjdvu4nHfDk](https://www.youtube.com/watch?v=Sjdvu4nHfDk) (Nov 12) | Not said in 80 s | "if you ever wanted to learn about this stuff, here's the right video" | Introduces the guest, then the interview | Neutral |
| TheAIGRID (faceless) | [qCE5A2bWZz8](https://www.youtube.com/watch?v=qCE5A2bWZz8) (Nov 14) | Not said in 80 s | "So in this video, I'm going to break down…" | Cold open ~53 s → "point number one" at ~63 s | Neutral |
| TheAIGRID | [VMraE40YKrU](https://www.youtube.com/watch?v=VMraE40YKrU) (Nov 12) | Not said in 80 s | "So in this video, I'm going to show you…" | Cold open ~40 s → "point number one" at ~44 s | Neutral |
| Wes Roth | [Sc6TriRAGnI](https://www.youtube.com/watch?v=Sc6TriRAGnI) (Nov 13) | Not said in 80 s | none | Montage cold open | Neutral |
| Wes Roth | [h76wUNmRqxc](https://www.youtube.com/watch?v=h76wUNmRqxc) (Nov 10) | Not said in 80 s | none | Straight in | Neutral |
| Fireship 👁 | [SOwUWPtXQ9I](https://www.youtube.com/watch?v=SOwUWPtXQ9I) (Nov 12) | **41 s** | "It is November 12, 2026, and you were watching the Code Report." | Cold open ~32 s → "In today's video, we'll look at…" → dated name line → **"THE CODE REPORT" title card** (~42–45 s) | Neutral, fixed format |
| Fireship | [1FurWDiK9A4](https://www.youtube.com/watch?v=1FurWDiK9A4) (Nov 10) | **59 s** | "It is November 10th, 2026, and we're watching the Code Report." | Same structure | Same |
| Fireship | [pf5tXJZM1Rw](https://www.youtube.com/watch?v=pf5tXJZM1Rw) (Nov 9) | **53 s** | "It is November 9th, 2026, and we're watching the Code Report." | Same structure | Same |
| ColdFusion 👁 | [pjNnDQgAvOw](https://www.youtube.com/watch?v=pjNnDQgAvOw) (Nov 13) | **0 s** | "Hi, welcome to another episode of ColdFusion." | Name first, then the story. **Logo at 0 s, wordmark ~3 s, corner watermark on every frame** | "another episode" assumes a series |
| ColdFusion | [FsEbLPqeo7k](https://www.youtube.com/watch?v=FsEbLPqeo7k) (Nov 11) | Not said in 80 s | none | Narrated cold open | Neutral |
| ColdFusion | [Sb1_j9A25nQ](https://www.youtube.com/watch?v=Sb1_j9A25nQ) (Nov 2) | Not said in 80 s | none | Archive-quote cold open | Neutral |
| Welch Labs | [QgH9sr7G13Q](https://www.youtube.com/watch?v=QgH9sr7G13Q) (Sep 1) | Not said in 80 s | none | Narrated problem setup | Neutral |

### What the table shows (counts, not judgments)

- **The name is spoken in the first ~60 s in 7 of 35 videos, from only 3 channels.** AI News (3 of 3, at 3–60 s),
  Fireship (3 of 3, at 41–59 s) and ColdFusion (1 of 3, at 0 s). Two Minute Papers names itself at 101 s in 1 of 3.
  The other 11 channels never spoke their name in the first 80 s of any sampled video.
- **When the name is said, it comes after a hook, in a fixed sentence that pairs the name with the format.** AI News
  says "Today on AI News, we'll…". Fireship says "It is <date>, and you're watching the Code Report." Two Minute
  Papers says "Dear fellow scholars, this is Two Minute Papers with…". ColdFusion is the only one to open on the name.
- **No sampled video says "welcome back"**, and none states a channel-level promise ("this channel explains X every
  week"). Every promise line we heard is about *this video*: "Today … we'll break down", "In this video I'm going to",
  "the TLDR is this".
- **The visual identity is constant, the spoken one isn't.**
  - PRO ROBOTS and ColdFusion keep a corner watermark on every frame.
  - Fireship shows a "THE CODE REPORT" title card with the date.
  - Our top-bar lockup plays the same role.
- **Weekly roundups** (Matt Wolfe's "AI News: …", AI Search's "…: AI NEWS", Matthew Berman's "…: AI NEWS") label the
  series in the *title*, not the narration.
- **Two Minute Papers' 2026 Shorts are short interview clips.**
- **Fireship changed a title between our two fetches**: SOwUWPtXQ9I first showed "The backdoor that got banned GPUs
  into China…", later "Jensen Huang just pulled out of Nvidia…". Most likely Test & Compare or an edit; unverified.

### Shorts (latest from each `/shorts` tab, 7 transcribed in full)

| Channel | Short | Date | Channel/series named? | Promise or CTA said |
|---|---|---|---|---|
| Matt Wolfe | [X3r-mZtPfP8](https://www.youtube.com/shorts/X3r-mZtPfP8) | 2026-11-15 | No | At ~51 s: "click that follow button to see the coolest tech in AI news every single day" |
| Matt Wolfe | [iTe0IB5Ugf8](https://www.youtube.com/shorts/iTe0IB5Ugf8) | 2026-11-12 | No | At ~2:19: "click that follow button to stay looped in with all the latest AI news and coolest tech every day" |
| PRO ROBOTS | [NQJJpIM12fY](https://www.youtube.com/shorts/NQJJpIM12fY) | 2026-09-03 | No (first 90 s) | none heard |
| Two Minute Papers | [82m7YqosdgU](https://www.youtube.com/shorts/82m7YqosdgU), [iz7elwbqZVI](https://www.youtube.com/shorts/iz7elwbqZVI) | 2026-06 | No | none (interview clips only) |
| AI News | [KY1UvKQcJDY](https://www.youtube.com/shorts/KY1UvKQcJDY) | 2025-09-13 | No (said); title ends "\| AI NEWS" | none heard |
| Fireship | [oRBvLbdMKM0](https://www.youtube.com/shorts/oRBvLbdMKM0) | 2024-09-09 | No | none |

**Pattern:**
- No sampled Short says the channel's name.
- The only channel-level promise is Matt Wolfe's. It is folded into the single end CTA and states what the channel
  posts and how often ("AI news every single day").
- Several channels have nearly stopped Shorts: Fireship's latest is from 2024 and AI Revolution's from 2023. Sample: 5
  channels.

---

## 2. Published guidance and evidence

Strength labels follow the house style (`research-shorts-retention.md`):
- **Official:** YouTube's own Help Center or blog.
- **Study:** peer-reviewed, or a large sample with a stated method.
- **Ad research:** platform or industry research on ads, not creator videos.
- **Practitioner:** blogs, vendors, creators, with no method.

| # | Finding | Source | Strength |
|---|---|---|---|
| G1 | "Make compelling content first … share your channel branding later." "A flashy intro may look cool, but it's not the star of the video." "Make it clear what your video is about early on." It cites TV shows that run "a great two-minute scene followed by the opening credits" | YouTube Creator Playbook tips: the first 15 seconds (Ryan Nugent, 2011-08-27), [blog.youtube](https://blog.youtube/creator-and-artist-stories/youtube-creator-playbook-tips-first-15) | Official, but old (2011) |
| G2 | "Don't start your long-form video with slow channel graphics, logos, or unrelated updates." "The first 5 to 10 seconds of your long video should directly address the topic … promised in the Short" | "How to convert YouTube Shorts views into long-form channel growth" (Julia Lee Harter, 2026-07-14), [blog.youtube](https://blog.youtube/creator-and-artist-stories/youtube-related-videos-traffic-guide/) | Official, current |
| G3 | Four hooks: tell them what they'll see ("Within the first five seconds the viewer knows exactly what this video will be"), cold open, open on a question, and "brand with a bang" where "the intro feels integrated into the content itself" and new viewers quickly learn what the show is | "Four tips to hook your viewers on YouTube" (2014-06-24), [blog.youtube](https://blog.youtube/creator-and-artist-stories/four-tips-to-hook-your-viewers-on/) | Official, old |
| G4 | The Intro metric is the share still watching after 30 s. A high value can mean "the content in the first 30 seconds matched the viewer's expectation of the video's thumbnail and title". Fixes: retitle/rethumbnail, or "modify the first 30 seconds … experiment with different styles" | YouTube Help, key moments for audience retention, [answer/9314415](https://support.google.com/youtube/answer/9314415?hl=en) | Official (a metric definition, not a rule on branding) |
| G5 | Channel trailer: "offers a preview of your channel so viewers can learn more and subscribe". Shown to people who haven't subscribed. "Once a viewer watches your channel trailer, it won't be shown to them again". A separate featured video can be set for returning subscribers. No ads on the trailer by default | YouTube Help, Customize your channel layout, [answer/3219384](https://support.google.com/youtube/answer/3219384?hl=en) | Official, current |
| G6 | Trailer tips: a clear subscribe CTA, "when in doubt 'show' what your channel is about, don't 'tell'", "Indicate the type of videos you upload and when you upload them", and "shorter is better" | "Convert viewers into subscribers with your Channel Trailer" (Andres Palmiter, 2013-03-18), [blog.youtube](https://blog.youtube/news-and-events/convert-viewers-into-subscribers-with/) | Official, old |
| G7 | Channel description: "Your first 45 characters should say what your channel is about". Upload schedule further down. The channel name should be "descriptive and succinct" | "Three tips to help potential viewers find your channel" (2012-09-06), [blog.youtube](https://blog.youtube/news-and-events/three-tips-to-help-potential-viewers/) | Official, old (the layout has changed since) |
| G8 | Video descriptions: use the first lines to "describe your video because it's what viewers will see first". Put channel information in the expanded part. Upload defaults can fill "basic channel information in all of your videos on upload". Link playlists | YouTube Help, Tips for video descriptions, [answer/12948449](https://support.google.com/youtube/answer/12948449?hl=en) | Official, current |
| G9 | Related video on Shorts plus a verbal CTA in the final 5 s ("Get the full step-by-step breakdown by clicking the link right here") | Same as G2; also `research-shorts-offers.md` B10/B12 | Official |
| G10 | **TV ads:** brand presence shapes zapping. Ads were skipped more when the brand was central on screen; brand size made no difference. Showing the brand in short **pulses**, with the same total exposure, "decreases commercial avoidance significantly". Data: 31 ads, about 2,000 viewers, eye tracking plus a controlled experiment | Teixeira, Wedel & Pieters, *Marketing Science* 29(5), 2010, [abstract (IDEAS)](https://ideas.repec.org/a/inm/ormksc/v29y2010i5p783-804.html) | Study, strong for TV ads; only indirect for creator videos |
| G11 | Google's ABCD guidance for YouTube *ads*: show the brand or product in the first 5 s, by voice as well as image, "so viewers immediately understand who's speaking to them". Third-party summaries note that early-branded ads are also skipped more | Think with Google ABCD ([playbook PDF](https://thinkwithgoogle.com/_qs/documents/8472/ABCD_Complete_V7b_HR_1.pdf), [branding page](https://www.thinkwithgoogle.com/intl/en-145/marketing-strategies/video/youtube-ads-branding-strategies)) | Ad research. The brand-lift figures come from Google/Kantar; transfer to creator videos is untested |
| G12 | "Welcome back"/greeting intros waste the seconds when viewers decide whether to stay, and new viewers "won't feel welcomed back". Keep branded intros ≤5 s or cut them | vidIQ ([intros](https://vidiq.com/blog/post/youtube-video-intros/), [retention](https://vidiq.com/blog/post/increase-audience-retention-youtube/)), Subscribr, Renderforest ([intro length](https://www.renderforest.com/blog/how-long-should-youtube-intro-be)); also `brand.md` | Practitioner (no data published) |
| G13 | Trailers of 30–90 s that assume the viewer doesn't know you, show the regular format, and end on a subscribe CTA. The returning-subscriber slot is for something timely | vidIQ ([trailer](https://vidiq.com/blog/post/youtube-channel-trailer/)), Descript, Wyzowl and others | Practitioner |

**Not found:**
- No study, official or independent, measures intro length, branded stings, "welcome back" lines or channel-identity
  statements against YouTube retention or subscriptions. The circulating numbers ("30–50% drop from animated intros",
  "78% retention with matching hooks", "half drop at 30 s, Fullscreen") have no traceable method, so we don't use them.
- The often-repeated claim "Creator Academy says the first 15 seconds are the most critical" could not be found on a
  YouTube page.

### Our reading (not a finding)

The guidance agrees that **content comes first and branding follows**:
- G1 and G2 are explicit; G3 and G4 say the same thing indirectly.
- None of the guidance says the channel must be named. The goal is that the viewer knows quickly **what this video
  is**.

For *who the channel is*, YouTube points to other surfaces:
- the trailer, which is shown once to non-subscribers (G5, G6);
- the description, which puts the video first and the channel info below (G8);
- the About text, whose first 45 characters say what the channel is (G7).

The ad studies (G10, G11) suggest that a brief brand mention that comes back later works better than one long brand
block. They are about ads, and only 3 of 15 competitors speak their name early.

---

## 3. Where we stand against this

| Element | Now | Evidence it touches |
|---|---|---|
| Cold open with story teasers, ~20 s | Matches G1/G3 and what most competitors do (AI Revolution, AI Search, AIGRID) | Supported |
| 1.6 s wordless sting | Short. G2 warns only against "slow" graphics; Fireship and ColdFusion use title cards | Neutral |
| Name visible in the top bar throughout | Like the PRO ROBOTS/ColdFusion watermarks | Supported as a pattern |
| Name never said until the outro; no "what we do" line anywhere | 12 of 15 competitors don't say their name in the first 80 s either. But no competitor's outro was checked, and nobody states a channel promise | The gap the user raised is real. Competitors leave it open too |
| Outro "That's Pixels to Actions…" | Names the channel but not what it does or how often | G6 (type of videos + when) is not covered |

---

## 4. Options for us (RECOMMENDATIONS, not findings)

Durations assume our narration rate of about 150 words a minute (2.5 words a second, as in the skill). TTS writes "A.
I." and "Pixels to Actions" as in the current outro. Nothing has been changed: no episode, skill or code was edited.

### 4.1 Episode intro

| Option | Exact narration and placement | Cost | Tied to |
|---|---|---|---|
| **A. Name + promise after the teaser (Fireship / AI News pattern)** | Replace the closing teaser line "Five stories, in plain English." with: **"This is Pixels to Actions: the week's robot and A.I. news, in plain English."** (13 words). Then the sting as now | About 5 s, replacing about 2 s: **+3 s net**. The name lands at about 18–20 s, still inside the 30 s Intro window (G4) | G1 (content first, then brand), the observed "hook → name + format line" sentence (AI News 3–60 s, Fireship 41–59 s), G10 (a brief brand pulse, not a block), G11 (spoken brand, ads only) |
| **B. Dated series signature over the sting** | Just before the sting: **"It's November fifteenth. This is Pixels to Actions."** (8 words). The sting follows as now. Keep "Five stories, in plain English." | About 3 s: **+3 s** | The Fireship pattern (3 of 3 videos), a fixed line that becomes the recognition cue like Two Minute Papers' greeting. It names the channel and dates it but doesn't explain it |
| **C. Keep the intro as is; solve this elsewhere** | No change to the intro. Put the promise in the outro (4.2), the description (4.4) and the trailer (4.5) | 0 s | 12 of 15 competitors don't say their name in the first 80 s. G2 says don't front-load channel graphics. G5–G8 put "who we are" on the channel surfaces |

*Our lean (opinion):* **A**. It's the only option that answers the user's concern inside the video at near-zero cost,
and it follows the one sentence pattern the competitors who do name themselves share. Check it after 2–3 episodes
against the Intro metric (G4). If the 30 s retention drops relative to the earlier episodes, fall back to C.

### 4.2 Outro

| Option | Exact narration | Cost |
|---|---|---|
| **A. Add what we do + cadence** | **"That's Pixels to Actions. Every week, the biggest stories in robots and physical A.I., explained in plain English. Every source is linked in the description. Subscribe, and I'll see you next week."** | About +3 s over the current 23-word line |
| **B. Keep it, change only the CTA reason** | "That's Pixels to Actions. Every source is linked in the description. **Subscribe for next week's robot and A.I. stories, in plain English.**" | About ±0 s |

Tied to:
- G6, "indicate the type of videos you upload and when". Our outro currently says neither.
- The single-CTA rule (`research-shorts-offers.md` A2.12), which both options keep (Subscribe only).

### 4.3 Shorts

| Option | Exact narration (last ~5 s, replacing the current CTA) | Cost |
|---|---|---|
| **A. The CTA carries the promise (Matt Wolfe pattern)** | **"The full story is in this week's Pixels to Actions episode: robot news, in plain English. Tap the link above."** | +1–2 s over a bare CTA |
| **B. Name nothing in the voice, put it in the packaging** | Keep the current CTA ("The full breakdown is in this week's episode. Tap the video link above.") and add the series label in the title or first description line, e.g. "… \| Pixels2Actions weekly" (the AI News "\| AI NEWS" pattern) | 0 s |

Tied to:
- G9 (Related video plus a verbal CTA in the final 5 s);
- the Shorts table: no sampled Short says the channel's name, and Matt Wolfe's is the only one with a channel promise,
  inside the CTA;
- the one-CTA rule.

The Shorts UI already shows the channel handle and the related-video link (G9 / B10), so naming the channel aloud is
optional.

### 4.4 Description template (each episode)

Following G8 (video first, channel info in the expanded part, upload defaults):

```
<1–2 lines: what this episode covers, e.g. "NVIDIA's GR00T 2 learns from videos of people, Google's robots in a
Mercedes-Benz plant, and three more physical-AI stories, explained with their sources.">

Chapters
0:00 This week
...

Sources (every claim, in story order)
...

About Pixels2Actions: every week, the top physical-AI stories (robots, humanoids, world models, self-driving)
explained in plain English for curious viewers, with every source linked. New episode weekly; Shorts in between.
```

Set the "About Pixels2Actions" paragraph as the upload default (G8) so it can't be forgotten. Cost: none in the video.

### 4.5 Channel About and trailer

- **About, first 45 characters** (G7): **"Weekly physical-AI news, in plain English."** (42 characters). Then: "Robots,
  humanoids, world models and self-driving, the week's top stories explained with every source linked. New
  episode every <weekday>." Fill in the weekday once the cadence is fixed (G6/G7: say when you upload).
- **Trailer, option A (cheapest):** set the strongest episode's first ~20 s plus outro as the non-subscriber trailer.
  Cost: an edit, no new script. Rests on G6 "show, don't tell"; reusing an existing video is a practitioner tip (G13).
- **Trailer, option B (dedicated, about 30–40 s, ~80 words):** 3–4 of the best clips from past episodes, each with one
  line, then:

  > **"Every week, Pixels to Actions picks the biggest stories in robots and physical A.I.: what happened, how it works
  > in plain English, the evidence, and a reality check. Every source is linked. New episode every <weekday>.
  > Subscribe."**

  It names our four fixed sections (`brand.md`), so a new viewer hears the format. Rests on G5 (shown once, to
  non-subscribers), G6 (type, cadence, subscribe CTA, shorter is better) and G13.
- **Returning-subscriber slot** (G5): the latest episode, updated weekly.

### How to tell whether it worked

- **Intro metric** (G4) per episode, before and after the change. Compare like with like: similar length, and YouTube's
  "typical" benchmark.
- **New vs returning viewers and subscribers gained**, per episode, in YouTube Studio.
- **Trailer:** subscribers from the channel page (Studio, traffic source "Channel pages").

None of this has public benchmark data, so our own numbers will decide it (as in `attention.md`).
