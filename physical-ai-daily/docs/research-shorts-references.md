# Reference Shorts: what the good and bad ones do (2026-11-16)

12 Shorts studied privately, by the user's decision: downloaded at low resolution with yt-dlp, analysed locally,
**deleted after analysis**, never republished or reused. Only these notes remain.

**Method.** From 8 channels in our space (`channels.md`), each channel's latest Shorts (up to 40) were ranked by views
against that channel's own median, which controls for channel size. The 6 best and 6 worst performers were measured
(scene cuts with ffmpeg scene detection at 0.3; time to first audio; words per minute from a local whisper.cpp
transcript) and their first 3 seconds and full runs were tiled into frames and looked at.

**Limits.** 12 videos is anecdote, not evidence: these are patterns to test, ranked below the studies in
`research-shorts-retention.md`. Views depend on far more than the video (topic, timing, the feed's testing). YouTube
doesn't publish "viewed vs swiped away" for other people's Shorts, so views relative to the channel median are a proxy.

## The set

| Short | Channel | vs channel median | Length | Cuts per 10 s | Words/min | First 3 seconds |
|---|---|---|---|---|---|---|
| Robot hand hammering (ICRA demo) | Kevin Wood | **301×** | 14 s | 0 | ~70 | The robot hand already hammering a nail; one handheld shot |
| Tesla's robot walks like THIS | PRO ROBOTS | **156×** | 52 s | 0.2 | ~77 | The robot standing, one caption line stating the news |
| Atlas does cartwheels | PRO ROBOTS | **18×** | 36 s | 1.4 | ~0 (music) | Humanoid already mid-cartwheel |
| Why do computers suck at math? | Fireship | **6.4×** | 54 s | 2.2 | ~210 | Black screen, "0.1" appears: a puzzle; "I always thought computers were better at math… until I tried to add 0.1 to 0.2" |
| OpenAI's Sora | bycloud | **4.9×** | 60 s | 6.2 | ~190 | Striking image plus on-screen claim "ALL THESE VIDEOS ARE AI GENERATED"; "If you look very closely… something is wrong" |
| Realistic honey simulation? | Two Minute Papers | **1.4×** | 54 s | 0 | ~143 | The object (honey dipper) plus a question on screen; "it looks great, however, this still doesn't seem right" |
| Full free OpenCV course (link below) | Kevin Wood | 0.14× | 59 s | 0 | ~160 | A course title slide; "Welcome to my OpenCV course" |
| Figure F.02 retired | PRO ROBOTS | 0.21× | 30 s | 9.0 | ~10 | Abstract sparks and fire, no subject |
| Quick overview of CSS | Fireship | 0.29× | 33 s | 1.5 | ~195 | A busy webpage screenshot |
| AI news you don't want to miss | AI Search | 0.33× | 60 s | 2.0 | ~165 | Talking head plus "HUGE"; "so many cool updates happened this past week, including…" (a list) |
| AI podcast: study hack or slop? | bycloud | 0.43× | 58 s | 4.1 | ~209 | A meme reaction clip, small, on black |
| Meet the AI co-scientist | Two Minute Papers | 0.48× | 26 s | 1.9 | ~140 | Talking head with "WATCH THE FULL TALK ↓" from the first frame; ~3.9 s before the narration starts |

## Patterns (to test, not proven)

1. **The winners open on the payoff.** The first frame is the thing itself, in motion or in question: the robot
   doing it, the puzzle, the image with its claim. None opens on a title, a logo or a setup.
2. **The losers open on the ask or on something generic.** A course slide and "link below", "WATCH THE FULL TALK"
   from frame one, a webpage, a meme, abstract sparks, "HUGE". Asking before giving is the clearest loser pattern;
   it matches Hormozi's "give value first" (see `research-shorts-offers.md`).
3. **One thing per Short beats a roundup.** The weekly-roundup Short ("so many updates this week, including…") is
   among the worst; every winner is about one thing.
4. **Hooks that work are a puzzle the viewer can see**: "look closely, something is wrong", "it looks great, however,
   it doesn't seem right", "until I tried to add 0.1 to 0.2". The question is answered at the end, and the explainer
   winners end on the answer (Fireship's last line restates the title: "that's why your computer sucks at math").
5. **Pace is not the lever.** Winners range from one unbroken 14-second shot to 6 cuts per 10 s; speech from none to
   ~210 words/min. What they share is a concrete subject from the first frame.
6. **Real footage of a robot doing something wins in our topic** (the three biggest multipliers). Our copyright rule
   (`video-requirements.md` 7.2) doesn't allow other parties' clips, so our Shorts need another way to show motion:
   official stills with movement (Ken Burns, sequences), our own animated explainers, and the source's own words on
   screen. **Resolved (user, 2026-11-16):** official footage the company or lab itself published may be used in short,
   credited, muted clips (`video-requirements.md` 7.2); other parties' clips stay out.

## What we take into the Shorts skill

- First frame = the most striking concrete thing in the story, with a one-line on-screen claim or question.
- Hook = a visible puzzle or a surprising result, answered at the end; the end line loops back to the opening.
- One story per Short.
- The ask ("full episode") comes after the payoff, never in the first seconds.
