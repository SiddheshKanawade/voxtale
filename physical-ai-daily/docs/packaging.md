# Packaging content: what makes a title and thumbnail get clicked (2026-11-22)

The user, 2026-11-22: "I am not liking the titles and thumbnails that you are generating. It's not a design problem,
rather a content." Rounds 1–3 in `docs/thumbnails.md` researched design (faces, contrast, word count, the receipt
card). This round researched content: what the words and the picture should say.

Two inputs:
1. A literature review of YouTube's own guidance, large headline A/B datasets, and creator data.
2. A pull of 77 recent robotics and AI videos with reliable channel sizes (yt-dlp metadata, 2025-11-20 to 2026-11-20),
   ranked by views ÷ subscribers. That is a proxy for packaging pull, not CTR.

## Findings

| # | Finding | Evidence | Strength |
|---|---|---|---|
| 1 | **The idea matters more than the wording.** Pick the single most surprising true result before writing any copy | Wording features picked the winner in only 54.4% of 24,333 Upworthy headline pairs ([Gligorić et al. 2023](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0281682)). Ritchie: "You need a compelling idea, or everything else really won't matter" ([YouTube Blog, Jan 2024](https://blog.youtube/creator-and-artist-stories/renes-top-five-january-18-2024/)) | Experimental + official |
| 2 | **Lead with one story; roundups as noun lists underperform** | None of the 14 niche videos that beat 1× their channel's subscribers is a roundup. Bottom: "The Latest Humanoid Robotics Breakthroughs You Need to See" (0.07×), "NVIDIA GTC 2026 Reveals 5…" (0.05×). AI Explained "lead story (plus X)": 154–247K views vs. lists 58–72K | Descriptive, small n |
| 3 | **The promise is paid off early.** The title's lead story is the video's first story | Test & Compare picks on watch time ([YouTube Help](https://support.google.com/youtube/answer/13861714)). YouTube measures survey-rated satisfaction ([Goodrow 2021](https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/)) | Official |
| 4 | **Give the setup and withhold the outcome, the "how" or the catch.** A complete factual sentence leaves no gap | Curiosity peaks at intermediate knowledge ([Kang et al. 2009](https://www.cmu.edu/dietrich/sds/docs/loewenstein/WickCandleLearning.pdf)) | Experimental (lab) |
| 5 | **Use plain words, with names last** | Simpler headlines won across ~31,000 tests ([Shulman et al. 2024](https://doi.org/10.1126/sciadv.adn2555)). On Argon itself, "Google Just Caught Up (Gemini-4 Argon)" (Matt Wolfe) got 279K vs. a 105K median | Experimental + descriptive |
| 6 | **A protagonist doing something concrete** | Single-individual headlines did better ([Banerjee & Urminsky, preliminary](https://home.uchicago.edu/ourminsky/Banerjee_Urminsky_Headlines.pdf)). Niche outperformers: "Robots Just Had Their GPT-3 Moment" (3.1×), "How I made this Voice-Controlled Robot Dog from Scratch" (10.7×) | Experimental (small) + descriptive |
| 7 | **Honest negativity beats hype** | Each negative word +2.3% CTR, each positive word −1.0% ([Robertson et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10202797/)) | Experimental |
| 8 | **Statements over questions** | Questions underperformed in several datasets (Kuiken et al. 2017; [vidIQ titles](https://vidiq.com/research/youtube-title-formulas-study/) 0.82×) | Mixed, consistent direction |
| 9 | **The thumbnail shows the moment: action, outcome or tension.** Use 0–3 words that add to the title | YouTube: titles and thumbnails "work together to create interest and tell a story" ([Help](https://support.google.com/youtube/answer/12340300)). Every niche outperformer's thumbnail showed one subject in action with 0–3 words, and none had a screenshot. "No text" was the most common pattern among breakouts ([vidIQ](https://vidiq.com/research/youtube-thumbnail-study/)) | Official + correlational |
| 10 | **No evidence that a source screenshot ("receipt") on a thumbnail helps.** It's a second idea and unreadable at feed size. Keep proof in the video's first minute and the description | Not found in any study. More thumbnail text went with fewer views ([1of10](https://1of10.com/blog/what-actually-makes-a-youtube-video-go-viral-in-2025), correlational) | Absence of evidence |
| 11 | **Honest curiosity, not deception.** "Type 1" packaging conveys the content accurately and briefly; deceptive packaging gets backlash | [Veritasium, "Clickbait is Unreasonably Effective"](https://www.youtube.com/watch?v=S2xHZPH5Sng). Clickbait lowers shares and trust ([Mukherjee et al. 2022](https://ideas.repec.org/a/spr/joamsc/v50y2022i3d10.1007_s11747-021-00830-x.html)) | Practitioner + observational |

Caveats:
- The headline A/B tests are web headlines, not YouTube.
- The niche pull is 77 videos, uses views ÷ subscribers rather than CTR, and comes from search ranking.
- Several niche outperformers use AI-generated fake images (e.g. a CEO "in a kitchen" with robots). We never do that.

## What we were doing wrong

- **Titles:** noun-list roundups ("Gemini 4 Argon, ETH Zurich's Rubik's Cube solved by foot, and more"). They make no
  promise and use names a newcomer doesn't know.
- **Thumbnails:**
  - Full factual sentences ("Apollo humanoid learned cornhole from a video") answer the question before the click.
  - A paper screenshot adds a second idea.
  - Short covers used two- or three-line sentence hooks.

## Rules (Shorts: in `promo-shorts`, from 2026-11-22)

- **Title:** a robot doing something concrete, in plain words, name last; a statement; one idea; no hype words. Draw
  an honest catch from the source.
- **Cover:** one real frame of the decisive instant, 1–3 words that add to the title, and a label naming the robot and
  its type. No sentences, no screenshots.
- First applied to Oct 11–18 (cover E of each Short). Examples: "A Robot Dog Solved a Giant Rubik's Cube With Its
  Feet" with "No hands"; "This Robot Learned Cornhole From a Video of a Person" with "7 in a row".

## Rules (episodes: in `youtube-thumbnails`, `weekly-package`, `physical-ai-daily`, from 2026-11-22)

- **Title:** one lead story, a robot doing something concrete, plain words first and the name last, then "(plus N
  more)" if anything. The lead story plays first in the episode.
- **Thumbnail:** the `moment` layout in `Thumbnail.tsx`: one real frame of the decisive instant, full-bleed, with 0–3
  words and a chip naming the robot and its type. The `evidence` (receipt) layout is retired as a default.
