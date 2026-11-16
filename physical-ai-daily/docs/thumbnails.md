# Thumbnails (2026-11-14)

## Evidence

| Finding | Source | Strength |
|---|---|---|
| 89% of 500 "breakout" thumbnails used a face, high contrast, or both. The most common text patterns were no text (28%) and 1–3 words (24%) | [vidIQ study](https://vidiq.com/research/youtube-thumbnail-study/) | Correlational, coded sample; best available |
| High-contrast thumbnails averaged 6.5% CTR vs 4.7% for low contrast (3,200 thumbnails, WCAG contrast) | [mer.vin](https://mer.vin/2026/09/youtube-thumbnail-ctr-for-song-videos-what-3200-thumbnails-teach-us/) | Weak (one niche, vendor blog) |
| Text effects may depend on traffic source: helped in search, slightly hurt in browse | [tubeanalytics](https://www.tubeanalytics.net/resources/research/thumbnail-ctr-patterns-study-2026) | Weak (method undisclosed) |
| Spec: 1280×720, 16:9, under 2 MB. Check readability at the 168×94 sidebar size | [TechSmith](https://www.techsmith.com/blog/youtube-thumbnail-sizes/), [testing guide](https://yt-thumbnail-tester.pages.dev/best-practices) | Practitioner consensus |
| YouTube Studio's **Test & Compare** A/B-tests up to three thumbnails on watch time | [thumbnailtest.com](https://thumbnailtest.com/guides/youtube-thumbnail-explained/) | Feature description (confirm in YouTube Help) |

No peer-reviewed study was found, and the percentage lifts disagree across sources. So: follow the consistent
direction (one focal subject, high contrast, few words, readable small), and let Test & Compare decide on this
channel's own traffic.

## Rules

- **Make 3 variants per episode** for Test & Compare, each a different idea, not a colour tweak: `face` (a robot's
  face or the main subject plus 2–3 words), `stat` (one huge number plus a short line), `split` (two stories side by
  side).
- **At most about 4 words, 2 lines.** The text auto-fits and never wraps.
- **Never repeat the title word for word**; the thumbnail adds the hook.
- **Same design as the episode** (`blueprint` / `blueprint-light`). The brand mark sits top-left, small, every time.
  The bottom-right stays clear because YouTube draws the duration there.
- **Accurate, no hype:** the same rule as the script. "96% in a real car factory" is fine because the pilot measured
  it; "Robots replace workers" is not.
- **Image rights:** thumbnails here use frames from the source's own video or figures. A thumbnail is promotional, so
  its fair-use footing is weaker than inside the commentary itself. Prefer CC-licensed figures (arXiv CC BY) or our own
  diagrams when the story allows; the user decides on press frames.

## Tooling

`episode.json` → `"thumbnails": [{layout, lines, image, image2?, focus?, stat?, label?}]`. Wrap a word in `*asterisks*`
to colour it. Then:

```
python3 tools/thumbnails.py episodes/<dir> [design ...]   # -> out/thumb_<n>_<design>.png + out/thumbs_preview.png
```

The preview shows each thumbnail beside its 168×94 version.

## Round 2: evidence-led template (2026-11-14)

The user didn't like round 1 (face / stat / split) and asked for a thumbnail that invites viewing and shows evidence.
More research:

| Finding | Source | Strength |
|---|---|---|
| Curiosity-gap titles reveal enough to raise a question but not answer it; misleading packaging raises clicks but repeated exposure breeds scepticism | [Clickbait linguistics (ResearchGate)](https://www.researchgate.net/publication/367299547_The_Linguistic_and_Typological_Features_of_Clickbait_in_Youtube_Video_Titles); [misleading news thumbnails survey (PDF)](https://pdfs.semanticscholar.org/c6ef/6cd28a85f864533800b6927aa082caa68dc6.pdf); [Wikipedia: Clickbait](https://en.wikipedia.org/wiki/Clickbait) | Moderate (small surveys, consistent with theory) |
| Viewer satisfaction = expectation (set by title and thumbnail) vs delivery | [Viewer responses to clickbait on YouTube](https://www.researchgate.net/publication/395261047_Viewer_Responses_to_Clickbait_on_YouTube_A_Marketing_Perspective) | Moderate |
| High-reputation sources get clicked more regardless of content; expert quotes raise trust; credibility *labels* alone don't change behaviour | [Winter & Krämer, source cues](https://www.researchgate.net/publication/268517126_A_question_of_credibility_-_Effects_of_source_cues_and_recommendations_on_information_selection_on_news_sites_and_blogs); [Science Advances, labels null](https://www.science.org/doi/10.1126/sciadv.abl3844) | Strong (experiments, incl. a field RCT) |
| Faces and overlay text rise with breakout size; contrast does not | [vidIQ July 2026](https://vidiq.com/vi/research/youtube-thumbnail-study/) | Correlational, coded sample |
| Title and thumbnail should split one promise, not repeat each other | [vidIQ titles](https://vidiq.com/blog/post/youtube-video-title/), [Hooklab](https://hooklab.co.uk/blog/should-your-youtube-title-and-thumbnail-say-the-same-thing), [Overseeros](https://www.overseeros.com/blog/youtube-click-promise) | Practitioner consensus |
| Arrows help only to point at small or unfamiliar things; otherwise a cliché | [Linus Tech Tips](https://rosetta.to/u/linustechtips/why-our-thumbnails-look-like-this), [thumbnailtest](https://thumbnailtest.com/?p=926) | Anecdotal |

**Design that follows:** the `evidence` layout. A big subject "face", a 2-line hook whose coloured line is an honest
curiosity gap the video closes (the "call to watch"), and a tilted **receipt**: a crop of the source stating the claim,
with the number highlighted and a "SOURCE: ..." tag. Round-1 layouts stay in the code but are no longer the default.
The working rules live in the `youtube-thumbnails` skill (`.claude/skills/youtube-thumbnails/SKILL.md`).

## Round 3: weekly episodes 2026-11-01, -08, -15 (2026-11-16)

Research re-run before designing. New or sharper findings:

| Finding | Source | Strength | Effect on this run |
|---|---|---|---|
| YouTube Help: the Studio feature is now "A/B test titles & thumbnails": up to 3 options, thumbnail-only, title-only or both; the winner is picked on **watch time**, and with no clear winner the **first** uploaded option stays. Desktop only; if any thumbnail is under 1280×720 all are downscaled to 480p; changing the title mid-test stops it | [YouTube Help](https://support.google.com/youtube/answer/13861714) | Official docs | Upload the pick first; all variants rendered at 1280×720 |
| Misleading thumbnails drew ~14% more clicks than accurate ones (2,843 videos, 8 countries), framed by the authors as eroding trust | [ThumbnailTruth, arXiv:2509.04714](https://arxiv.org/abs/2509.04714) | Detection paper, not a trust experiment | Confirms the honesty rule: a click lift exists, so the rule has to be enforced, not assumed |
| No rigorous evidence that short text hurts CTR; vendor claims go both ways, and "mystery" videos may do better with less text | [thumbnailtest](https://thumbnailtest.com/guides/text-on-youtube-thumbnail/), vendor blogs | Weak | Keep 2 short lines; no change |
| Faces stay the most consistent claim across 2026 vendor data; percentage lifts disagree and are mostly unverifiable | [Mark Studios review](https://www.markstudios.com/research/video-thumbnail-and-ctr-statistics) | Weak / correlational | No change |

**Audience rule (user, 2026-11-16):** viewers have little or no physical-AI background. Hooks use plain words
(no model names, "VLA", "GPU", "zero-shot"), inform rather than sell (the round-2 "96%... BUT?" teaser was rephrased to
"Car factory test / 96% success"), and the receipt shows the source sentence, including the caveat when it is the
highlighted part (e.g. "proof-of-concept").

**Tooling changes:** `Thumbnail.tsx` gained `stamp` (a label on the subject image; required for generated footage such as
Genie 4, e.g. "AI-GENERATED (GENIE 4)") and `receipt.width` (wider card for tight crops). Receipt crops were computed
from macOS Vision OCR of each capture: a window of ~0.25-0.4 page widths centred on the key phrase, with a `mark` box
around the number, so the words read at full size (the full-width crops of round 2 came out at ~14 px text).
