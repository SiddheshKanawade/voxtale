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
