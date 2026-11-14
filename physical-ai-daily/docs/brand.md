# Channel identity (FROZEN 2026-11-14: approved by the user; the skill's Format section is the summary)

**Goal:** anyone who sees a video can tell it's from this channel, without the video advertising Pixels2Actions or the
website. The user likes the name and logo in the top bar and found the old navy background bluish and dated.

## What the research says

| Finding | Source | Strength |
|---|---|---|
| A channel brand is a *system* repeated in every video: palette, fonts, intro, lower thirds, recurring structure | [Pixflow](https://pixflow.net/blog/youtube-channel-branding/), [Mark Studios](https://www.markstudios.com/blog/channel-branding-visual-identity-system-for-creators) | Practitioner guides (vendor blogs) |
| Keep the palette small: one primary, one accent, one neutral | [Pixflow](https://pixflow.net/blog/youtube-channel-branding/) | Practitioner guide |
| Intros should be about 5 s or shorter, or skipped entirely. A recurring phrase or cold-open structure can be the intro | [Faceless channel branding](https://easyviral.ai/blog/build-brand-identity-faceless-youtube-channel-2026), [YouTube branding guide](https://www.youtubetoolshub.com/blog/youtube-channel-branding-guide) | Practitioner guides |
| **Sound cues beat logos and colours for branded attention.** Ipsos found sonic brand cues strongly tied to high-performing ads; logos and colours were used most but were less present in top ads | [Stephen Arnold Music, reporting Ipsos](https://stephenarnoldmusic.com/sonic-branding-ranked-1-for-brand-awareness/), [tfa.studio summary](https://www.tfa.studio/en/sonic-branding-research) | Industry research, reported second-hand |
| Most logos are not truly distinctive; only about 19% reach the top distinctiveness tier | [ANA](https://www.ana.net/miccontent/show/id/aa-2025-02-distinctive-brand-assets) | Industry study |
| The same narrator on every video is a core part of a channel's identity (Kurzgesagt) | [Wikipedia: Kurzgesagt](https://en.wikipedia.org/wiki/Kurzgesagt) | Encyclopedic |
| A fixed spoken opener or format line works as a signature (Two Minute Papers) | [Grokipedia: Two Minute Papers](https://grokipedia.com/page/Two_Minute_Papers) | Wiki mirror, weak |
| A consistent visual house style (Vox) can become recognisable enough that others imitate it | [Medium essay](https://medium.com/@seijiyushin/why-every-documentary-on-youtube-suddenly-looks-like-vox-6220588e60ba) | Opinion |

**Caveats:** there's no YouTube-specific study on intro length or recognition. The sonic-branding figures come from
vendors reporting Ipsos and System1, and some widely repeated numbers ("68% memorability loss" and similar) have no
traceable source, so we ignore them.

## Proposed elements (in `episodes/2026-11-14-sample-v6-identity`)

**Update:** the user picked the **Blueprint** design from five candidates, with **Blueprint Light** as the light
alternative (`episodes/2026-11-14-sample-v7-*`). Attention rules that shaped it are in `docs/attention.md`.

| Element | What it is | Where |
|---|---|---|
| Narrator | ElevenLabs v3 Eric (frozen) | every beat |
| Name in the top bar | Small logo mark + "Pixels2Actions" wordmark; the user likes it | `brand.tsx` Lockup, `Episode.tsx` TopBar |
| Palette | Neutral charcoal `#0f0f11` (replaces navy), brand red `#f43f2e` for labels and the progress bar, brand yellow `#f0c945` for the accent and caption highlight, warm white text | `theme.ts` |
| Backdrop | Charcoal with a faint red glow top-left, a faint yellow glow bottom-right, a vignette, and a drifting grid of square "pixels" | `Episode.tsx` Backdrop |
| Sonic logo | Original 1.6 s cue generated in code: four "pixel" blips rising an A-major arpeggio, a whoosh, then a soft chord. Plays with the sting after the cold open and on a 1.6 s end-card tail, never under the voice | `tools/brand_audio.py`, `Episode.tsx` |
| Sting | 1.6 s after the cold open: pixels pop in, the arrow drives forward, the name wipes in | `brand.tsx` Sting |
| Named sections | Every story uses the same four markers: IN PLAIN ENGLISH, THE EVIDENCE, WHY IT MATTERS, REALITY CHECK. Each gets a chip at top right. (A tick sound was tried and removed: it overlapped speech, see `docs/attention.md`.) Set per beat with `"section": "plain" / "evidence" / "matters" / "reality"` | `Episode.tsx` SectionChip |
| Spoken cues | "Story one / Story two" opens each story, the caveat beat opens with "Reality check.", and the outro opens with "That's Pixels to Actions." | narration rules |
| End card | Lockup, "Sources linked in the description" and Subscribe; no URL, no newsletter | `scenes.tsx` End |

**Not in the video:** the website, its URL, the newsletter, pricing.
