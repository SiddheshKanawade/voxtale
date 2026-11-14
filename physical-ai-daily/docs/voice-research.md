# Voice research: making the narration engaging

*Started 2026-11-13. The video format (v2) is frozen, and this doc covers only the voice.*
*Problem: after watching sample v2, the user said the voice "is very monotonous".*

## 1. What the evidence says makes a narration voice engaging

| Finding | Strength of evidence | Source |
|---|---|---|
| Charismatic speakers show a **larger F0 (pitch) range, more pitch and loudness variability, a faster rate, and well-placed pauses**. Steve Jobs differed from Mark Zuckerberg on almost all of these parameters. | Peer-reviewed acoustic studies (correlational) | Niebuhr et al. 2016, *Speech Prosody* ([FAU record](https://cris.fau.de/converis/publicweb/Publication/1062694)); Niebuhr 2020 ([Cadernos de Linguística](https://cadernos.abralin.org/index.php/cadernos/article/view/272)); summary in [Frontiers in Psychology 2022](https://www.frontiersin.org/articles/10.3389/fpsyg.2022.857647/pdf) |
| The same prosodic settings that make politicians and preachers sound charismatic also work for business speakers. | Peer-reviewed, about 1,350 phrases | Niebuhr 2020 (above) |
| A **"smiling" voice quality** was rated significantly more positively on all personality traits (medium effect sizes). | Peer-reviewed experiment | Pearsell & Pape 2023, [*Frontiers in Communication*](https://www.frontiersin.org/articles/10.3389/fcomm.2022.909427) |
| Professional voice-over technique: **emphasize key words** (pitch, volume, tempo) and **vary tempo and pitch** to avoid sounding monotone. | Expert practice, from an HCI paper | Berkeley/Adobe *NarrationCoach* ([PDF](https://visualization.berkeley.edu/papers/narrationcoach/narrationcoach.pdf)) |
| Human narrators shift prosody by context (e.g. lower pitch in narrative passages). **Machine TTS does not**, so it sounds samey. | Peer-reviewed, 93 audiobook pairs | [arXiv 2310.06930](https://arxiv.org/pdf/2310.06930v1.pdf) |
| Claims like "AI voices with natural stress are a third more effective" | **Unsourced vendor marketing; ignore** | e.g. [speechactors.com](https://speechactors.com/article/?p=4927) |

**Gap:** no controlled study isolates pitch variability against measured *engagement* such as YouTube retention.
The best evidence is about perceived charisma and appeal, so our own A/B test (listening plus later retention data) has
to settle it.

## 2. What "monotonous" turned out to mean (our measurement)

The tool `tools/prosody.py` measures pitch variability and range (semitones), loudness variability, pauses, and **phrase-level
variety**: how much each phrase's pitch level differs from the others. All samples read the same 63-word passage
(`voicelab/passage.txt`). The human references are 40 s snippets from popular narrators, measured and then deleted. Their
pause and loudness numbers are confounded by background music, and wpm doesn't apply to them.

| Voice | pitch SD (st) | pitch range (st) | phrase pitch variety (st) | wpm |
|---|---|---|---|---|
| **Chatterbox, *directed*** (per-phrase exaggeration/cfg + pauses) | 5.67 | 17.0 | **3.12** | 181 |
| Chatterbox exaggeration 0.5, cfg 0.5 | 5.81 | 17.3 | 1.48 | 199 |
| Chatterbox exaggeration 0.9, cfg 0.3 | 5.65 | 17.4 | 1.55 | 217 |
| Chatterbox exaggeration 0.7, cfg 0.3 | 5.66 | 18.1 | 1.06 | 208 |
| edge Andrew, *directed* (per-phrase rate/pitch + pauses) | 5.03 | 16.6 | 2.08 | 148 |
| **edge Andrew (current v2 voice)** | **4.87** | **16.3** | 1.83 | 174 |
| edge Emma | 4.47 | 13.8 | 2.17 | 179 |
| kokoro af_heart | 4.61 | 20.0 | 0.61 | 177 |
| kokoro am_fenrir | 4.08 | 14.2 | 1.45 | 199 |
| human: Two Minute Papers (1.84M subs) | 4.07 | 13.4 | 1.99 | n/a |
| human: Fireship (4.33M subs) | 3.57 | 11.8 | n/a (music bed) | n/a |
| edge Christopher | 3.41 | 11.6 | 1.19 | 159 |
| kokoro am_michael | 3.13 | 10.5 | 0.94 | 162 |
| edge Brian | 2.27 | 7.5 | 0.90 | 174 |

**Key insight:** by the standard monotony metric, our voice (Andrew) is *not* flat. It varies pitch more than both
human narrators. What the user hears as monotonous is likely:
1. **Predictable cadence.** Every sentence gets the same rise-fall melody, at a constant 174 wpm.
2. **No meaning-driven emphasis.** Punchlines ("ninety-six percent") are read like filler.
3. **Neutral energy, with no "smile" or excitement.**

Swapping to another stock voice doesn't fix these (the Kokoro and other Edge voices aren't more varied). They need
**direction** (per-phrase pace, pitch and pauses), **an expressive model** (an emotion or exaggeration control), or both.

**Both together measured best.** Chatterbox on its own has the most pitch movement, but higher exaggeration
speeds it up (200–217 wpm, as its model card warns) and *flattens* phrase variety. Directed per phrase (more energy on reveals,
slower on explanations, real pauses), it reaches **3.12 st phrase variety**. That's 1.7× the current voice and 1.6× the
Two Minute Papers narrator, at a natural 181 wpm.

**Intelligibility check** (Whisper-small transcription vs. the script, word error rate): every voice is at about 5%. That
floor is mostly "96%" vs "ninety-six percent" normalization. Chatterbox at exaggeration 0.7–0.9 is slightly worse (8–9%), so
keep per-phrase exaggeration at 1.0 or below and re-check WER on every episode.

## 3. Options

| Option | Cost | Expressiveness control | Notes |
|---|---|---|---|
| Edge TTS (current) | free | global rate/pitch/volume per call; **no** styles or emphasis tags | Unofficial endpoint; flaky (retries needed) |
| Edge TTS + **direction** | free | per-phrase rate/pitch + inserted pauses | Implemented in `voicelab/gen_directed.py`; modest measured gain |
| **Kokoro 82M** (Apache-2.0, local) | free | speed only | Fast and clean, but measured *less* varied than Andrew |
| **Chatterbox** (MIT, local, Resemble AI) | free | `exaggeration` 0–1+ (flat to dramatic), `cfg_weight` (pace) | Outputs carry an inaudible Perth watermark. The vendor claims 63.8% preference over ElevenLabs in its own blind test (unverified). See [model card](https://huggingface.co/ResembleAI/chatterbox) |
| **ElevenLabs v3** (paid API) | about $0.08–0.17 per 1K chars, so **about $0.30–0.60 per 4-min episode** (about 3.5K chars) | inline audio tags (`[excited]`, `[whispers]`, ...) | Most-cited quality leader for narration ([pricing](https://elevenlabs.io/pricing/api)). Needs the user's account and API key |
| Fish Audio S2 Pro / Step Audio EditX | varies | emotion controls | Top of some 2026 leaderboards. Fish needs a commercial license; Step is Apache-2.0 ([overview](https://www.marktechpost.com/2026/05/30/best-text-to-speech-tts-models-in-2026-a-benchmark-based-comparison/)) |
| Higgsfield `generate_audio` | credits | n/a | Account had 0 credits on 2026-11-13 |

## 4. Script-side fixes (apply whichever voice wins)

The research points to delivery cues that the *script* can force, even with a plain TTS:
- Shorter sentences, plus a **question before each punchline** ("The result?"), then a pause.
- Mark **one emphasis word per beat**, and render it slower or higher (direction), or with an audio tag (ElevenLabs).
- **Vary pace by beat role:** hooks and reveals faster, explanations and caveats slower.
- Speak to the viewer ("you"), and use contrast pairs ("not a robot, a brain").

## 5. How to reproduce

```bash
# samples (same passage)
.venv/bin/python voicelab/gen_directed.py en-US-AndrewNeural          # Edge, directed
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-tts/bin/python voicelab/gen_local.py  # Kokoro + Chatterbox (+ directed)
# metrics + listening page
.venv/bin/python tools/prosody.py voicelab/samples/* --words 63
.venv/bin/python voicelab/make_compare.py && open voicelab/compare.html
```

Setup notes: `.venv-tts` (Python 3.11 via uv, about 1.2 GB; plus about 1.5 GB of model weights in `~/.cache/huggingface`).
Chatterbox needs `setuptools<81`, because its watermarker imports `pkg_resources`. Kokoro needs spaCy's `en_core_web_sm`.
The first Chatterbox weight download stalled once, and re-running resumed it. On an M4 Pro, Chatterbox generates a
21 s passage in well under a minute on MPS.

## 6. How to compare audio samples (method, researched 2026-11-14)

| Finding | Source | What it means for us |
|---|---|---|
| Listening tests are the standard for TTS. MOS (ITU-T P.800, 1–5 scale) gives absolute ratings, CMOS/AB gives pairwise preference, and MUSHRA (ITU-R BS.1534) ranks many samples side by side. A MOS is only meaningful within one test. | [Soniox guide](https://soniox.com/wiki/evaluating-tts); [Rethinking MUSHRA, arXiv 2411.12719](https://arxiv.org/pdf/2411.12719); [Picovoice](https://picovoice.ai/blog/measuring-tts-quality/) | The final decision is a listening test. Numbers only shortlist. |
| Objective metrics are "frequently used, but rarely validated against subjective ones". Only 1 of 16 open metrics (TTSDS) reached ρ > 0.5 against human ratings everywhere. | [TTSDS2, arXiv 2506.19441](https://arxiv.org/pdf/2506.19441) | Treat our prosody scores as diagnosis, not a verdict. |
| Automatic MOS predictors: UTMOS is strong in-domain (SRCC ≈ 0.90 in VoiceMOS 2022), but correlation drops out-of-domain. NISQA was weak or negative on modern TTS. | [UTMOS summary](https://www.emergentmind.com/topics/utmos-score); TTSDS2 (above) | Don't use a "quality score" model to pick a narrator. |
| Expressiveness depends on context. Paragraph-level tests give different results from isolated sentences. | [arXiv 2206.12559](https://arxiv.org/pdf/2206.12559) | Judge a multi-sentence passage now, then confirm in a full episode. |
| Word emphasis (prominence) is carried by **duration + loudness + pitch together**. Some reviews say duration and intensity matter most, with F0 secondary. The Wavelet Prosody Toolkit combines all three. | [Wavelet Prosody Toolkit](https://gitee.com/atomai/wavelet_prosody_toolkit); [Suni et al. 2017](https://www.bohrium.com/paper-details/hierarchical-representation-and-estimation-of-prosody-using-continuous-wavelet-transform/811112782979137536-2629); [Mishra et al., Interspeech 2012](https://www.isca-archive.org/interspeech_2012/mishra12_interspeech.html) | Our first emphasis score used only pitch + loudness. **Add word duration.** |
| Whisper's own word timestamps are "too unreliable" and fold pauses into the next word. WhisperX and forced aligners (e.g. MFA) fix this. | [WhisperX, arXiv 2303.00747](https://www.arxiv.org/pdf/2303.00747); [CrisperWhisper](https://huggingface.co/nyralabs/CrisperWhisper) | Confirmed on our data: the "pause before 96%" came out as **0.0 s for every sample**, which is an artifact. Measure pauses from the loudness envelope instead. |
| "Louder often sounds better". The fix is to level by BS.1770 integrated LUFS, then present the samples blind and shuffled. Reference order also biases listeners. | [Blind listening test tool](https://blindlisteningtest.pages.dev/) (vendor); [AES 2022](https://aes2.org/publications/elibrary-page/?id=21908) (BS.1770 is within about 2 dB of listeners) | Our samples range from **-17.4 to -27.3 LUFS** (Chatterbox loudest, Kokoro quietest). The first compare page was biased. |

**Protocol we'll use:**
1. Level every sample to -14 LUFS, the same as our videos.
2. Objective screen, used to shortlist only:
   - Phrase-level pitch variety.
   - Emphasis on "snaps" and "ninety-six": duration, loudness and pitch, each relative to the speaker's own average.
   - Pause before the punchline, measured from the loudness envelope.
   - Pace close to an explainer's pace.
   - WER at or below 10% as a pass/fail gate.
3. Show the waveform and pitch contour for each shortlisted sample, so the reasons for the ranking are visible.
4. The user runs a blind listening test: shuffled, unnamed, loudness-matched samples A–F. Rank them (MUSHRA-style), then do an AB test between the top two.
5. Confirm the winner in a full re-voiced episode, since context matters.

### Objective screen results (2026-11-14, `voicelab/waveforms.py`, all samples at -14 LUFS)

**Scoring.** Each criterion is ranked across all 19 samples, and the ranks are averaged with equal weights; no study gives better
weights. A WER above 10% disqualifies a sample. Credit for the pause before the punchline is capped at 0.6 s. That cap is **our
assumption**, added after the uncapped version rewarded 1.6 s of dead air. The pace target of 150 wpm is industry convention
(explainers 130–150, audiobooks 150–160), not a research finding.

| Rank | Sample | Score | Variety (st) | Emphasis: snaps / 96% (z) | Pause before 96% (s) | wpm |
|---|---|---|---|---|---|---|
| 1 | Andrew, directed | 0.754 | 2.07 | 0.72 / 0.64 | 1.01 | 148 |
| 2 | Christopher | 0.732 | 1.19 | 0.77 / 0.77 | 0.95 | 159 |
| 3 | Eric | 0.688 | 0.76 | 0.87 / 0.86 | 0.96 | 157 |
| 4 | **Andrew (current)** | 0.678 | 1.81 | 1.21 / 0.63 | 0.48 | 174 |
| 5 | Kokoro am_michael | 0.660 | 0.94 | 0.80 / 0.88 | 0.58 | 162 |
| 6 | Emma | 0.622 | 2.17 | 0.92 / 0.68 | 0.45 | 179 |
| 11 | Chatterbox, directed | 0.472 | **3.08** | 0.43 / 0.28 | 0.79 | 184 |

**What the corrected screen changed:**
- Loudness-matching and the duration-aware emphasis measure knocked **Chatterbox directed** from first to 11th. It has the
  most phrase variety, but the two key words barely stand out, and it is fast.
- My direction pushed "snaps" faster (+12% rate), which *shortened* the word. Duration is a main cue for emphasis, so the
  direction worked against itself. **Lesson: slow down on the emphasis word; don't speed through it.**
- Christopher and Eric score well on pace and pauses but have low phrase variety (1.19 and 0.76). In the plots they look like
  a dense wall of sound, which is the "monotonous" complaint itself.
- Andrew-directed is the only sample that beats the current voice on variety, pace *and* the pause before the punchline. The
  margins are small, though (0.754 vs 0.732), and the screen is weakly validated (TTSDS2).

**Screen recommendation:** Andrew + direction. It's the same voice, free, and needs no new install. The blind test decides.

**Validation of the measure:** the pause measure was checked against pauses we inserted ourselves. The directed samples have
0.45 s of silence inserted after "The result?", and the screen measured 1.01 s (Andrew) and 0.79 s (Chatterbox), both
≥ 0.45 s. ✓

The shortlist (top 5 plus the current voice as an anchor) is on `voicelab/blind.html` as A–F. The answer key is in
`voicelab/analysis/blind_key.json`; don't open it before rating.

## 7. Directed Andrew in the pipeline (2026-11-14)

**How it works.** `tools/voice.py` now supports per-beat `"direct": [[style, text, pause_s], ...]`. Each phrase is voiced
with its style's rate and pitch, trimmed to its words using the WordBoundary times, and joined with explicit pauses. Word
timings are shifted to match, so captions and cues still sync. Styles are neutral, hook, explain, build, lift, punch,
question, caveat and aside. `VOICE_STRENGTH` scales the offsets and `PAUSE_SCALE` scales the pauses.

**Whole-episode measurements** (sample v2 script, 36 beats, Whisper WER per beat):

| Version | Phrase variety (st) | Pace (wpm) | Pauses (%) | WER |
|---|---|---|---|---|
| v2 (undirected Andrew) | 2.62 | 171 | 19.5 | 5.1% |
| directed, strength 1 | 2.61 | 168 | 18.7 | 4.4% |
| directed, strength 2, pauses ×1.5 | 2.65 | 162 | 20.0 | 3.5% |
| **directed, strength 3, pauses ×1.5 (rendered as v3-voice)** | **2.78** | **158** | **20.0** | 4.2% |

**Findings:**
- **Strength 1 changed nothing measurable.** Andrew sits at about 112 Hz, so a +4 Hz offset is about 0.6 st. Trimming
  also removed Edge's built-in silence of about 0.5 s around each clip, which the voicelab sample had kept.
- **Edge applies only about 60% of the requested pitch.** We asked for +13 Hz and +25 Hz and measured +8 Hz and +15 Hz
  (median F0 116 → 124 → 131 Hz).
- **On Edge, direction mostly buys pace and pauses, not melody.** Across a whole episode, natural sentence intonation
  dominates phrase variety, and even strength 3 only moves it 2.62 → 2.78. A clearly more melodic delivery needs an
  expressive model (Chatterbox, ElevenLabs v3).
- Strength 3 pushes rates well beyond what voicelab tested: punch −34%, hook +26%. WER stays fine, but **naturalness was
  not measured**. That's the main thing to listen for.

## 8. Results of the listening test

*(filled in once the user has listened: winner, runner-up, and what they noticed)*
