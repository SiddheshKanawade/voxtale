# Physical AI Daily

Pipeline for a daily YouTube explainer on physical-AI news, X posts and arXiv papers, aimed at a product-manager-level
audience. It runs locally on a Mac.

```
episode.json (script + visuals per beat)
  -> tools/voice.py    narration per beat (edge-tts, optional per-phrase direction) -> audio + word timings -> props.json
  -> video/ (Remotion) each beat's visual lasts as long as its audio; captions and cues follow the word timings
  -> tools/render.sh   QA stills + contact sheet, render, loudness-normalize to -14 LUFS
```

| Folder | What |
|---|---|
| `tools/` | `capture.mjs` (page, tweet, element and video-frame screenshots with highlights), `crop.py`, `discover.py` (arXiv + X sources), `voice.py`, `prosody.py` (pitch-variety and pause metrics), `render.sh` |
| `video/` | Remotion project: episode layout, scene catalog, animated diagrams (`npm install` first) |
| `episodes/` | Episode scripts (`episode.json`). Captured assets, audio and renders are not committed |
| `voicelab/` | Voice comparison experiments (Edge, Kokoro, Chatterbox) and the objective screen + blind-test generator |
| `docs/voice-research.md` | Evidence and measurements behind the narration-voice choice |
| `../.claude/skills/physical-ai-daily/` | The Claude Code skill that drives episode production (symlinked into `~/.claude/skills`) |

Render an episode with `tools/render.sh episodes/<dir>`. Add `--skip-voice` to reuse existing audio.
