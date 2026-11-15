# Skills index

Every Claude Code skill this project uses lives in the repo under `.claude/skills/`. On a new machine, run
`physical-ai-daily/tools/link_skills.sh`: it fetches the pinned submodules and links each skill into
`~/.claude/skills/`, so the repo is the single source of truth. Before adding a third-party skill: check stars,
license, recent activity and whether it needs keys or outside services; read every file; record it here.

## Ours

| Skill | Does | Notes |
|---|---|---|
| `physical-ai-research` | The week's research: harvest (arXiv, X, news), web discovery, triage, chatter, verification, cards, LLM judge panel, `research.md` | Rules: sourcing (no generated opinions), learning over knowing, YouTube calibration |
| `physical-ai-daily` | Production: script (`episode.json`), capture, voice, render, hand-off | Frozen v1.0 identity (light default); acceptance checklist in `docs/video-requirements.md` |
| `promo-shorts` | 2–3 promotional 9:16 Shorts per episode: hook on frame 0, payoff, one CTA to the episode; light and dark | Evidence: `docs/research-shorts-*.md`. Official press footage allowed (short, credited, muted) |
| `youtube-thumbnails` | Evidence-led thumbnails: research, 5–7 candidates, rubric, pick plus two for Test & Compare | Open: whether news-site screenshots may appear in thumbnails |

## Third-party

| Skill | Source | License | How it's in the repo | Pinned at | Notes |
|---|---|---|---|---|---|
| `youtube-search` | [Mathews-Tom/armory](https://github.com/Mathews-Tom/armory) `skills/youtube-search` | MIT | Copied, with `LICENSE` | `169d5df25490` | yt-dlp locally, no key. Local notes added: pacing (YouTube bot-checks bursts), never use browser cookies, channel-stats command |
| `remotion-best-practices` | [remotion-dev/skills](https://github.com/remotion-dev/skills) | Remotion License (source-available, not open source) | **Git submodule** `vendor/remotion-skills`, symlinked from `.claude/skills/` | `7e541aa4d736` (v4.0.543) | Not copied: no open-source license, and the repo is public. Our `physical-ai-daily` rules win where it differs (render via `render.sh`; don't open Studio first) |
| `youtube-stats` | [digitalsamba/claude-code-video-toolkit](https://github.com/digitalsamba/claude-code-video-toolkit) `.claude/skills/youtube-stats` | MIT | Copied, with `LICENSE` | `01e85f4bbf5b` | **Not runnable yet**: its scripts belong to that toolkit, and the YouTube Analytics setup isn't done. Adopt when the first episode is live (`youtube-feedback.md`) |

## Considered and not adopted

| Skill | Why not |
|---|---|
| [ZeroPointRepo/youtube-skills](https://github.com/ZeroPointRepo/youtube-skills), [sickn33/agentic-awesome-skills](https://github.com/sickn33/agentic-awesome-skills) YouTube skills | Route every query through TranscriptAPI.com, a paid third-party service |
| [gupsammy/Claudest](https://github.com/gupsammy/Claudest) `search-youtube` | No license |
| [affaan-m/everything-claude-code](https://github.com/affaan-m/everything-claude-code) `video-editing` | Built for cutting real camera footage, not screenshot-and-explainer videos |
| OpenMontage `ffmpeg` | Basic ffmpeg commands only |
| digitalsamba `video-design` | Useful rules (one accent colour, five text sizes), but our design is frozen at v1.0; revisit at the next design change |

## Updating a third-party skill

- **Submodule** (Remotion): `git -C vendor/remotion-skills fetch && git -C vendor/remotion-skills checkout <sha>`,
  read the diff, then commit the new pointer and update the table.
- **Copied** (MIT): re-fetch at the new commit, keep the `LICENSE` and our local-notes section, read the diff, update
  the table.
