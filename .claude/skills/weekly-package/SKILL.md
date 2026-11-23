---
name: weekly-package
description: One command from a week to a complete Pixels2Actions posting package (project physical-ai-daily) — research the week, make the long-form episode (4K, light and dark), thumbnails, 3–4 promo Shorts with covers, and publish.md with titles, descriptions, chapters, sources and tags. Takes the week (its end date, e.g. "2026-10-18" or "Oct 11–18"). Use when the user says "/weekly-package <week>", "make the package for <week>", or asks for the complete posting package of a week.
---

# Weekly posting package (kickoff)

`/weekly-package <week>`. One argument: the week. Accept `2026-10-18` (the week's **end** date, the folder
convention), `Oct 11–18` or `week of Oct 11`. The window is 8 days inclusive, end minus 7 through end, and the
folder is `physical-ai-daily/episodes/<end>-weekly/`. Example: `episodes/2026-10-25-weekly` covers Oct 18–25. If the
week ended more than ~10 days ago it is a **backlog week**, so use dated framing throughout.

Typing the command means **run unattended**: don't stop for the pick review. Print the picks when they're chosen and
go on. Stop and ask only for a HARD item that can't be met (checklist `docs/video-requirements.md`), or for a
decision that is genuinely the user's.

This skill only orders the work. The rules live in the phase skills; read each one when its phase starts, and
`physical-ai-daily/docs/feedback.md` before scripting:

| Phase | Skill | Output |
|---|---|---|
| 1 Research | `physical-ai-research` | `<ep>/research.md` (5 picks, ≥3 learning on backlog weeks) |
| 2 Episode | `physical-ai-daily` | `<ep>/episode.json`, `<ep>/out/*.mp4` (4K light + dark) |
| 3 Thumbnails | `youtube-thumbnails` | 8+ titles and 8–10 "moment" thumbnails on a lettered contact sheet, one per story, and the **A/B test set** (3 title + thumbnail pairs) unless the user says no test this week |
| 4 Shorts | `promo-shorts` | 3–4 Shorts in `<ep>/shorts/<slug>/out/` (dark + light), each with a cover |
| 5 Package | (this skill) | `<ep>/publish.md` complete, the hand-off message |

## 0. Preflight (2 minutes, before anything else)

- If `research.md` already exists for the week, skip phase 1 and show the picks. If episode outputs exist, say what's
  there and build only what's missing.
- `python3 tools/credits.py`: twitterapi.io, NewsData, ElevenLabs.
  - Voice: ElevenLabs Eric (`cjVigY5qzO86Huf0OWal`, `eleven_v3`) when there is enough credit. An episode is about
    3,600 characters and a Short about 600. Otherwise use edge-tts `en-US-AndrewNeural` for all of it.
  - Say which voice and why in the hand-off. Never print the key.
- Check every judge model answers (Fable quota runs out; fall back to Haiku, see the research skill).
- `df -h /System/Volumes/Data`: keep about 10 GB free for a 4K episode, and check `.render.lock` isn't held by a dead
  process.
  - The disk is shared with other projects. On 2026-11-20, MyoChallenge runs and Brave temp files took it from 8.6 GB
    to 1.6 GB mid-run.
  - Check again before each 4K render. If space is short, ask the user what to free, and offer this project's own
    leftovers first: old samples' ignored `out/`, `audio/`, `qa/` and `assets/`. Keep the reference episode named in the
    physical-ai-daily skill and every git-tracked file.
  - Never delete outside this project.
  - Free RAM matters too: under ~4.2 GB free at render start, Remotion caches every frame on disk (2026-11-23, see
    the physical-ai-daily skill). Disk also drops while phases 3–4 run beside the 4K render: hold their large writes
    until it finishes.

## 1–4. Run the phases in order, each in its own subagent

The phases are long, and their tool output would flood one context. Run each as a subagent:
- Brief it with the week, the folder, this skill's rules section and the phase skill to load.
- Research spawns its own subagents (triage, judges, the pairwise judge), so start it as a `general-purpose` agent,
  not a fork: forks can't spawn.
- Wait for it to finish, check its output files yourself, then start the next phase.
- **Between research and the episode, pick the lead story** (`youtube-thumbnails`, "Pick the lead story"). It is the
  most surprising true result a newcomer can picture, with a showable moment. Tell the episode phase to open with it:
  the title sells it, so it must play first (2026-11-22, `docs/packaging.md`).
- After research, read `research.md`'s run notes. If the pairwise cut line wasn't answered by a fresh judge
  (2026-11-20: a resumed run answered it itself after seeing the scores), re-ask it from this session. Move
  `judge/pairwise.json` aside, then give a fresh `opus` agent only `judge/pairwise.md`, with no other file and no web.
  Then run `judge_panel.py settle`, then `research_md.py <ep> --window FROM:TO --min-learning 3` (backlog weeks). Add
  `--models "Opus, Sonnet and Haiku"` when a fallback model judged. The generator overwrites `research.md`, so
  re-append the hand-written run notes afterwards.
  - Check a "doesn't match" report against `pairwise_index.json` before changing anything: "second" in a reversed
    pair names the same story.

Phases 3 and 4 may overlap phase 2's final 4K render. They need `episode.json`, the assets and the per-story
thumbnails, not the final MP4s. `tools/render_both.sh` takes the shared lock, so renders queue.

## Rules that cut across phases (user decisions; keep them)

- **Sourcing:** never generate facts or opinions. Every number in narration, covers and descriptions is on a
  captured page.
- **Backlog framing:**
  - Intro "In the week of October 11th…". On-screen date "Oct 11–18, 2026".
  - Never "this week", "today" or "just announced".
  - Say what changed since, when a source says so.
- **Intro** ends, word for word: "This is Pixels to Actions. Every week, we go through what's new in robotics and
  physical AI." Inform, don't sell.
- **Episode:**
  - 4K (render.sh defaults).
  - Narration at 1.15×: `"speed": 1.15` in episode.json (user, 2026-11-21). Chapter times in `publish.md` come from
    the sped-up `props.json`.
  - No per-story time target; hard cap 20 min.
  - Screenshots hold still.
  - Tags small and subtle.
  - "AI-generated" only on media we generate.
- **Audio:** one continuous take per segment, cut at natural pauses (`voice.py`), never per-beat pieces.
- **Clips:** official footage first, a few seconds each, muted, credited, URL in the description.
  - Raw downloads stay git-ignored and are deleted after cutting.
  - The user OK'd official clips for episodes; ask before any non-official footage.
- **Titles and thumbnails: content first** (user, 2026-11-22: "not liking the titles and thumbnails… it's content,
  not design"). Research and sources are in `docs/packaging.md`; the working rules are in `youtube-thumbnails` and
  `promo-shorts`.
  - **Episode title:** one lead story, a robot doing something concrete, plain words first and the name last, and the
    rest as "(plus 4 more)" if anything. Never a noun list: rejected "Gemini 4 Argon, ETH Zurich's Rubik's Cube solved
    by foot, and more". Use statements, not questions, and no hype words. An honest catch from the source is a good
    angle.
  - **Thumbnail / cover:** one real frame of the decisive instant, 0–3 words that add to the title, and a label naming
    the robot and its type. No sentences, no source screenshots ("receipts").
- **Wide options (the user iterates a lot here):**
  - At least 8 titles per video, each tagged by angle, always including *current* (names companies or models) and
    *plainer* (a newcomer understands); one marked ★; each names the thumbnail/cover it pairs with.
  - Episode: 8–10 thumbnails; each Short: 3–4 covers. Vary the image and the words. One lettered contact sheet per
    video, in the hand-off.
  - No "Robotics this week:" prefix.
  - Title and cover/thumbnail text complement each other and never repeat.
- **A/B test set (YouTube Test & Compare, "Title and thumbnail" mode; user, 2026-11-18):**
  - First read `physical-ai-daily/docs/research-ab-testing.md`, which covers how Test & Compare picks a winner, how
    different the variants should be, and small-channel traffic.
    - If it is missing or more than ~60 days old, do a deep web search first (WebSearch `extended`, Firecrawl,
      YouTube Help and Creator Insider, practitioner studies). Then update the doc with dated sources, and follow
      it over the defaults below where they differ.
  - Build **3 pairs**, each a different idea, so the test compares ideas, not wording. Make them about the lead
    story (outcome vs. catch vs. how), or another story only if it plays in the first ~2 minutes: the test picks on
    watch time. ★ goes first, because YouTube keeps the first when there's no clear winner.
  - In each pair the thumbnail words add to the title and never repeat it. Title + label + words together say what
    happened and to what, never a bare number. The label names the robot and its type: "Unitree G1", "Apollo
    humanoid".
  - Avoid vague labels such as "AI-run". Rejected examples: "Learned from 13.2 minutes", "Doubling every 54 days?".
  - Titles stand alone and fit in about 70 characters. The user's A/B titles had no date suffix; the week stays in
    the description.
  - Two strong pairs beat three with a weak one. Test whole ideas (a different lead story, or another framing of
    it), never colour or word-order tweaks: YouTube says similar options rarely produce a winner.
  - In the hand-off, tell the user to:
    - start the test at publish and let it run to its end (YouTube says within two weeks);
    - never edit the title or thumbnail mid-test, because that stops it;
    - act only on a "Winner" label. Inconclusive is likely at our traffic, and YouTube then keeps the first option.
  - Log each test (mode, the three angles, the label, the shares) in `docs/youtube-feedback.md`, so patterns show up
    across episodes.
  - Render each pair's thumbnail in light and dark to `out/ab-test/<n>-<slug>-{light,dark}.png`. Look at each at
    sidebar size (168×94).
- **Shorts:**
  - Newcomer format: two-sentence plain opening (background, then question), what they did, result (≤1 number),
    why it matters as the source says it, one caveat, and a CTA.
  - Speed 1.175×.
  - Moment cover (ShortCover): one real frame of the decisive instant, 1–3 words and a label naming the robot. Not
    frame 0, and no sentences.
  - The end card shows **that Short's own story** thumbnail with "FULL EPISODE ↓".
  - The user uploads the **dark** versions.
- **Never upload, never post.** Commit only code, skill and doc changes. Never commit `.env`, `p2a_topics.json`,
  media, or episode working files. The repo is public.

## 5. Package and hand off

Finish `<ep>/publish.md` so the user can post from it alone:
- **Episode:**
  - The "Options" section: all titles with angle and paired thumbnail, the contact sheet, rubric scores.
  - The description: names the week, then chapters with timestamps from the rendered `props.json`, every source
    URL, clip credits, and tags.
  - **A/B test** section ready to paste into Test & Compare: the 3 pairs in order, each with title, thumbnail file
    and a one-line "what this variant tests".
  - MP4 paths (light and dark).
- **Each Short:**
  - 8+ title options and 3–4 covers (contact sheet), paired.
  - Each ★ title passes the same "what happened, to what" test as the A/B set.
  - Description starting `Full episode: <EPISODE_URL>` (a placeholder until the user posts and sends the link; then
    fill it in).
  - The Related video setting.
  - Cover path and dark MP4 path.
  - A suggested posting order (strongest hook first).

Before handing off, look at the episode's beat sheet, every thumbnail and every Short cover yourself.

The hand-off message lists:
- Every path the user uploads.
- **The A/B set in full** (3 titles + thumbnail paths), and every title option with its angle, in chat, not
  only in publish.md. The user picks from the chat.
- The contact sheets.
- The voice used.
- What was dropped or unverified, and why.
- Any credits spent (Higgsfield, ElevenLabs).

Then append to `docs/feedback.md` or the right skill anything this run taught (tool failures, workarounds),
so the next clean session has it. Commit.
