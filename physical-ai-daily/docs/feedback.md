# Feedback log (user's review notes, to act on in the next videos)

Newest first. Each item: what the user said, where it applies in the code, and its status. Read this before making
or re-rendering a video, and move an item to "Done" with the commit once it's fixed.

## Open

### 2026-11-16: the zoom on text is uncomfortable to watch
- **User:** "The zoom effect we have on text (papers, images) is paining in eye." Fix it in the next video; the current
  renders stay as they are.
- **Where it comes from:**
  - Episodes, `video/src/scenes.tsx`:
    - `Shot`: zooms in on the `--find` highlight (up to 2.2×), with a slow 1.03–1.08× push-in.
    - The "punch-in" cut (×1.3) on static beats longer than `HOLD_CAP_S` = 7 s, in both `Shot` and `Photo`.
    - `Photo`: a Ken Burns pan and zoom.
  - Shorts, `video/src/shorts/Short.tsx`: `Picture` adds a 1.0–1.08× push-in, and frames "shot" beats tightly on
    the highlight.
- **To decide when fixing:**
  - Hold text still. Options are no push-in on screenshots and papers, or one gentle settle at the start and then
    static.
  - Show the highlighted sentence at a readable size from the first frame instead of zooming towards it.
  - Replace the punch-in on text beats with a cut to a different visual.
  - Photos of real things may keep a very slow drift. Check that with the user.
- **Status:** open. Next video.

## Done

(none yet)
