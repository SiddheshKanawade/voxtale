# When a source blocks us (user rule, 2026-11-15)

Many sites block automated clients for technical reasons (bot protection, CDN rules) while the content itself is public.
We cite every source, so getting to public content another way is fine. Work down this ladder and stop at the first
step that works. Note in the hand-off which step got each source.

1. **Retry slowly** with our honest user agent (`Pixels2Actions-research/1.0`): a few attempts, with backoff.
2. **Find the same thing at an official mirror:** the company's own YouTube or X copy of the video, its press kit or
   newsroom, a direct media URL in the page source (`<video>`, `og:video`, JSON in `<script>`), the PDF version, the
   RSS feed.
3. **Use another route to the same page:** an archive.org snapshot, Firecrawl (the `firecrawl` skill), or the user's own
   Chrome through the Claude-in-Chrome extension to read or screenshot it as a person would.
4. **Use workarounds:** a regular browser user agent with `curl`, `capture.mjs` or `yt-dlp`; cookies for YouTube from a
   cookie file (`yt-dlp --cookies <file>`). Use a **secondary account's** exported cookie file, because yt-dlp warns that
   accounts used this way can be flagged. Read the user's main browser profile (`--cookies-from-browser`) only if the
   user has said so for that run.
5. **Be creative about getting the information.** The same fact is often stated elsewhere:
   - the original press release behind a news story;
   - a syndicated copy of the article;
   - the author's or company's own post;
   - the paper or its supplementary material;
   - a regulator's filing;
   - an interview transcript;
   - a different outlet quoting the same source.

   Write a one-off script if that helps (`docs/tools.md`). The **sourcing rule** still holds: every fact must be
   quoted from a page you actually read, attributed to whoever published it. Being creative means finding another
   real source. It never means filling the gap from memory or inference.
6. **Drop it.** If all of the above fail, drop that fact or clip and say so in the hand-off, with the steps you tried.

**Limits that don't change:**
- **Paywalled or login-only content is not public.** Don't bypass paywalls.
- **Clips still follow `video-requirements.md` HARD 7.2:** a few seconds, muted, credited, official first, wire
  agencies last. Getting the file is not the same as having the right to show all of it.
- **Claude Code's permission system has the final say.** If it refuses a command, that route is closed for this run.
  Don't try to get the same thing through a different command. Report the refusal so the user can allow it in
  `/permissions` if they want to.
