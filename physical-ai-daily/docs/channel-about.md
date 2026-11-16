# Channel About text (final for launch, 2026-11-16)

YouTube shows the first ~45 characters of the About text in search and on the channel page, so the first line says
what the channel is (docs/research-channel-intro.md). Wording follows the episode intro line the user approved
("This is Pixels to Actions. Every week, we go through what's new in robotics and physical AI.") and the user's
"inform, don't sell" rule.

## About

```
What's new in robotics and physical AI, explained so anyone can follow.

Every week we go through the main robot and physical-AI stories: new research, products and deals. For each one we
say what happened, how it works in plain words, why it matters, and what isn't known yet. Every source is linked
in the video description.

New episode every week, plus short explainers through the week.

🌐 https://pixelstoactions.com
📩 Business inquiries: <email>
```

First line: 71 characters. Fill in the business email (or delete that line); add the posting day once fixed
("New episode every Sunday").

## How competitors write theirs (16 channels, fetched 2026-11-16 with yt-dlp)

| Pattern | Who does it | Example (≤15 words) |
|---|---|---|
| One-line promise first, often ≤8 words | bycloud, AI Search, Matt Wolfe, Kevin Wood | "Frontier AI Research Explained" (bycloud) |
| Says who it's for / that it's easy to follow | AI Search, Matthew Berman, Wes Roth, AI Revolution | "Making AI easy to understand for everyone" (AI Search) |
| Upload schedule | Matt Wolfe, Fireship | "New videos every Friday" (Matt Wolfe) |
| Links block (site, newsletter, socials) | Matt Wolfe, Fireship, Yannic Kilcher, ColdFusion | |
| Business-contact email | 8 of 16 | |
| Very short or a catchphrase only | Two Minute Papers, Welch Labs | "What a time to be alive!" (Two Minute Papers) |
| Hype or sales language | PRO Robots, TheAIGrid | (avoided: the user's "inform, don't sell" rule) |

## Other channel fields

- **Links:** pixelstoactions.com (set by the user). Optional: the public repo as "How we make these videos".
- **Keywords** (Settings → Channel → Basic info): robotics, physical AI, humanoid robots, robot news, AI news,
  embodied AI, world models, self-driving, robot learning, weekly tech news
- **Watermark:** brand/LogoMark.png (Customization → Branding → Video watermark), shown for the entire video; the
  bottom-right corner is free in our layout.
- **Profile picture:** brand/ChannelIcon.png. **Banner:** brand/ChannelBanner.png (tools/brand_assets.sh).

## Channel trailer (optional, later)

YouTube shows the trailer only to people who aren't subscribed yet. The research suggests it say what the videos
are and how often they come out. The simplest version: the best episode's cold open, then the channel line, then
"New episode every week."
