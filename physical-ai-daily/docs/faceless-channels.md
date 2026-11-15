# Faceless channels

Pixels2Actions is faceless: a narrator's voice over source visuals, explainers and captions. These are the channels
in our space that work the same way, and what we can learn from each. Numbers from the 2026-11-15 scan
(`channels.md`).

**How "faceless" was checked:** the latest 4 thumbnails of each channel, by eye, for the creator's own face. Faces
of public figures (CEOs, researchers) don't count. Limits: a creator can appear inside a video without being on the
thumbnail, and a few thumbnails is a small sample. The "what it does" notes come from titles, thumbnails and length,
not from watching; that's the next step below.

## In our space

| Channel | Subs | Median views | Length | Uploads | What its packaging shows | What to take from it |
|---|---|---|---|---|---|---|
| [Two Minute Papers](https://www.youtube.com/@TwoMinutePapers) | 1.84M | 182,500 | 5.4 min | ~2/wk | One result per video; illustrated thumbnails (an AI logo smashing a wall), a recurring logo badge | The proof that one research result, explained short, can carry a channel. Closest to our "learning" stories and our length |
| [Welch Labs](https://www.youtube.com/@WelchLabs) | 914K | 443,500 | 24 min | rare | Clean animated figures ("FAILS → LEARNS"), one researcher's photo | Animation-led explanation earns the highest median views here, at a slow cadence. A model for our `diagrams.tsx` explainers |
| [AI Explained](https://www.youtube.com/@aiexplained-official) | 459K | 109,500 | 22 min | ~0.6/wk | Public figures plus a short quoted claim in quote marks ("'Behavior we cannot verify'") | Quote-led packaging, close to our "receipt" thumbnail idea: the claim, sourced, carries the click |
| [bycloud](https://www.youtube.com/@bycloudAI) | 239K | 70,000 | 18 min | ~1/wk | Memes and a cat, a striking number ("-50% training compute") | Humour plus one hard number in technical topics |
| [AI Search](https://www.youtube.com/@theAIsearch) | 774K | 370,000 | 33 min | ~2/wk | A recurring frame ("huge AI news", "this is wild") over two images | A fixed weekly-roundup frame that viewers recognise; very long videos |
| [TheAIGRID](https://www.youtube.com/@TheAiGrid) | 407K | 63,500 | 26 min | ~5/wk | CEOs' faces plus shouty quotes ("We Cant Release It!", "AI Now BANNED!") | What we deliberately don't do: hype claims in packaging |
| [AI Revolution](https://www.youtube.com/@airevolutionx) | 578K | 69,000 | 14 min | ~6/wk | AI-generated robot and humanoid renders, launch-stage photos | The biggest faceless channel covering robots, built on volume and hype. Our contrast: sourced, fewer, explained |
| [PRO ROBOTS](https://www.youtube.com/@PROROBOTS) | 373K | 27,500 | 21 min | ~1/wk | Robots only, with arrows and short text ("THAT'S THE WHOLE ROBOT?!") | Closest topic match; robot-first thumbnails with one pointed detail |
| [AI News](https://www.youtube.com/@AINewsOfficial) | 94K | 3,200 | 8 min | ~3.5/wk | Robot renders plus product logos | Same topic and length as us at small scale: same-week coverage alone doesn't grow a channel |

## Craft references outside our space

| Channel | Subs | Median views | Length | What to take from it |
|---|---|---|---|---|
| [Fireship](https://www.youtube.com/@Fireship) | 4.33M | 1.95M | 6 min | Fast, dense, funny news at our length; a branded series badge ("THE CODE REPORT") on every thumbnail |
| [ColdFusion](https://www.youtube.com/@ColdFusion) | 5.26M | 962,500 | 19 min | Narrated documentary storytelling; brand bar on every thumbnail |
| [3Blue1Brown](https://www.youtube.com/@3blue1brown) | 8.76M | 2.3M | 30 min | Animation does the explaining; one big concept word on the thumbnail ("Entropy") |

## Patterns across them (our reading, to test)

1. **Faceless works at every scale here**: from AI News (94K) to Fireship (4.3M). The big ones win on explanation
   quality or voice, not on having a face.
2. **The teaching channels have the best views per subscriber**: Welch Labs (443K median on 914K subs), Two
   Minute Papers, AI Explained. The hype channels upload far more for fewer views per video.
3. **A recurring brand element on thumbnails** (Fireship's series badge, ColdFusion's bar, Two Minute Papers' logo,
   AI Search's frame) is common among the largest. Worth checking against our `docs/thumbnails.md` rules.
4. **Nobody faceless explains physical AI well yet**: the robot-focused faceless channels are hype-led (AI
   Revolution) or low-reach (PRO ROBOTS, AI News).

## Next step

Read transcripts of 2–3 recent videos from Two Minute Papers, Welch Labs, AI Explained, bycloud and PRO ROBOTS
(`chatter.py --transcripts` or the `youtube-search` skill plus whisper.cpp) to compare hooks, structure, pacing and
how they explain a hard idea, then update `docs/attention.md` with anything that holds up.
