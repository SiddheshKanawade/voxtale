#!/usr/bin/env python3
"""Build the judge panel's story cards from stored sources only, so no card carries the researcher's own wording.

Usage: python3 -I tools/build_cards.py <ep> [--today YYYY-MM-DD]
Reads <ep>/judge/stories.json (one entry per story, below) and writes <ep>/judge/cards.json + cards.md.

A card has four parts, each from a stored source:
  Source facts     verbatim quotes recorded by the verification step (<ep>/verify/*.json), each with its URL
  Not stated       what the source leaves out, recorded by the verification step ("gaps")
  Signals          raw counts: harvest cluster size, X likes and engagement percentile, HN points, YouTube views
                   (only videos uploaded inside the window)
  What people say  verbatim X posts, HN titles and YouTube comments (5+ likes, no links) from <ep>/chatter/*.json,
                   each with handle, followers and likes
No summaries, verdicts, ratings or "consensus" lines: an opinion appears only as someone's own words.

stories.json entry (the only hand-written input; every field is a pointer or a filter, not prose):
  {"id": "s04", "label": "Google DeepMind: Gemini Robotics 2.5", "type": "release", "company": "Google DeepMind",
   "verify": ["Gemini Robotics 2.5"],        # substrings of <ep>/verify/*.json "story" names
   "finds": [],                              # substrings of <ep>/web/finds_*.json "event" names (summaries; marked so)
   "chatter": ["gemini-robotics"],           # <ep>/chatter/<slug>.json
   "keywords": ["gemini robotics 2.5"],      # a post/video/HN title must contain one to count (lowercase)
   "cluster": ["gemini robotics"],           # substrings of triage "event" names, for the harvest count
   "exclude": {"<post url>": "<why it's off-topic>"},   # shown on the card
   "date": "2026-11-10"}                     # the primary source's date
Label it "<who>: <what>" with no adjectives, numbers or hooks: a judge shouldn't be sold by the headline.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import sys
from datetime import date as Date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discover import engagement  # noqa: E402

MIN_COMMENT_LIKES = 5


def load(pattern: str) -> list:
    return [e for f in sorted(glob.glob(pattern)) for e in json.load(open(f))]


def pick(rows: list, field: str, key: str) -> dict:
    hits = [r for r in rows if key in r[field]]
    if len(hits) != 1:
        sys.exit(f"build_cards: '{key}' matches {len(hits)} entries in {field}; make it unique")
    return hits[0]


def card(st: dict, ep: Path, ctx: dict) -> str:
    entries = [pick(ctx["verify"], "story", k) for k in st.get("verify", [])]
    legacy = any("quotes" not in v for v in entries)
    out = ["**Source facts** (" + ("as recorded by the verification step; quotes are verbatim, the rest are its "
                                   "paraphrases" if legacy else "verbatim, as recorded by the verification step") + "):"]
    visuals, gaps = [], []
    for v in entries:
        out.append(f"- From {v['primary_url']} ({v.get('primary_date')}):")
        for q in v.get("quotes") or v.get("facts") or []:  # "facts" is the pre-2026-11-16 contract
            out.append(f"  - {q['text'] + ' (' + q['url'] + ')' if isinstance(q, dict) else q}")
        visuals += v.get("visuals") or []
        gaps += v.get("gaps") or []
    for k in st.get("finds", []):
        f = pick(ctx["finds"], "event", k)
        out.append(f"- From {f['primary_url']} ({f['date']}), the web-discovery step's summary, not verbatim: {f['what']}")
    if gaps:
        out += ["\n**Not stated in the source:**"] + [f"- {g}" for g in gaps]

    kws = [k.lower() for k in st.get("keywords", [])]
    rel = lambda text: not kws or any(k in (text or "").lower() for k in kws)  # noqa: E731
    lo, hi = ctx["window"]
    xs, hn, yt = [], [], []
    for slug in st.get("chatter", []):
        d = json.load(open(ep / "chatter" / f"{slug}.json"))
        xs += [t for t in d["x"] if lo <= t["published"][:10] <= hi and rel(t["text"])]
        hn += [h for h in d["hn"] if lo <= (h.get("created") or "")[:10] <= hi and rel(h.get("title"))]
        yt += [v for v in d["youtube"] if rel(v.get("title"))]
    seen: set = set()
    xs = [t for t in sorted(xs, key=lambda t: -(t.get("likes") or 0)) if not (t["url"] in seen or seen.add(t["url"]))]
    excl = st.get("exclude", {})
    dropped, xs = [t for t in xs if t["url"] in excl], [t for t in xs if t["url"] not in excl]
    undated = [v for v in yt if not str(v.get("uploaded") or "").isdigit()]
    yt = [v for v in yt if str(v.get("uploaded") or "").isdigit() and lo.replace("-", "") <= v["uploaded"]]
    cluster = [v for v in ctx["verdicts"] if v["verdict"] in ("keep", "maybe")
               and any(k in v["event"].lower() for k in st.get("cluster", []))]

    out.append(f"\n**Signals** (raw counts; window {lo} to {hi}):")
    out.append(f"- Harvest items in this cluster (kept or maybe at triage): {len(cluster)}")
    if st.get("chatter"):
        best = max([ctx["pct"](t) for t in xs] or [0])
        out.append(f"- X posts found by chatter search: {len(xs)}; most likes: {xs[0]['likes'] if xs else 0}; "
                   f"best engagement percentile vs the week's harvest: {best:.2f}")
        out.append(f"- Hacker News stories: {len(hn)}; most points: {max([h['points'] or 0 for h in hn] or [0])}")
        out.append(f"- YouTube videos with the story in the title, uploaded in the window: {len(yt)}; total views: "
                   f"{sum(v.get('views') or 0 for v in yt):,}" + (f" ({len(undated)} more matched, upload date unknown, "
                                                                 f"not counted)" if undated else ""))
        out += [f"- Excluded as off-topic: X post by @{t['handle']} ({t['likes']:,} likes): {excl[t['url']]}"
                for t in dropped]
    else:
        out.append("- No chatter search was run for this story.")

    out.append("\n**What people say** (verbatim, attributed; most-liked first):")
    n = len(out)
    out += [f"- X @{t['handle']} ({t.get('followers') or 0:,} followers, {t.get('likes') or 0:,} likes, "
            f"{t['published'][:10]}): \"{' '.join(t['text'].split())[:280]}\"" for t in xs[:4]]
    out += [f"- Hacker News story \"{h['title']}\" ({h['points']} points, {h.get('comments')} comments)"
            for h in sorted(hn, key=lambda h: -(h.get("points") or 0))[:2]]
    for v in sorted(yt, key=lambda v: -(v.get("views") or 0))[:3]:
        out.append(f"- YouTube \"{v['title']}\" ({v.get('channel')}, {v.get('views') or 0:,} views)")
        out += [f"  - comment ({c['likes']} likes): \"{' '.join(c['text'].split())[:200]}\""
                for c in [c for c in v.get("top_comments") or []
                          if (c.get("likes") or 0) >= MIN_COMMENT_LIKES and "http" not in c["text"]][:2]]
    if len(out) == n:
        out.append("- No posts or comments found.")

    out.append("\n**Visuals listed by the verification step:** " + ("; ".join(visuals) if visuals else "none recorded"))
    days = (Date.fromisoformat(ctx["today"]) - Date.fromisoformat(st["date"])).days
    out.append(f"\n**Primary date:** {st['date']} ({days} days before {ctx['today']})")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ep", type=Path)
    ap.add_argument("--today", default=Date.today().isoformat())
    args = ap.parse_args()
    ep = args.ep
    harvest = json.load(open(ep / "candidates.json"))
    xeng = sorted(engagement(t) for t in harvest if t["type"] == "x")
    lo = (Date.fromisoformat(args.today) - timedelta(days=7)).isoformat()
    ctx = {"verify": load(str(ep / "verify/*.json")), "finds": load(str(ep / "web/finds_*.json")),
           "verdicts": load(str(ep / "triage/verdicts_*.json")), "today": args.today, "window": (lo, args.today),
           "pct": lambda t: bisect.bisect_left(xeng, engagement(t)) / len(xeng)}
    stories = json.load(open(ep / "judge" / "stories.json"))
    cards = [{"id": s["id"], "story": s["label"], "type": s["type"], "company": s["company"],
              "card": card(s, ep, ctx)} for s in stories]
    (ep / "judge" / "cards.json").write_text(json.dumps(cards, indent=1, ensure_ascii=False) + "\n")
    (ep / "judge" / "cards.md").write_text("\n\n".join(
        f"## {c['id']}: {c['story']}\nType: {c['type']} | Company: {c['company']}\n\n{c['card']}" for c in cards) + "\n")
    sizes = [len(c["card"]) for c in cards]
    print(f"{len(cards)} cards, {min(sizes)}-{max(sizes)} characters")
    for c in cards:
        if len(c["card"]) > 2 * sorted(sizes)[len(sizes) // 2]:
            print(f"  {c['id']} is over twice the median length: split the story or trim its sources")


if __name__ == "__main__":
    main()
