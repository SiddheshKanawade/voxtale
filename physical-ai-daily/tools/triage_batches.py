#!/usr/bin/env python3
"""Split a harvest into reading batches for the research skill's triage subagents.

Usage: python3 -I tools/triage_batches.py <ep>/candidates.json <ep>/triage [--papers 120 --x 300 --news 150]

Each batch is one source, so a reader judges like with like, and carries everything needed to judge an item without
opening it: the full abstract (with authors' comment, organisations and signals) for papers, the full post (with its
links and engagement) for X, the headline and description for news. Items are ordered by their prior so the strongest
come first. Writes batch_NN_<source>.md files plus index.json (batch -> item urls). Batch sizes keep each file around
30-40k tokens, so a weekly harvest of ~1,500 items makes about 8 batches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def paper(p: dict) -> str:
    s = p.get("signals") or {}
    meta = [f"prior {p.get('prior', 0):.2f}", f"HF upvotes {s.get('hf_upvotes', 0)}",
            f"alphaXiv votes {s.get('ax_votes', 0)} / visits {s.get('ax_visits_7d', 0)}",
            f"X accounts {s.get('x_accounts', 0)}", f"stars {s.get('stars')}" if s.get("github") else "no code",
            f"max author h {s.get('max_h', 0)}"]
    lines = [f"### {p['url']}", f"**{p['title']}** ({p['published'][:10]})",
             f"Orgs: {', '.join(s.get('orgs') or []) or 'unknown'} | Authors: {', '.join(p.get('authors', [])[:6])}"
             + (" et al." if len(p.get("authors", [])) > 6 else ""),
             " | ".join(meta) + (f" | venue: {s['venue']}" if s.get("venue") else ""),
             p.get("text", "")]
    if p.get("comment"):
        lines.append(f"Comment: {p['comment']}")
    return "\n".join(lines)


def post(t: dict) -> str:
    meta = f"@{t['handle']} ({t.get('followers') or '?'} followers) {t['published'][:10]} | likes {t.get('likes')} | " \
           f"reposts {t.get('retweets')} | eng pct {t.get('eng_pct', 0):.2f}" + (" | media" if t.get("has_media") else "")
    links = f"\nLinks: {' '.join(t['links'])}" if t.get("links") else ""
    return f"### {t['url']}\n{meta}\n{t['text']}{links}"


NEWS_TEXT_MAX = 1500  # some saved pulls carry a whole scraped page (100k+ chars), which made one batch ~500k tokens


def article(n: dict) -> str:
    text = n.get("text") or "(no description)"
    if len(text) > NEWS_TEXT_MAX:
        text = text[:NEWS_TEXT_MAX] + " [truncated; open the URL for the rest]"
    return f"### {n['url']}\n**{n.get('title')}** ({n.get('outlet')}, {n['published'][:10]})\n{text}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--papers", type=int, default=120)
    ap.add_argument("--x", type=int, default=300)
    ap.add_argument("--news", type=int, default=150)
    args = ap.parse_args()
    items = json.loads(args.candidates.read_text())
    groups = [
        ("papers", sorted((i for i in items if i["type"] == "paper"), key=lambda p: -p.get("prior", 0)), paper, args.papers),
        ("x", sorted((i for i in items if i["type"] == "x"), key=lambda t: -t.get("eng_pct", 0)), post, args.x),
        ("news", sorted((i for i in items if i["type"] == "news"), key=lambda n: n["published"], reverse=True),
         article, args.news),
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    index, n = {}, 0
    for name, rows, fmt, size in groups:
        for i in range(0, len(rows), size):
            n += 1
            chunk = rows[i:i + size]
            path = args.out / f"batch_{n:02d}_{name}.md"
            for it in chunk:  # cross-source context the reader can't see from one batch
                if it.get("mentions"):
                    it.setdefault("_note", f"linked by {len(it['mentions'])} other items")
            body = "\n\n".join(fmt(it) + (f"\n_{it['_note']}_" if it.get("_note") else "")
                               + (f"\nDigest topics (lexical): {', '.join(it['topics'])}" if it.get("topics") else "")
                               for it in chunk)
            path.write_text(f"# Triage batch {n}: {name} ({len(chunk)} items)\n\n{body}\n")
            index[path.name] = [it["url"] for it in chunk]
            print(f"{path.name}: {len(chunk)} items, ~{len(body) // 4 // 1000}k tokens")
    (args.out / "index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
