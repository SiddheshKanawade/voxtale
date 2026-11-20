#!/usr/bin/env python3
"""Write <ep>/research.md and <ep>/scores.json from the judge panel, the verification records and the cards.

Usage: python3 -I tools/research_md.py <ep> --window FROM:TO [--picks 5]

Everything in the shortlist comes from stored sources: the panel's ranking and the judges' own reasoning (quoted),
verbatim quotes and gaps from <ep>/verify/*.json, and the card's attributed reactions and links. Nothing is
summarised in the researcher's words, so the sourcing rule holds here as on the cards. The scripting step writes the
summaries, explanations and analogies, from these sources.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

MIN_LEARNING = 2


def section(card: str, name: str) -> list[str]:
    m = re.search(rf"\*\*{re.escape(name)}\*\*[^\n]*\n(.*?)(?=\n\*\*|\Z)", card, re.S)
    return [l for l in (m.group(1).strip().splitlines() if m else []) if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ep", type=Path)
    ap.add_argument("--window", required=True)
    ap.add_argument("--picks", type=int, default=5)
    ap.add_argument("--models", default="Opus, Sonnet and Fable",
                    help="the models that judged, as written in research.md (e.g. when a fallback model stood in)")
    ap.add_argument("--min-learning", type=int, default=MIN_LEARNING,
                    help="backlog weeks (published months later) use 3: learning stories age better than news")
    args = ap.parse_args()
    ep, J = args.ep, args.ep / "judge"
    panel = json.loads((J / "panel.json").read_text())
    cards = {c["id"]: c for c in json.loads((J / "cards.json").read_text())}
    stories = {s["id"]: s for s in json.loads((J / "stories.json").read_text())}
    verify = [e for f in sorted(glob.glob(str(ep / "verify/*.json"))) for e in json.loads(Path(f).read_text())]
    # raw_decode, as in judge_panel.py: a stray "}" or note after a judge's JSON object is ignored
    judges = [json.JSONDecoder().raw_decode(Path(p).read_text().lstrip())[0] for p in sorted(glob.glob(str(J / "judge_*.json")))]
    reason = {(v["judge"], s["id"]): s["reasoning"] for v in judges for s in v["scores"]}
    # Editorial rule: at least MIN_LEARNING of the picks are "learning" stories (a new capability, method or
    # experiment with a result). Short of that, the lowest-ranked news pick gives way to the best-ranked learning story.
    kind = lambda r: stories[r["id"]].get("kind", "news")  # noqa: E731
    picks, rest, moved = panel[:args.picks], panel[args.picks:], []
    while sum(kind(r) == "learning" for r in picks) < args.min_learning and any(kind(r) == "learning" for r in rest):
        out_r = max((r for r in picks if kind(r) != "learning"), key=lambda r: r["rank"])
        in_r = min((r for r in rest if kind(r) == "learning"), key=lambda r: r["rank"])
        picks = [in_r if r is out_r else r for r in picks]
        rest = [out_r if r is in_r else r for r in rest]
        moved.append(f"{in_r['story']} (rank {in_r['rank']}, learning) replaces {out_r['story']} (rank {out_r['rank']}, news)")
    picks.sort(key=lambda r: r["rank"])
    rest.sort(key=lambda r: r["rank"])

    out = [f"# Research {ep.name} (window: {args.window.replace(':', ' → ')})", "",
           f"Ranked by the judge panel: {len(judges)} passes ({args.models}, two card orders each), cut line "
           f"settled pairwise in both orders. Cards: `judge/cards.md`. Everything below is quoted from stored sources; "
           f"summaries and explanations are written at the scripting step.", "",
           "## Shortlist", "",
           "| # | Story | Kind | Panel | Sig | Pull | Evid | Show | Mom | Judges' lineups | Primary source (date) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in picks:
        c, st = r["criteria"], stories[r["id"]]
        v = next(e for e in verify if e["story"].startswith(r["id"]))
        out.append(f"| {r['rank']} | {r['story']} | {kind(r)} | {r['overall']} (spread {r['spread']}) | {c['significance']} | "
                   f"{c['viewer_pull']} | {c['evidence']} | {c['showability']} | {c['momentum']} | "
                   f"{r['lineup_votes']}/{len(judges)} | {v['primary_url']} ({st['date']}) |")
    out += ["", f"Editorial rule (at least {args.min_learning} learning stories): "
            + ("; ".join(moved) if moved else "already met, nothing moved") + "."]
    for r in picks:
        card = cards[r["id"]]["card"]
        v = next(e for e in verify if e["story"].startswith(r["id"]))
        out += ["", f"### {r['rank']}. {r['story']}", "", f"Status: **{v.get('status')}**. Primary: {v['primary_url']}", "",
                "**Source quotes:**"]
        out += [f"- \"{q['text']}\" ({q['url']})" if isinstance(q, dict) else f"- {q}"
                for q in (v.get("quotes") or v.get("facts") or [])]
        if v.get("corrections"):
            out += ["", "**Corrections:**"] + [f"- {x}" for x in v["corrections"]]
        if v.get("gaps"):
            out += ["", "**Not stated in the source:**"] + [f"- {g}" for g in v["gaps"]]
        out += ["", "**What people say** (from the card):"] + (section(card, "What people say") or ["- none found"])
        out += ["", "**What the judges wrote:**"]
        for j in sorted({v["judge"] for v in judges}):
            if (j, r["id"]) in reason:
                out.append(f"- {j}: \"{reason[(j, r['id'])]}\"")
        out += ["", "**Visuals:** " + "; ".join(v.get("visuals") or ["none recorded"])]
        out += ["", "**Links:**"] + section(card, "Links")
    out += ["", "## Reserves", ""]
    out += [f"- {r['rank']}. {r['story']}: panel {r['overall']}"
            + (f", pairwise wins {r['pairwise_wins']}" if "pairwise_wins" in r else "") for r in rest[:3]]
    out += ["", "## Not picked (judged)", ""]
    out += [f"- {r['rank']}. {r['story']} ({r['overall']})" for r in rest[3:]]
    out += ["", "## Calibration", "", "Uncalibrated until the episode's YouTube feedback is in `youtube.json` "
            "(see `docs/youtube-feedback.md`)." if not (ep / "youtube.json").exists() else
            "YouTube feedback: `youtube.json`."]
    (ep / "research.md").write_text("\n".join(out) + "\n")
    chosen = {r["id"] for r in picks}
    rows = [{"story": r["story"], "type": r["type"], "kind": kind(r), "picked": r["id"] in chosen, "rank": r["rank"], "overall": r["overall"],
             "spread": r["spread"], "order_spread": r.get("order_spread"), "by_judge": r["by_judge"],
             "criteria": r["criteria"], "pairwise_wins": r.get("pairwise_wins"),
             "segment": f"story-{[p['id'] for p in picks].index(r['id']) + 1}" if r["id"] in chosen else None}
            for r in panel]
    (ep / "scores.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print(f"research.md: {len(picks)} picks, {len(panel) - len(picks)} others")


if __name__ == "__main__":
    main()
