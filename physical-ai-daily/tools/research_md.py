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


def section(card: str, name: str) -> list[str]:
    m = re.search(rf"\*\*{re.escape(name)}\*\*[^\n]*\n(.*?)(?=\n\*\*|\Z)", card, re.S)
    return [l for l in (m.group(1).strip().splitlines() if m else []) if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ep", type=Path)
    ap.add_argument("--window", required=True)
    ap.add_argument("--picks", type=int, default=5)
    args = ap.parse_args()
    ep, J = args.ep, args.ep / "judge"
    panel = json.loads((J / "panel.json").read_text())
    cards = {c["id"]: c for c in json.loads((J / "cards.json").read_text())}
    stories = {s["id"]: s for s in json.loads((J / "stories.json").read_text())}
    verify = [e for f in sorted(glob.glob(str(ep / "verify/*.json"))) for e in json.loads(Path(f).read_text())]
    judges = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(J / "judge_*.json")))]
    reason = {(v["judge"], s["id"]): s["reasoning"] for v in judges for s in v["scores"]}
    picks = panel[:args.picks]

    out = [f"# Research {ep.name} (window: {args.window.replace(':', ' → ')})", "",
           f"Ranked by the judge panel: {len(judges)} passes (Opus, Sonnet and Fable, two card orders each), cut line "
           f"settled pairwise in both orders. Cards: `judge/cards.md`. Everything below is quoted from stored sources; "
           f"summaries and explanations are written at the scripting step.", "",
           "## Shortlist", "",
           "| # | Story | Panel | Sig | Pull | Evid | Show | Mom | Judges' lineups | Primary source (date) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in picks:
        c, st = r["criteria"], stories[r["id"]]
        v = next(e for e in verify if e["story"].startswith(r["id"]))
        out.append(f"| {r['rank']} | {r['story']} | {r['overall']} (spread {r['spread']}) | {c['significance']} | "
                   f"{c['viewer_pull']} | {c['evidence']} | {c['showability']} | {c['momentum']} | "
                   f"{r['lineup_votes']}/{len(judges)} | {v['primary_url']} ({st['date']}) |")
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
            + (f", pairwise wins {r['pairwise_wins']}" if "pairwise_wins" in r else "") for r in panel[args.picks:args.picks + 3]]
    out += ["", "## Not picked (judged)", ""]
    out += [f"- {r['rank']}. {r['story']} ({r['overall']})" for r in panel[args.picks + 3:]]
    out += ["", "## Calibration", "", "Uncalibrated: the user's blind ranking for this week isn't recorded "
            "(`judge/calibration.json`)." if not (J / "calibration.json").exists() else
            "See `judge/calibration.json`."]
    (ep / "research.md").write_text("\n".join(out) + "\n")
    rows = [{"story": r["story"], "type": r["type"], "picked": i < args.picks, "rank": r["rank"], "overall": r["overall"],
             "spread": r["spread"], "order_spread": r.get("order_spread"), "by_judge": r["by_judge"],
             "criteria": r["criteria"], "pairwise_wins": r.get("pairwise_wins"),
             "segment": f"story-{i + 1}" if i < args.picks else None} for i, r in enumerate(panel)]
    (ep / "scores.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print(f"research.md: {len(picks)} picks, {len(panel) - len(picks)} others")


if __name__ == "__main__":
    main()
