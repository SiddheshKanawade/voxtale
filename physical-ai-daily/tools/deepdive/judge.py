#!/usr/bin/env python3
"""Blind judge panel that picks the week's deep-dive story (skill: physical-ai-deep-dive).

Usage:
  python3 -I tools/deepdive/judge.py prepare <dd> --week <weekly-ep> --ids s04,s05,s07 [--audit <dd>/audit.md]
  python3 -I tools/deepdive/judge.py aggregate <dd>
  python3 -I tools/deepdive/judge.py pairs <dd>
  python3 -I tools/deepdive/judge.py settle <dd>

Only stories that passed the five gates get a card. Each card is the digest card the weekly research built from stored
sources (<weekly-ep>/judge/cards.json, never written by hand) plus that story's section of the depth audit, so the
judges see the sources' facts and the audit's counts, not the researcher's opinion.

Same design as the digest panel (tools/judge_panel.py, docs/judge-evidence.md): three models, each in two shuffled
orderings of cards and criteria, reasoning before numbers, anchored 1-10 scales. Differences, as the user approved on
2026-11-24: the criteria are D1-D8 for a single 12-20 minute story, the rank is the plain mean of D1-D7 (equal
weights until calibrated on 6 episodes), and D8 momentum only breaks ties. The top two are then compared pairwise in
both orders by one fresh judge; a win counts only when it holds both ways, otherwise the panel order stands.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

# Anchors at 1, 3, 5, 8 and 10 so judges share one scale (the digest's first run bunched without them).
CRITERIA = {
    "D1_mechanism_depth": "How much real 'how' the sources contain: architecture, data, training recipe, control loop, "
        "hardware, and ablations that show which parts matter. 1 = only a result, no method; 3 = the method in a "
        "paragraph, no ablations; 5 = a full method section and one or two ablations; 8 = a detailed method, ablations "
        "that isolate the key parts, an appendix; 10 = all of that plus a failure analysis and enough detail to "
        "reproduce it.",
    "D2_surprise": "A sourced result that contradicts what a practitioner would predict, or a common belief the story "
        "corrects. 1 = what any practitioner would expect; 3 = a modest gain in the expected direction; 5 = a result "
        "a practitioner might doubt; 8 = a result most practitioners would have predicted the other way, shown with "
        "evidence; 10 = it overturns a belief the field widely holds, with strong evidence.",
    "D3_known_subject": "Viewers know the name (lab, robot, model, product, a widely shared demo, or the problem "
        "itself), but the 'how' has not been explained elsewhere. 1 = an unknown team on an unfamiliar problem; 3 = a "
        "niche problem specialists know; 5 = a known lab, robot, model or problem; 8 = a name or demo many engineers "
        "have seen, with no existing explanation of how it works; 10 = a famous subject people are asking about, "
        "and nobody has explained the how.",
    "D4_story_arc": "A real sequence from the sources: the problem, earlier attempts, the turn(s), the result, what is "
        "still open, with stakes for the field. 1 = a single result, no tension; 3 = problem and result, no obstacle; "
        "5 = problem, approach, one obstacle, result; 8 = earlier attempts fell short, the team hit failures and "
        "changed course, then the result and open problems; 10 = several sourced turns or reversals with clear stakes.",
    "D5_practical_transfer": "Something an engineer can use: released code, weights or data, a reproducible recipe, a "
        "design rule, cost or hardware numbers. 1 = nothing usable; 3 = general lessons only; 5 = a recipe, design "
        "rule or released data; 8 = code and data or weights under an open licence, plus cost, time or hardware "
        "numbers; 10 = an engineer could put it to use this week.",
    "D6_durability": "Still worth watching in 6-12 months. 1 = stale within a month; 3 = a follow-up in a line already "
        "explained elsewhere; 5 = a solid instance of a known method class; 8 = the first clear instance of a new "
        "method class, or a foundational result; 10 = likely the reference people cite for years.",
    "D7_visual_depth": "Beyond the minimum: success and failure clips, simulation-and-real pairs, figures that can be "
        "animated into our own diagrams. Decorative footage doesn't count. 1 = only the gate minimum; 3 = some "
        "footage, mostly decorative; 5 = real footage of success and several figures; 8 = success and failure clips, "
        "sim-and-real pairs, animatable figures; 10 = every chapter can have its own distinct visual.",
    "D8_momentum": "Attention this week, from the card's signals only (tie-breaker; it does not count toward the "
        "rank). 1 = nobody noticed; 3 = a few specialists posted; 5 = the field's main accounts or outlets covered "
        "it; 8 = widely shared beyond robotics; 10 = everyone is talking about it.",
}
RANKED = [c for c in CRITERIA if c != "D8_momentum"]

BRIEF = """**The video.** One story, 12-20 minutes, on the Pixels2Actions channel (robotics and physical AI). The viewer
is an engineer working in the field: full technical depth, explained so it's easy to follow, told as a story, never a
lecture. Every fact in the video comes from a source; nothing is invented.

**Your job.** Score each story on the criteria below. Every story here already passed five gates: a readable technical
primary source, verified quotes, enough footage and figures to show, one central question the sources answer, and at
least five sourced sub-topics. So judge which one makes the best deep dive, not whether it's true.

**The cards.** Judge only from the cards. Each has the digest card (verbatim source facts, gaps, attention signals,
quoted reactions) and a depth audit (counts of pages, figures, clips, code, lineage; facts only). What a source leaves
out is listed as a gap: treat it as unknown. Card order is random and means nothing. Don't favour a card for being
longer or more enthusiastic."""

CONTRACT = """Write your verdicts as JSON to {out}:
{{"judge": "{name}",
 "scores": [{{"id": "s01", "reasoning": "<3-5 sentences, written BEFORE the numbers>", {fields}, "overall": <1-10>}}],
 "top": ["<id>", ... every story, best deep dive first]}}
Score every card on every criterion, using the anchors and the full range. "overall" is how strongly this story should
be this week's deep dive; it is a sanity check, not the rank (the rank is computed from D1-D7)."""


def section(md: str, story: str) -> str:
    """The audit.md section whose '## N. <title>' heading contains `story` (case-insensitive)."""
    parts = re.split(r"(?m)^(?=## \d+\. )", md)
    hits = [p for p in parts if p.startswith("## ") and story.lower() in p.splitlines()[0].lower()]
    if len(hits) != 1:
        raise SystemExit(f"audit section for {story!r}: {len(hits)} matches")
    return hits[0].split("\n", 1)[1].strip()


def prepare(dd: Path, week: Path, ids: list[str], audit: Path, keys: list[str]) -> None:
    cards = {c["id"]: c for c in json.loads((week / "judge" / "cards.json").read_text())}
    md = audit.read_text()
    out = dd / "judge"
    out.mkdir(parents=True, exist_ok=True)
    deck = []
    for cid, key in zip(ids, keys):
        c = cards[cid]
        deck.append({"id": cid, "story": c["story"], "type": c["type"], "company": c.get("company"),
                     "card": f"{c['card']}\n\n### Depth audit (facts only, {audit.name})\n\n{section(md, key)}"})
    (out / "cards.json").write_text(json.dumps(deck, indent=1) + "\n")
    for name in (f"{m}{o}" for m in "ABC" for o in (1, 2)):
        rng = random.Random(f"{dd.name}-{name}")
        order = deck[:]
        rng.shuffle(order)
        crit = list(CRITERIA)
        rng.shuffle(crit)
        rubric = "\n".join(f"- **{c}** (1-10): {CRITERIA[c]}" for c in crit)
        fields = ", ".join(f'"{c}": <1-10>' for c in crit)
        body = "\n\n".join(f"## {c['id']}: {c['story']}\nType: {c['type']} | Company: {c.get('company') or '-'}\n\n"
                           f"{c['card']}" for c in order)
        (out / f"judge_{name}.md").write_text(
            f"# Deep-dive judge {name}\n\nYou are one of three independent judges choosing this week's single "
            f"deep-dive story.\n\n{BRIEF}\n\n## Criteria\n\n{rubric}\n\n## Output\n\n"
            f"{CONTRACT.format(out=out / f'judge_{name}.json', name=name, fields=fields)}\n\n# Cards ({len(order)})\n\n{body}\n")
        print(f"judge_{name}.md: {len(order)} cards, criteria order {', '.join(c.split('_')[0] for c in crit)}")


def aggregate(dd: Path) -> list[dict]:
    out = dd / "judge"
    cards = {c["id"]: c for c in json.loads((out / "cards.json").read_text())}
    verdicts = [json.JSONDecoder().raw_decode(p.read_text().lstrip())[0] for p in sorted(out.glob("judge_*.json"))]
    rows = []
    for cid, c in cards.items():
        per_model: dict[str, list[float]] = {}
        crit: dict[str, list[float]] = {}
        for v in verdicts:
            for s in v["scores"]:
                if s["id"] != cid:
                    continue
                per_model.setdefault(v["judge"][0], []).append(statistics.mean(s[k] for k in RANKED))
                for k in CRITERIA:
                    crit.setdefault(k, []).append(s[k])
        model_means = [statistics.mean(x) for x in per_model.values()]  # one number per model, so orders don't outvote
        rows.append({"id": cid, "story": c["story"], "score": round(statistics.mean(model_means), 2),
                     "spread": round(max(model_means) - min(model_means), 2),
                     "order_spread": round(max(max(x) - min(x) for x in per_model.values()), 2),
                     "momentum": round(statistics.mean(crit["D8_momentum"]), 2),
                     "criteria": {k: round(statistics.mean(x), 2) for k, x in crit.items()},
                     "by_judge": {v["judge"]: next((round(statistics.mean(s[k] for k in RANKED), 2)
                                                    for s in v["scores"] if s["id"] == cid), None) for v in verdicts},
                     "overall_by_judge": {v["judge"]: next((s["overall"] for s in v["scores"] if s["id"] == cid), None)
                                          for v in verdicts}})
    rows.sort(key=lambda r: (-r["score"], -r["momentum"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    (out / "panel.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"{len(verdicts)} judges, {len(rows)} stories (score = mean of D1-D7; D8 breaks ties)")
    for r in rows:
        flag = "  <- models differ by 3+" if r["spread"] >= 3 else ""
        print(f"{r['rank']}. {r['score']:>5} (D8 {r['momentum']}) {r['story']}  {r['by_judge']}{flag}")
        print("   " + "  ".join(f"{k.split('_')[0]} {v}" for k, v in r["criteria"].items()))
    return rows


def pairs(dd: Path) -> None:
    out = dd / "judge"
    rows = json.loads((out / "panel.json").read_text())
    cards = {c["id"]: c for c in json.loads((out / "cards.json").read_text())}
    a, b = rows[0]["id"], rows[1]["id"]
    asks = [(a, b), (b, a)]
    body = "\n\n".join(f"## p{i:02d}\n**First:** {x}: {cards[x]['story']}\n**Second:** {y}: {cards[y]['story']}"
                       for i, (x, y) in enumerate(asks, 1))
    card_text = "\n\n".join(f"## {cid}: {cards[cid]['story']}\n{cards[cid]['card']}" for cid in sorted((a, b)))
    (out / "pairwise.md").write_text(
        f"# Pairwise deep-dive judge\n\nTwo stories are close for this week's single deep dive. For each pair, decide "
        f"which makes the better deep dive.\n\n{BRIEF}\n\n## Criteria\n\n"
        + "\n".join(f"- **{c}**: {d}" for c, d in CRITERIA.items()) +
        f"\n\nOrder within a pair is random and means nothing; the pair repeats in the other order: judge each on its "
        f"own.\n\nWrite JSON to {out / 'pairwise.json'}:\n"
        f'[{{"pair": "p01", "reasoning": "<2-3 sentences first>", "winner": "first" | "second" | "tie"}}]\n\n'
        f"# Pairs ({len(asks)})\n\n{body}\n\n# Cards\n\n{card_text}\n")
    (out / "pairwise_index.json").write_text(json.dumps({f"p{i:02d}": list(p) for i, p in enumerate(asks, 1)}))
    print(f"pairwise.md: {a} vs {b}, both orders")


def settle(dd: Path) -> None:
    out = dd / "judge"
    rows = json.loads((out / "panel.json").read_text())
    index = json.loads((out / "pairwise_index.json").read_text())
    won = []
    for v in json.loads((out / "pairwise.json").read_text()):
        x, y = index[v["pair"]]
        won.append(x if v["winner"] == "first" else y if v["winner"] == "second" else None)
    if won[0] and won[0] == won[1] and won[0] != rows[0]["id"]:
        rows[0], rows[1] = rows[1], rows[0]
        rows[0]["rank"], rows[1]["rank"] = 1, 2
    verdict = won[0] if won[0] and won[0] == won[1] else None
    for r in rows:
        r["pairwise"] = ("won both orders" if r["id"] == verdict else "lost both orders" if verdict and r["rank"] <= 2
                         else "split or tie: panel order stands" if r["rank"] <= 2 else None)
    (out / "panel.json").write_text(json.dumps(rows, indent=1) + "\n")
    for r in rows:
        print(f"{r['rank']}. {r['score']:>5} {r['story']}  {r['pairwise'] or ''}")
    print(f"pick: {rows[0]['id']} {rows[0]['story']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "aggregate", "pairs", "settle"])
    ap.add_argument("dd", type=Path, help="the deep-dive episode folder")
    ap.add_argument("--week", type=Path, help="the weekly episode folder whose judge/cards.json holds the digest cards")
    ap.add_argument("--ids", help="comma-separated card ids of the stories that passed the gates")
    ap.add_argument("--audit-keys", help="comma-separated words that find each story's '## N.' section in audit.md")
    ap.add_argument("--audit", type=Path)
    a = ap.parse_args()
    if a.cmd == "prepare":
        ids = a.ids.split(",")
        prepare(a.dd, a.week, ids, a.audit or a.dd / "audit.md", a.audit_keys.split(","))
    else:
        {"aggregate": aggregate, "pairs": pairs, "settle": settle}[a.cmd](a.dd)


if __name__ == "__main__":
    main()
