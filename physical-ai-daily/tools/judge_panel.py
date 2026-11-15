#!/usr/bin/env python3
"""LLM judge panel for the research skill's ranking step: prepare the judges' files, then aggregate their verdicts.

Usage:
  python3 -I tools/judge_panel.py prepare <ep>/judge/cards.json <ep>/judge [--judges 3]
  python3 -I tools/judge_panel.py aggregate <ep>/judge [--picks 5]
  python3 -I tools/judge_panel.py pairs <ep>/judge [--picks 5 --width 2]
  python3 -I tools/judge_panel.py settle <ep>/judge

cards.json is one card per story cluster, written by the skill after clustering, chatter and verification:
  [{"id": "s01", "story": "...", "type": "deployment|release|funding|paper|policy|opinion", "company": "...",
    "card": "<markdown built by tools/build_cards.py: source facts, gaps, signals, verbatim reactions, visuals>"}]
Cards are generated from stored sources by tools/build_cards.py, never written by hand, so the judges see the
sources' words and not the researcher's opinions.

Why a panel and not one judge: a single LLM judge is biased by the order it reads candidates in and by the order of
the rubric's criteria, and different models have different biases, so averaging independent judges dilutes them.
Each judge here gets the cards in its own shuffled order and the criteria in its own order, and judges alone. The
cut line, where one place decides whether a story makes the video, is settled by pairwise comparisons asked in both
orders; a win counts only when it holds both ways.

  prepare    writes judge_<X>.md for each judge (rubric, cards, output contract) to be read by one subagent each,
             which writes judge_<X>.json
  aggregate  combines judge_*.json into panel.json: mean overall score, mean per criterion, spread between judges,
             Borda points from each judge's top picks; prints the ranking and flags disagreements
  pairs      writes pairwise.md: every pair among the stories around the cut line, in both orders, shuffled
  settle     reads pairwise.json and re-orders the cut-line stories by consistent pairwise wins; updates panel.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import string
from pathlib import Path

CRITERIA = {
    "significance": "How much changed for machines that act in the physical world: a capability jump, a deployment "
                    "at real scale, serious money, a first, an exit or reversal. 1 = incremental; 10 = the field will "
                    "still talk about it in a year. Reward concrete, specific change; a technically novel but vaguely "
                    "described claim scores lower than a concrete, high-impact one.",
    "viewer_pull": "Would our viewer (slight technical background, little knowledge of the field) care and click: it "
                   "touches what they could see, buy, use or worry about, involves an actor they know, or surprises. "
                   "1 = only specialists care; 10 = they'd tell a friend.",
    "evidence": "Strength of the best primary evidence on the card: an official source with concrete numbers or a "
                "demo scores high; vendor-only claims, vague announcements, 'in talks' or unconfirmed reports score "
                "low. 1 = rumour; 10 = primary source, specific numbers, independently corroborated.",
    "showability": "Can we show and explain it in about 90 seconds: real footage, a figure or demo, an everyday "
                   "analogy for the idea. 1 = text only and hard to explain; 10 = striking footage and a clear "
                   "analogy.",
    "momentum": "Is the field paying attention right now, from the card's signals (X engagement, paper prior, "
                "independent sources, YouTube coverage and views). YouTube runs on trends. 1 = nobody noticed; "
                "10 = everyone is talking about it.",
}

CONTRACT = """Write your verdicts as JSON to {out}:
{{"judge": "{name}",
 "scores": [{{"id": "s01", "reasoning": "<2-4 sentences, written BEFORE the numbers>",
             {fields}, "overall": <1-10>}}],
 "top": ["<id>", ... your {picks} picks for the video, best first]}}
Score every card. "overall" is not an average: it is how strongly this story should be one of the week's {picks},
weighing everything above. Use the full 1-10 range, and give different stories different scores where you can."""


def prepare(cards_path: Path, out: Path, judges: int, picks: int) -> None:
    cards = json.loads(cards_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    for name in string.ascii_uppercase[:judges]:
        rng = random.Random(f"{cards_path.parent}-{name}")
        order = cards[:]
        rng.shuffle(order)
        crit = list(CRITERIA)
        rng.shuffle(crit)
        rubric = "\n".join(f"- **{c}** (1-10): {CRITERIA[c]}" for c in crit)
        body = "\n\n".join(f"## {c['id']}: {c['story']}\nType: {c['type']} | Company: {c.get('company') or '-'}\n\n"
                           f"{c['card']}" for c in order)
        fields = ", ".join(f'"{c}": <1-10>' for c in crit)
        (out / f"judge_{name}.md").write_text(
            f"# Story judge {name}\n\nYou are one of {judges} independent judges choosing the {picks} stories for "
            f"this week's Pixels2Actions video: a weekly YouTube roundup of physical AI (robots, humanoids, embodied "
            f"and world models, autonomy) for viewers with a slight technical background but little knowledge of the "
            f"field. The question every story answers: what changed in machines that act in the physical world, and "
            f"why it matters.\n\nJudge only from the cards. Each card's facts were checked against the sources it "
            f"cites, and what a source leaves out is listed under \"Not stated in the source\": treat a claim with a "
            f"listed gap as unconfirmed. Every opinion on a card is quoted and attributed; a single post or comment is "
            f"one person's view, not a consensus, and a company praising its own product is not evidence. Story labels "
            f"are deliberately plain; judge the facts, not the label. Card order is random and means nothing. Don't "
            f"favour a card for being longer or more enthusiastic.\n\n## Criteria\n\n{rubric}\n\n"
            f"## Output\n\n{CONTRACT.format(out=out / f'judge_{name}.json', name=name, fields=fields, picks=picks)}"
            f"\n\n# Cards ({len(order)})\n\n{body}\n")
        print(f"judge_{name}.md: {len(order)} cards, criteria order {', '.join(crit)}")


def aggregate(out: Path, picks: int) -> list[dict]:
    cards = {c["id"]: c for c in json.loads((out / "cards.json").read_text())}
    verdicts = [json.loads(p.read_text()) for p in sorted(out.glob("judge_*.json"))]
    rows = []
    for cid, c in cards.items():
        got = [s for v in verdicts for s in v["scores"] if s["id"] == cid]
        if not got:
            continue
        overall = [s["overall"] for s in got]
        borda = sum(picks - v["top"].index(cid) for v in verdicts if cid in v["top"][:picks])
        rows.append({"id": cid, "story": c["story"], "type": c["type"], "company": c.get("company"),
                     "overall": round(statistics.mean(overall), 2), "spread": max(overall) - min(overall),
                     "borda": borda, "votes": sum(cid in v["top"][:picks] for v in verdicts),
                     "criteria": {k: round(statistics.mean(s[k] for s in got), 2) for k in CRITERIA},
                     "by_judge": {v["judge"]: next((s["overall"] for s in v["scores"] if s["id"] == cid), None)
                                  for v in verdicts}})
    rows.sort(key=lambda r: (-r["overall"], -r["borda"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    (out / "panel.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"{len(verdicts)} judges, {len(rows)} stories")
    for r in rows:
        flag = "  <- judges disagree" if r["spread"] >= 3 else ""
        print(f"{r['rank']:>2}. {r['overall']:>5} (borda {r['borda']:>2}, votes {r['votes']}) {r['story']}"
              f"  {r['by_judge']}{flag}")
    return rows


def pairs(out: Path, picks: int, width: int) -> None:
    rows = json.loads((out / "panel.json").read_text())
    cards = {c["id"]: c for c in json.loads((out / "cards.json").read_text())}
    band = rows[max(0, picks - width):picks + width]
    asks = [(a["id"], b["id"]) for a, b in itertools.combinations(band, 2)]
    asks += [(b, a) for a, b in asks]
    random.Random(str(out)).shuffle(asks)
    body = "\n\n".join(f"## p{i:02d}\n**First:** {a}: {cards[a]['story']}\n**Second:** {b}: {cards[b]['story']}"
                       for i, (a, b) in enumerate(asks, 1))
    card_text = "\n\n".join(f"## {cid}: {cards[cid]['story']}\n{cards[cid]['card']}" for cid in sorted(r["id"] for r in band))
    (out / "pairwise.md").write_text(
        f"# Pairwise judge\n\nThese stories sit around the cut line for this week's {picks}-story Pixels2Actions "
        f"video (weekly physical-AI roundup for viewers with a slight technical background). For each pair, decide "
        f"which story more deserves a place in the video: significance, viewer pull, evidence, showability and "
        f"momentum, weighed together. Judge only from the cards: their facts were checked against the cited sources, "
        f"listed gaps mean unconfirmed, and every opinion is quoted and attributed (one post is one person's view). "
        f"Order within a pair is random and means nothing; some pairs repeat in the other order, judge each on its own.\n\nWrite JSON to {out / 'pairwise.json'}:\n"
        f'[{{"pair": "p01", "reasoning": "<1-2 sentences first>", "winner": "first" | "second" | "tie"}}]\n\n'
        f"# Pairs ({len(asks)})\n\n{body}\n\n# Cards\n\n{card_text}\n")
    (out / "pairwise_index.json").write_text(json.dumps({f"p{i:02d}": [a, b] for i, (a, b) in enumerate(asks, 1)}))
    print(f"pairwise.md: {len(band)} stories around the cut, {len(asks)} comparisons")


def settle(out: Path) -> None:
    rows = json.loads((out / "panel.json").read_text())
    index = json.loads((out / "pairwise_index.json").read_text())
    said = {}
    for v in json.loads((out / "pairwise.json").read_text()):
        a, b = index[v["pair"]]
        said[(a, b)] = a if v["winner"] == "first" else b if v["winner"] == "second" else None
    wins: dict[str, float] = {}
    for (a, b), w in said.items():
        if a > b or (b, a) not in said:
            continue
        w2 = said[(b, a)]
        for cid in (a, b):
            wins.setdefault(cid, 0)
        if w and w == w2:  # consistent in both orders
            wins[w] += 1
        else:  # split or tie: half each
            wins[a] += 0.5
            wins[b] += 0.5
    band = [r for r in rows if r["id"] in wins]
    start = min(r["rank"] for r in band) - 1
    band.sort(key=lambda r: (-wins[r["id"]], -r["overall"]))
    for i, r in enumerate(band):
        r["pairwise_wins"] = wins[r["id"]]
        r["rank"] = start + i + 1
    rows.sort(key=lambda r: r["rank"])
    (out / "panel.json").write_text(json.dumps(rows, indent=1) + "\n")
    for r in rows:
        extra = f"  pairwise wins {r['pairwise_wins']}" if "pairwise_wins" in r else ""
        print(f"{r['rank']:>2}. {r['overall']:>5} {r['story']}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "aggregate", "pairs", "settle"])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--judges", type=int, default=3)
    ap.add_argument("--picks", type=int, default=5)
    ap.add_argument("--width", type=int, default=2)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.paths[0], args.paths[1], args.judges, args.picks)
    elif args.cmd == "aggregate":
        aggregate(args.paths[0], args.picks)
    elif args.cmd == "pairs":
        pairs(args.paths[0], args.picks, args.width)
    else:
        settle(args.paths[0])


if __name__ == "__main__":
    main()
