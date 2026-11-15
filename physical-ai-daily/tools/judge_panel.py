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

# Each criterion has anchors at 1, 3, 5, 8 and 10, so judges share one scale instead of each inventing their own
# (anchored only at the ends, 11 of 19 stories bunched between 2.7 and 6 in the first run).
CRITERIA = {
    "significance": "How much changed for machines that act in the physical world. 1 = nothing new (a re-announcement, "
                    "a minor update); 3 = an incremental step (a modest funding round, a benchmark gain of a few "
                    "points, a small pilot); 5 = a real step for one company or one research line (a new model "
                    "version with clear gains, a first deployment at one site); 8 = a step the whole field notices (a "
                    "deployment at production scale, a capability few thought close, a major lab entering or leaving "
                    "robotics, $500M+); 10 = the field will still talk about it in a year. Reward concrete, specific "
                    "change: a technically novel but vaguely described claim scores below a concrete, high-impact one.",
    "viewer_pull": "Would our viewer (slight technical background, little knowledge of the field) care and click. "
                   "1 = only specialists care; 3 = interesting to tech-curious people who follow AI; 5 = touches "
                   "something the viewer knows (a big brand, a product category, their job); 8 = touches what they "
                   "could see, buy, use or worry about soon; 10 = they'd tell a friend. Shock or outrage alone isn't "
                   "pull: a story must also tell them something true and new.",
    "evidence": "Strength of the evidence on the card. 1 = rumour or a single unverified post; 3 = press reports with "
                "unnamed sources, or 'in talks'; 5 = an official announcement, but vague or with self-reported "
                "numbers only; 8 = an official primary source with specific numbers or a demo, gaps stated; 10 = "
                "primary source, specific numbers, and independent confirmation (third-party test, regulator, filing).",
    "showability": "Can we show and explain it in about 90 seconds. 1 = text only and hard to explain; 3 = a chart or "
                   "headline to show, and a concept that needs work; 5 = photos or figures, and a clear explanation; "
                   "8 = real footage or a demo, and an everyday analogy; 10 = striking footage that explains itself.",
    "learning_value": "Will the viewer learn something new and unexpected about what machines can do, or how: a new "
                      "capability, method or experiment with a result. A deployment counts by what it shows: a hypothesis tested in "
                      "the real world, or a large measured impact on something critical (say half the cost, failures or "
                      "time), not a robot simply replacing a worker or a line running a bit faster. 1 = nothing to learn (a funding round, listing, "
                      "deal, policy step or product sale, however big); 3 = a familiar trend reaches a new milestone; 5 = "
                      "a real result the field expected; 8 = a result most viewers wouldn't have guessed was possible "
                      "yet; 10 = it changes what they think machines can do.",
    "momentum": "Is the field paying attention now, from the card's signals only (X likes and engagement percentile, "
                "harvest cluster size, HN points, YouTube views of videos uploaded this week). 1 = nobody noticed; "
                "3 = a few specialists posted; 5 = the field's main accounts or outlets covered it; 8 = widely "
                "shared beyond robotics (thousands of likes, mainstream coverage); 10 = everyone is talking about it.",
}

BRIEF = """**The video.** A weekly YouTube roundup of physical AI (robots, humanoids, embodied and world models,
autonomy) for viewers with a slight technical background but little knowledge of the field. It covers {picks} stories.
Each answers: what changed in machines that act in the physical world, and why it matters.

**The goal.** A viewer who finishes the video has learned something they didn't expect about what machines can now
do, knows the week's most important changes, and stayed for it. The channel is about learning more than knowing: a
new capability, method or experiment beats a generic trend (a company moving from commercial to personal sales, one
more funding round), and at least two of the {picks} stories must be ones the viewer learns from. When goals pull
apart:
- Significance, learning value and evidence come first; viewer pull and showability decide between close stories.
- Evidence is a floor, not a trade-off: buzz never rescues a weak claim. A story whose key claim is unconfirmed can
  make the video only if its significance is high and the video can say plainly what isn't known.
- Momentum is a tie-breaker: YouTube runs on trends, but trending alone isn't a reason.
- Sensitive stories (weapons, deaths, crime, scandal) are judged like any other, on what is known and why it matters
  for physical AI; shock value counts for nothing.

**The cards.** Judge only from the cards. Each card's facts were checked against the sources it cites, and what a
source leaves out is listed under "Not stated in the source": treat a claim with a listed gap as unconfirmed. Every
opinion on a card is quoted and attributed; a single post or comment is one person's view, not a consensus, and a
company praising its own product is not evidence. Story labels are deliberately plain: judge the facts, not the
label. Card order is random and means nothing. Don't favour a card for being longer or more enthusiastic."""

CONTRACT = """Write your verdicts as JSON to {out}:
{{"judge": "{name}",
 "scores": [{{"id": "s01", "reasoning": "<2-4 sentences, written BEFORE the numbers>",
             {fields}, "overall": <1-10>}}],
 "top": ["<id>", ... your {picks} picks for the video, best first],
 "lineup": {{"picks": ["<id>", ...], "reserves": ["<id>", "<id>"],
            "why": "<2-3 sentences: why this set works together - variety of topics and companies, at least one
                    research result when a strong one exists, nothing the viewer has already seen everywhere>"}}}}
Score every card. "overall" is how strongly this story should be one of the week's {picks}, following the goal above;
it is not an average. Use the anchors, and the full range. "top" ranks stories one by one; "lineup" is the set you
would actually publish, which may differ (two near-identical stories shouldn't both run)."""


def prepare(cards_path: Path, out: Path, judges: int, picks: int, orders: int = 2) -> None:
    cards = json.loads(cards_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    # Each model judges the cards in `orders` different shuffles (A1, A2, …), each by a separate subagent: order
    # bias in list judgments shrinks when several orderings are combined (Tang et al., arXiv 2310.07712).
    for name in (f"{m}{o}" for m in string.ascii_uppercase[:judges] for o in range(1, orders + 1)):
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
            f"this week's Pixels2Actions video.\n\n{BRIEF.format(picks=picks)}\n\n## Criteria\n\n{rubric}\n\n"
            f"## Output\n\n{CONTRACT.format(out=out / f'judge_{name}.json', name=name, fields=fields, picks=picks)}"
            f"\n\n# Cards ({len(order)})\n\n{body}\n")
        print(f"judge_{name}.md: {len(order)} cards, criteria order {', '.join(crit)}")


def aggregate(out: Path, picks: int) -> list[dict]:
    cards = {c["id"]: c for c in json.loads((out / "cards.json").read_text())}
    verdicts = [json.loads(p.read_text()) for p in sorted(out.glob("judge_*.json"))]
    model = lambda v: v["judge"][0]  # noqa: E731  judge "A2" is model A's second ordering
    rows = []
    for cid, c in cards.items():
        got = [s for v in verdicts for s in v["scores"] if s["id"] == cid]
        if not got:
            continue
        per_model: dict = {}
        for v in verdicts:
            for sc in v["scores"]:
                if sc["id"] == cid:
                    per_model.setdefault(model(v), []).append(sc["overall"])
        overall = [statistics.mean(x) for x in per_model.values()]  # one number per model, so orders don't outvote
        borda = sum(picks - v["top"].index(cid) for v in verdicts if cid in v["top"][:picks])
        rows.append({"id": cid, "story": c["story"], "type": c["type"], "company": c.get("company"),
                     "overall": round(statistics.mean(overall), 2), "spread": round(max(overall) - min(overall), 2),
                     "order_spread": round(max(max(x) - min(x) for x in per_model.values()), 2),
                     "borda": borda, "votes": sum(cid in v["top"][:picks] for v in verdicts),
                     "lineup_votes": sum(cid in (v.get("lineup") or {}).get("picks", []) for v in verdicts),
                     "criteria": {k: round(statistics.mean(s[k] for s in got), 2) for k in CRITERIA
                                  if all(k in s for s in got)},  # panels judged before a criterion existed lack it
                     "by_judge": {v["judge"]: next((s["overall"] for s in v["scores"] if s["id"] == cid), None)
                                  for v in verdicts}})
    rows.sort(key=lambda r: (-r["overall"], -r["borda"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    (out / "panel.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"{len(verdicts)} judges, {len(rows)} stories")
    for v in verdicts:
        lu = v.get("lineup") or {}
        print(f"  judge {v['judge']} lineup: {', '.join(lu.get('picks', []))} (reserves {', '.join(lu.get('reserves', []))})"
              f": {lu.get('why', '')[:200]}")
    for r in rows:
        flag = "  <- judges disagree" if r["spread"] >= 3 else ""
        print(f"{r['rank']:>2}. {r['overall']:>5} (borda {r['borda']:>2}, votes {r['votes']}, in lineups {r['lineup_votes']}) {r['story']}"
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
        f"# Pairwise judge\n\nThese stories sit around the cut line for this week's Pixels2Actions video. For each "
        f"pair, decide which story more deserves a place in the video.\n\n{BRIEF.format(picks=picks)}\n\n## Criteria\n\n"
        + "\n".join(f"- **{c}**: {d}" for c, d in CRITERIA.items()) +
        f"\n\nOrder within a pair is random and means nothing; some pairs repeat in the other order: judge each on its "
        f"own.\n\nWrite JSON to {out / 'pairwise.json'}:\n"
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
    ap.add_argument("--orders", type=int, default=2, help="shuffled orderings per judge model")
    ap.add_argument("--picks", type=int, default=5)
    ap.add_argument("--width", type=int, default=2)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.paths[0], args.paths[1], args.judges, args.picks, args.orders)
    elif args.cmd == "aggregate":
        aggregate(args.paths[0], args.picks)
    elif args.cmd == "pairs":
        pairs(args.paths[0], args.picks, args.width)
    else:
        settle(args.paths[0])


if __name__ == "__main__":
    main()
