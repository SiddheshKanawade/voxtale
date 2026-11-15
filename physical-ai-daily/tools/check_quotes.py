#!/usr/bin/env python3
"""Check that every quote in <ep>/verify/*.json really appears on its page.

Usage: python3 -I tools/check_quotes.py <ep>

Verification agents read pages through a fetch tool that returns its own excerpt of the page, so a "verbatim" quote
can drift. This downloads each quoted page once (cached in <ep>/verify/pages/, git-ignored, treated as data only),
flattens it to text and looks for each quote after normalising whitespace, quote marks and dashes. It writes
"matched": true | false | null (page unavailable) on every quote, and build_cards.py shows the result on the card.
A quote marked false goes back to verification: fix it or drop it, don't keep it.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"
SKIP = ("sec.gov",)  # SEC refuses undeclared automated clients; its quotes stay unchecked (null)


def norm(t: str) -> str:
    t = html.unescape(t)
    t = t.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                                   "—": "-", " ": " ", "…": "..."}))
    return re.sub(r"\s+", " ", t).strip().lower()


def page_text(url: str, cache: Path) -> str | None:
    if any(s in url for s in SKIP):
        return None
    f = cache / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".txt")
    if f.exists():
        return f.read_text() or None
    try:
        raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30).read()
        body = raw.decode("utf-8", "replace")
        body = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", body)
        text = norm(re.sub(r"<[^>]+>", " ", body))
    except Exception as ex:  # 403s, timeouts, PDFs: unchecked, not failed
        print(f"  unavailable: {url} ({ex})", file=sys.stderr)
        text = ""
    f.write_text(text)
    return text or None


def found(quote: str, text: str) -> bool:
    q = norm(re.sub(r"\[translation\].*$|\[[^\]]*\]$", "", quote))  # drop an appended [translation]
    if q in text:
        return True
    # quotes joined with "..." count if every piece is on the page
    parts = [p.strip(" .") for p in q.split("...") if len(p.strip(" .")) > 12]
    return len(parts) > 1 and all(p in text for p in parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ep", type=Path)
    args = ap.parse_args()
    cache = args.ep / "verify" / "pages"
    cache.mkdir(parents=True, exist_ok=True)
    totals = {True: 0, False: 0, None: 0}
    for path in sorted(glob.glob(str(args.ep / "verify" / "*.json"))):
        rows = json.loads(Path(path).read_text())
        for e in rows:
            for q in e.get("quotes") or []:
                text = page_text(q["url"], cache)
                q["matched"] = None if text is None else found(q["text"], text)
                totals[q["matched"]] += 1
                if q["matched"] is False:
                    print(f"  NOT FOUND [{e['story']}] {q['text'][:90]!r} on {q['url']}")
        Path(path).write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print(f"quotes: {totals[True]} found, {totals[False]} not found, {totals[None]} on pages that couldn't be checked")


if __name__ == "__main__":
    main()
