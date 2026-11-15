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
import gzip
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

UA = "Pixels2Actions-research/1.0 (quote check; github.com/SiddheshKanawade/voxtale)"  # honest; no browser impersonation
SKIP = ("sec.gov",)  # SEC refuses undeclared automated clients; its quotes stay unchecked (null)


def norm(t: str) -> str:
    t = html.unescape(t)
    t = t.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                                   "—": "-", " ": " ", "…": "..."}))
    return re.sub(r"\s+", " ", t).strip().lower()


def page_text(url: str, cache: Path) -> str | None:
    """The page's text, normalised. Raw bytes are cached so the checker can be improved without re-downloading."""
    if any(x in url for x in SKIP):
        return None
    f = cache / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".raw")
    if not f.exists():
        try:
            f.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}),
                                                 timeout=30).read())
        except Exception as ex:  # 403s, timeouts: unchecked, not failed
            print(f"  unavailable: {url} ({ex})", file=sys.stderr)
            f.write_bytes(b"")
    raw = f.read_bytes()
    if not raw:
        return None
    if raw[:2] == b"\x1f\x8b":  # some servers (deepmind.google) send gzip even when not asked
        raw = gzip.decompress(raw)
    if raw[:5] == b"%PDF-":  # PDFs (NHTSA, Federal Register, filings): extract the text layer
        if not shutil.which("pdftotext"):
            return None
        # both reading orders: table layout keeps rows together, plain mode keeps wrapped sentences together
        text = " ".join(subprocess.run(["pdftotext", *mode, str(f), "-"], capture_output=True, text=True).stdout
                        for mode in (["-layout"], []))
        return norm(text) or None
    body = raw.decode("utf-8", "replace")
    # Keep script contents: many sites ship their text as JSON inside <script> (Next.js and the like).
    body = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), body)
    body = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), body)  # Google Sites escapes
    body = body.replace('\\"', '"').replace("\\n", " ").replace("\\/", "/")
    body = html.unescape(html.unescape(body))  # Google Sites (corl.org) embeds the page as escaped HTML
    return norm(re.sub(r"<[^>]+>", " ", body))


HOW = {"loose": "ignoring punctuation and spacing", "cells": "table row: every cell found, not contiguous"}


def loose(t: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", t)  # letters, digits and CJK only


def found(quote: str, text: str) -> bool | str:
    q = norm(re.sub(r"\[translation\].*$|\[[^\]]*\]$", "", quote))  # drop an appended [translation]
    if q in text:
        return True
    parts = [x.strip(" .") for x in q.split("...") if len(x.strip(" .")) > 12]
    if len(parts) > 1 and all(x in text for x in parts):
        return True
    # Same words and numbers, different punctuation or spacing (tags, ®, line breaks, quote marks)
    lt = loose(text)
    if loose(q) and loose(q) in lt:
        return "loose"
    # A table row quoted as one line (label and numbers in separate cells): every cell must be on the page
    cells = [loose(c) for c in q.split() if len(loose(c)) >= 4]
    if len(cells) >= 3 and all(c in lt for c in cells):
        return "cells"
    return False


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
                got = None if text is None else found(q["text"], text)
                q["matched"] = None if got is None else got is not False
                q.pop("match", None)
                if got in HOW:
                    q["match"] = HOW[got]
                totals[q["matched"]] += 1
                if q["matched"] is False:
                    print(f"  NOT FOUND [{e['story']}] {q['text'][:90]!r} on {q['url']}")
        Path(path).write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print(f"quotes: {totals[True]} found, {totals[False]} not found, {totals[None]} on pages that couldn't be checked")


if __name__ == "__main__":
    main()
