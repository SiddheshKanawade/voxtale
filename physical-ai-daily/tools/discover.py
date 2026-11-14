#!/usr/bin/env python3
"""Collect today's candidate stories from arXiv and curated X accounts (no logins needed).

Usage: python3 -I tools/discover.py [--hours 36] > candidates.json

Sources and accounts live in sources.json. News sites are not scraped here; search for them with
WebSearch and verify each one against its primary source (see the skill's verification rules).
"""
import argparse
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0"}  # X rate-limits by window (30 req); keep the account list well under that
ATOM = "{http://www.w3.org/2005/Atom}"


def get(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read().decode("utf-8", "ignore")


def arxiv(categories: list[str], since: datetime, max_results: int = 200) -> list[dict]:
    q = "+OR+".join(f"cat:{c}" for c in categories)
    feed = ET.fromstring(get(f"https://export.arxiv.org/api/query?search_query={q}"
                             f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"))
    out = []
    for e in feed.findall(f"{ATOM}entry"):
        published = datetime.fromisoformat(e.findtext(f"{ATOM}published").replace("Z", "+00:00"))
        if published < since:
            continue
        out.append({
            "type": "paper",
            "id": e.findtext(f"{ATOM}id").rsplit("/", 1)[-1],
            "url": e.findtext(f"{ATOM}id"),
            "title": " ".join(e.findtext(f"{ATOM}title").split()),
            "summary": " ".join(e.findtext(f"{ATOM}summary").split()),
            "published": published.isoformat(),
            "categories": [c.get("term") for c in e.findall(f"{ATOM}category")],
        })
    return out


def x_timeline(handle: str, since: datetime) -> list[dict]:
    # Public profile timeline used by X's embed widget; returns ~20 recent posts without login.
    html = get(f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{handle}")
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return []
    out = []
    for e in json.loads(m.group(1))["props"]["pageProps"]["timeline"]["entries"]:
        t = e.get("content", {}).get("tweet")
        if not t:
            continue
        created = parsedate_to_datetime(t["created_at"])
        if created < since or t["text"].startswith("RT @"):
            continue
        out.append({
            "type": "x",
            "id": t["id_str"],
            "url": f"https://x.com/{handle}/status/{t['id_str']}",
            "handle": handle,
            "text": t["text"],
            "published": created.isoformat(),
            "likes": t.get("favorite_count", 0),
            "has_media": bool(t.get("mediaDetails")),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=36)
    args = ap.parse_args()
    cfg = json.loads((Path(__file__).resolve().parent.parent / "sources.json").read_text())
    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)

    items = arxiv(cfg["arxiv_categories"], since)
    print(f"arxiv: {len(items)} papers", file=sys.stderr)
    for h in cfg["x_accounts"]:
        try:
            posts = x_timeline(h, since)
            items += posts
            print(f"x/{h}: {len(posts)} posts", file=sys.stderr)
        except Exception as ex:  # one bad handle shouldn't sink the run
            print(f"x/{h}: FAILED {ex}", file=sys.stderr)
        time.sleep(1.5)  # be gentle with the public endpoint
    json.dump(items, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
