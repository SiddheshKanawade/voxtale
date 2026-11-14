#!/usr/bin/env python3
"""Collect today's candidate stories from arXiv, X, news and Hacker News into one deduplicated JSON list.

Usage: python3 -I tools/discover.py [--hours 36] [--paper-hours 72] [--only arxiv,x,news,hn] > <ep>/candidates.json

Sources, accounts and queries live in sources.json. Methods follow the prazo5 digest engine (the user's Pixels2Actions
newsletter backend), minus its embedding scorer: relevance is judged by the model reading this file (see the skill's
Research section), not by cosine similarity.

| Source | Method                                                     | Key (env or .env)                       |
|--------|------------------------------------------------------------|-----------------------------------------|
| arXiv  | export.arxiv.org API: cs.RO + keyword-filtered cs.CV/LG/AI | none                                    |
| X      | twitterapi.io advanced_search: batched from: + keywords    | TWITTERAPI_IO_KEY (else public timeline) |
| News   | NewsData.io /latest, one call per query                    | NEWSDATAIO (else skipped: use WebSearch) |
| HN     | Algolia search, points above a floor                       | none                                    |

A missing key or a failing source degrades to "no items" with a line on stderr; it never sinks the run. Every item
gets `mentions`: how many other items link to it (a tweet or HN post pointing at a paper or press release), which is
the cheapest cross-source signal of what people are actually talking about.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "Mozilla/5.0"}
ATOM = "{http://www.w3.org/2005/Atom}"


def key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip() or None
    return None


def keys(name: str) -> list[str]:
    # prazo5 convention: one key plain, or several as a JSON array to pool free-tier quota (rotate on 429).
    raw = key(name)
    if not raw:
        return []
    return json.loads(raw) if raw.startswith("[") else [raw]


def get(url: str, headers: dict | None = None) -> str:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8", "ignore")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# --- arXiv -----------------------------------------------------------------------------------------------------------

def arxiv(cfg: dict, since: datetime) -> list[dict]:
    # cs.RO in full, plus papers from the broader ML categories whose abstract names an embodied topic (world models,
    # VLAs and robot learning are often filed under cs.LG or cs.CV only).
    terms = "+OR+".join(f'abs:%22{urllib.parse.quote(t)}%22' for t in cfg.get("arxiv_terms", []))
    other = "+OR+".join(f"cat:{c}" for c in cfg.get("arxiv_also", []))
    queries = ["+OR+".join(f"cat:{c}" for c in cfg["arxiv_categories"])]
    if terms and other:
        queries.append(f"%28{other}%29+AND+%28{terms}%29")
    out, seen = [], set()
    for q in queries:
        feed = ET.fromstring(get(f"https://export.arxiv.org/api/query?search_query={q}"
                                 f"&sortBy=submittedDate&sortOrder=descending&max_results=300"))
        for e in feed.findall(f"{ATOM}entry"):
            published = datetime.fromisoformat(e.findtext(f"{ATOM}published").replace("Z", "+00:00"))
            pid = e.findtext(f"{ATOM}id").rsplit("/", 1)[-1].split("v")[0]
            if published < since or pid in seen:
                continue
            seen.add(pid)
            out.append({
                "type": "paper",
                "id": pid,
                "url": f"https://arxiv.org/abs/{pid}",
                "title": " ".join(e.findtext(f"{ATOM}title").split()),
                "text": " ".join(e.findtext(f"{ATOM}summary").split()),
                "authors": [a.findtext(f"{ATOM}name") for a in e.findall(f"{ATOM}author")][:8],
                "published": published.isoformat(),
                "categories": [c.get("term") for c in e.findall(f"{ATOM}category")],
            })
        time.sleep(3)  # arXiv asks for one request every 3 s
    return out


# --- X ---------------------------------------------------------------------------------------------------------------

def _tweet(t: dict) -> dict | None:
    author = t.get("author") or {}
    handle, text = author.get("userName"), (t.get("text") or "").strip()
    if not handle or not text or text.startswith("RT @"):
        return None
    links = [u.get("expanded_url") for u in (t.get("entities") or {}).get("urls", []) if u.get("expanded_url")]
    return {
        "type": "x",
        "id": str(t["id"]),
        "url": f"https://x.com/{handle}/status/{t['id']}",
        "handle": handle,
        "name": author.get("name"),
        "followers": author.get("followers"),
        "text": text,
        "links": links,
        "published": parsedate_to_datetime(t["createdAt"]).isoformat(),
        "likes": t.get("likeCount"), "retweets": t.get("retweetCount"), "quotes": t.get("quoteCount"),
        "replies": t.get("replyCount"), "views": t.get("viewCount"),
        "has_media": bool((t.get("extendedEntities") or {}).get("media")),
    }


def x_search(api_key: str, query: str, pages: int) -> list[dict]:
    out, cursor = [], ""
    for _ in range(pages):
        params = {"query": query, "queryType": "Latest", **({"cursor": cursor} if cursor else {})}
        payload = json.loads(get("https://api.twitterapi.io/twitter/tweet/advanced_search?"
                                 + urllib.parse.urlencode(params), {"X-API-Key": api_key}))
        out += [p for p in map(_tweet, payload.get("tweets") or []) if p]
        cursor = payload.get("next_cursor")
        if not payload.get("has_next_page") or not cursor:
            break
    return out


def x_paid(cfg: dict, since: datetime, api_key: str) -> list[dict]:
    window = f"since_time:{int(since.timestamp())}"
    handles = cfg["x_accounts"]
    out = []
    for i in range(0, len(handles), 10):  # 10 handles per query, as prazo5 does: one call instead of ten
        batch = handles[i:i + 10]
        q = f"({' OR '.join(f'from:{h}' for h in batch)}) -filter:replies {window}"
        try:
            out += x_search(api_key, q, pages=5)
        except Exception as ex:
            log(f"x: batch {batch[0]}..: FAILED {ex}")
    for kw in cfg.get("x_keywords", []):
        # Keyword search finds posts from people we don't follow; the engagement floor keeps out the long tail.
        q = f"{kw} lang:en -filter:retweets -filter:replies min_faves:{cfg.get('x_min_faves', 50)} {window}"
        try:
            out += x_search(api_key, q, pages=1)
        except Exception as ex:
            log(f"x: keyword {kw!r}: FAILED {ex}")
    return out


def x_public(cfg: dict, since: datetime) -> list[dict]:
    # Fallback without a key: the embed widget's profile timeline. Rate-limited (~30 per window) and sometimes stale.
    out = []
    for h in cfg["x_accounts"]:
        try:
            html = get(f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{h}")
            m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
            for e in json.loads(m.group(1))["props"]["pageProps"]["timeline"]["entries"] if m else []:
                t = e.get("content", {}).get("tweet")
                if not t or parsedate_to_datetime(t["created_at"]) < since or t["text"].startswith("RT @"):
                    continue
                out.append({"type": "x", "id": t["id_str"], "url": f"https://x.com/{h}/status/{t['id_str']}",
                            "handle": h, "text": t["text"], "links": [], "published": parsedate_to_datetime(t["created_at"]).isoformat(),
                            "likes": t.get("favorite_count", 0), "has_media": bool(t.get("mediaDetails"))})
        except Exception as ex:
            log(f"x/{h}: FAILED {ex}")
        time.sleep(1.5)
    return out


# --- News (NewsData.io) ----------------------------------------------------------------------------------------------

def news(cfg: dict, since: datetime, pool: list[str]) -> list[dict]:
    out = []
    for q in cfg["news_queries"]:
        params = {"q": q, "language": "en", "timeframe": "48"}
        while pool:
            try:
                payload = json.loads(get("https://newsdata.io/api/1/latest?"
                                         + urllib.parse.urlencode({**params, "apikey": pool[0]})))
                break
            except urllib.error.HTTPError as ex:
                if ex.code == 422 and "timeframe" in params:  # paid-only on some plans; filter by date below instead
                    params.pop("timeframe")
                elif ex.code == 429:  # this key's quota is spent: rotate to the next pooled key
                    log(f"news: key {len(pool)} rate-limited, rotating")
                    pool.pop(0)
                else:
                    log(f"news: {q!r}: HTTP {ex.code}")
                    payload = {}
                    break
        else:
            log("news: every key is rate-limited")
            break
        for r in payload.get("results") or []:
            try:
                published = datetime.fromisoformat(r["pubDate"].replace(" ", "T")).replace(tzinfo=timezone.utc)
            except (KeyError, TypeError, ValueError):
                continue
            if published < since or not r.get("link"):
                continue
            text = r.get("description") or ""
            out.append({"type": "news", "id": r.get("article_id"), "url": r["link"], "title": r.get("title"),
                        "text": text if "ONLY AVAILABLE IN PAID" not in text else "", "outlet": r.get("source_id"),
                        "published": published.isoformat(), "query": q})
    return out


# --- Hacker News (Algolia) -------------------------------------------------------------------------------------------

def hn(cfg: dict, since: datetime) -> list[dict]:
    out = []
    for q in cfg.get("hn_queries", []):
        params = {"query": q, "tags": "story", "hitsPerPage": 50,
                  "numericFilters": f"created_at_i>{int(since.timestamp())},points>{cfg.get('hn_min_points', 10)}"}
        payload = json.loads(get("https://hn.algolia.com/api/v1/search_by_date?" + urllib.parse.urlencode(params)))
        for h in payload.get("hits") or []:
            discussion = f"https://news.ycombinator.com/item?id={h['objectID']}"
            out.append({"type": "hn", "id": h["objectID"], "url": h.get("url") or discussion, "discussion": discussion,
                        "title": h.get("title"), "points": h.get("points"), "comments": h.get("num_comments"),
                        "published": datetime.fromtimestamp(h["created_at_i"], timezone.utc).isoformat()})
    return out


# --- merge -----------------------------------------------------------------------------------------------------------

def canon(url: str) -> str:
    u = urllib.parse.urlsplit(url.strip())
    host = u.netloc.lower().removeprefix("www.").replace("twitter.com", "x.com")
    path = re.sub(r"(arxiv\.org/)(pdf|html)/", r"\1abs/", host + u.path).rstrip("/")
    return re.sub(r"(arxiv\.org/abs/\d{4}\.\d{4,5})(v\d+)?(\.pdf)?$", r"\1", path)


def merge(items: list[dict]) -> list[dict]:
    by_url: dict[str, dict] = {}
    for it in items:
        # First wins: the same post can match a handle batch and a keyword. An HN post is keyed by its discussion so it
        # isn't merged away into the article it links to; it counts as a mention of that article instead.
        by_url.setdefault(canon(it["discussion"] if it["type"] == "hn" else it["url"]), it)
    for it in by_url.values():
        it["mentions"] = []
    for it in by_url.values():
        for link in it.get("links", []) + ([it["url"]] if it["type"] == "hn" else []):
            target = by_url.get(canon(link))
            if target is not None and target is not it:
                target["mentions"].append(it["url"] if it["type"] != "hn" else it["discussion"])
    return sorted(by_url.values(), key=lambda i: i["published"], reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=36, help="window for X, news and HN")
    ap.add_argument("--paper-hours", type=int, default=72, help="window for arXiv (announcements lag and skip weekends)")
    ap.add_argument("--only", default="arxiv,x,news,hn")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "sources.json").read_text())
    now = datetime.now(timezone.utc)
    since, paper_since = now - timedelta(hours=args.hours), now - timedelta(hours=args.paper_hours)
    only = set(args.only.split(","))

    items: list[dict] = []
    runs = {
        "arxiv": lambda: arxiv(cfg, paper_since),
        "x": lambda: x_paid(cfg, since, k) if (k := key("TWITTERAPI_IO_KEY")) else (log("x: no TWITTERAPI_IO_KEY, "
                     "falling back to the public timeline (unreliable)") or x_public(cfg, since)),
        "news": lambda: news(cfg, since, pool) if (pool := keys("NEWSDATAIO")) else (log("news: no NEWSDATAIO key, "
                        "skipped; cover news with WebSearch") or []),
        "hn": lambda: hn(cfg, since),
    }
    for name, run in runs.items():
        if name not in only:
            continue
        try:
            got = run()
            log(f"{name}: {len(got)} items")
            items += got
        except Exception as ex:  # one source failing shouldn't sink the run
            log(f"{name}: FAILED {ex}")
    merged = merge(items)
    log(f"total: {len(merged)} after dedup, {sum(1 for i in merged if i['mentions'])} with cross-source mentions")
    json.dump(merged, sys.stdout, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
