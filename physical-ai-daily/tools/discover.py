#!/usr/bin/env python3
"""Collect the week's candidate stories from arXiv, X, news and Hacker News into one deduplicated JSON list.

Usage: python3 -I tools/discover.py [--days 7] [--only arxiv,x,news,hn] > <ep>/candidates-<date>.json
       python3 -I tools/discover.py --merge <ep>/candidates-*.json > <ep>/candidates.json

Sources, accounts and queries live in sources.json, plus the newsletter's six topic profiles in p2a_topics.json
(git-ignored; refresh with tools/sync_p2a_topics.py). Methods follow the prazo5 digest engine behind the Pixels2Actions
newsletter, minus its embedding scorer: relevance is judged by the model reading this file (the physical-ai-research
skill), not by cosine similarity.

| Source | Method                                                          | Key (env or .env)                        |
|--------|-----------------------------------------------------------------|------------------------------------------|
| arXiv  | export.arxiv.org API: cs.RO + keyword-filtered cs.CV/LG/AI, paged | none                                     |
| X      | twitterapi.io advanced_search: batched from: + topic keywords   | TWITTERAPI_IO_KEY (else public timeline) |
| News   | NewsData.io /latest, title match, paged (covers ~48 h only)     | NEWSDATAIO (else skipped: use WebSearch) |
| HN     | Algolia search, points above a floor                            | none                                     |

News only reaches back about two days on the free plan, so for a weekly episode either run this daily and --merge the
files, or cover older news with web search. A missing key or a failing source degrades to "no items" with a line on
stderr; it never sinks the run.

Every item gets:
- `mentions`: the other items that link to it (a tweet or HN post pointing at a paper or press release), the cheapest
  cross-source signal of what people are actually talking about.
- `topics`: newsletter topics whose keywords appear in its text, or whose account posted it. A lexical hint for
  triage, not a relevance score.
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
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_topics() -> list[dict]:
    path = ROOT / "p2a_topics.json"
    if not path.exists():
        log("topics: no p2a_topics.json (run tools/sync_p2a_topics.py); harvesting without newsletter topics")
        return []
    return json.loads(path.read_text())


# --- arXiv -----------------------------------------------------------------------------------------------------------

def arxiv(cfg: dict, since: datetime, cap: int = 3000) -> list[dict]:
    # cs.RO in full, plus papers from the broader ML categories whose abstract names an embodied topic (world models,
    # VLAs and robot learning are often filed under cs.LG or cs.CV only). Newest first, paged until past the window.
    terms = "+OR+".join(f'abs:%22{urllib.parse.quote(t)}%22' for t in cfg.get("arxiv_terms", []))
    other = "+OR+".join(f"cat:{c}" for c in cfg.get("arxiv_also", []))
    queries = ["+OR+".join(f"cat:{c}" for c in cfg["arxiv_categories"])]
    if terms and other:
        queries.append(f"%28{other}%29+AND+%28{terms}%29")
    out, seen = [], set()
    for q in queries:
        for start in range(0, cap, 500):
            feed = ET.fromstring(get(f"https://export.arxiv.org/api/query?search_query={q}&sortBy=submittedDate"
                                     f"&sortOrder=descending&start={start}&max_results=500"))
            entries = feed.findall(f"{ATOM}entry")
            for e in entries:
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
            oldest = entries[-1].findtext(f"{ATOM}published") if entries else None
            if not oldest or datetime.fromisoformat(oldest.replace("Z", "+00:00")) < since:
                break
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


def x_accounts(cfg: dict, topics: list[dict]) -> list[str]:
    seen, out = set(), []
    for h in cfg["x_accounts"] + [h for t in topics for h in t.get("x_accounts") or []]:
        if h.lower() not in seen:
            seen.add(h.lower())
            out.append(h)
    return out


def x_paid(cfg: dict, topics: list[dict], since: datetime, api_key: str) -> list[dict]:
    window = f"since_time:{int(since.timestamp())}"
    handles = x_accounts(cfg, topics)
    out = []
    for i in range(0, len(handles), 10):  # 10 handles per query, as prazo5 does: one call instead of ten
        batch = handles[i:i + 10]
        q = f"({' OR '.join(f'from:{h}' for h in batch)}) -filter:replies {window}"
        try:
            out += x_search(api_key, q, pages=cfg.get("x_pages_per_batch", 10))
        except Exception as ex:
            log(f"x: batch {batch[0]}..: FAILED {ex}")
    # Keyword search finds posts from people we don't follow. The newsletter's topic keywords are specific enough to
    # keep out the crypto "physical AI" spam a generic phrase pulls in; the likes floor drops the long tail.
    per_topic = cfg.get("x_keywords_per_topic", 4)
    kws = cfg.get("x_keywords", []) + [f'"{k}"' for t in topics for k in (t.get("keywords") or [])[:per_topic]]
    for kw in dict.fromkeys(kws):
        q = f"{kw} lang:en -filter:retweets -filter:replies min_faves:{cfg.get('x_min_faves', 50)} {window}"
        try:
            out += x_search(api_key, q, pages=1)
        except Exception as ex:
            log(f"x: keyword {kw!r}: FAILED {ex}")
    return out


def x_public(cfg: dict, topics: list[dict], since: datetime) -> list[dict]:
    # Fallback without a key: the embed widget's profile timeline. Rate-limited (~30 per window) and sometimes stale,
    # so only our own short account list, not the newsletter's.
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
    # Title-only matching (qInTitle): a plain `q` matches anywhere in the body and returns stock-ticker pages. No
    # `timeframe` param: the free plan rejects it (403 AccessDenied), so filter by pubDate here instead.
    out = []
    for q in cfg["news_queries"]:
        page = ""
        for _ in range(cfg.get("news_pages", 3)):
            payload: dict = {}
            params = {"qInTitle": q, "language": "en", **({"page": page} if page else {})}
            while pool:
                try:
                    payload = json.loads(get("https://newsdata.io/api/1/latest?"
                                             + urllib.parse.urlencode({**params, "apikey": pool[0]})))
                    break
                except urllib.error.HTTPError as ex:
                    if ex.code == 429:  # this key's quota is spent: rotate to the next pooled key
                        log(f"news: key {len(pool)} rate-limited, rotating")
                        pool.pop(0)
                    else:
                        log(f"news: {q!r}: HTTP {ex.code} {ex.read()[:160]!r}")
                        break
            if not pool:
                log("news: every key is rate-limited")
                return out
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
            page = payload.get("nextPage")
            if not page:
                break
    return out


# --- Hacker News (Algolia) -------------------------------------------------------------------------------------------

def hn(cfg: dict, since: datetime) -> list[dict]:
    out = []
    for q in cfg.get("hn_queries", []):
        params = {"query": q, "tags": "story", "hitsPerPage": 100,
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


def tag(items: list[dict], topics: list[dict]) -> None:
    owners: dict[str, list[str]] = {}
    for t in topics:
        for h in t.get("x_accounts") or []:
            owners.setdefault(h.lower(), []).append(t["name"])
    for it in items:
        text = f"{it.get('title') or ''} {it.get('text') or ''}".lower()
        hits = [t["name"] for t in topics if any(k.lower() in text for k in t.get("keywords") or [])]
        it["topics"] = sorted(set(hits + owners.get((it.get("handle") or "").lower(), [])))


def merge(items: list[dict], topics: list[dict]) -> list[dict]:
    by_url: dict[str, dict] = {}
    for it in items:
        # First wins: the same post can match a handle batch and a keyword. An HN post is keyed by its discussion so it
        # isn't merged away into the article it links to; it counts as a mention of that article instead.
        by_url.setdefault(canon(it["discussion"] if it["type"] == "hn" else it["url"]), it)
    for it in by_url.values():
        it["mentions"] = []
    for it in by_url.values():
        targets = [canon(link) for link in it.get("links", []) + ([it["url"]] if it["type"] == "hn" else [])]
        # Papers are often linked through mirrors (alphaxiv, Hugging Face Papers, yesnoerror): match the arXiv id itself.
        blob = " ".join(it.get("links", []) + [it.get("text") or "", it["url"] if it["type"] == "hn" else ""])
        targets += [f"arxiv.org/abs/{pid}" for pid in re.findall(r"(?<![\d.])(\d{4}\.\d{4,5})(?!\d)", blob)]
        for t in dict.fromkeys(targets):
            target = by_url.get(t)
            if target is not None and target is not it:
                target["mentions"].append(it["url"] if it["type"] != "hn" else it["discussion"])
    tag(list(by_url.values()), topics)
    return sorted(by_url.values(), key=lambda i: i["published"], reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7, help="window for every source (one episode per week)")
    ap.add_argument("--only", default="arxiv,x,news,hn")
    ap.add_argument("--merge", nargs="+", metavar="JSON", help="merge earlier harvests instead of fetching")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "sources.json").read_text())
    topics = load_topics()

    if args.merge:
        items = [it for f in args.merge for it in json.loads(Path(f).read_text())]
    else:
        since = datetime.now(timezone.utc) - timedelta(days=args.days)
        only = set(args.only.split(","))
        items = []
        runs = {
            "arxiv": lambda: arxiv(cfg, since),
            "x": lambda: x_paid(cfg, topics, since, k) if (k := key("TWITTERAPI_IO_KEY")) else (log("x: no "
                         "TWITTERAPI_IO_KEY, falling back to the public timeline (unreliable)") or x_public(cfg, topics, since)),
            "news": lambda: news(cfg, since, pool) if (pool := keys("NEWSDATAIO")) else (log("news: no NEWSDATAIO "
                            "key, skipped; cover news with web search") or []),
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
    merged = merge(items, topics)
    log(f"total: {len(merged)} after dedup, {sum(1 for i in merged if i['mentions'])} with cross-source mentions, "
        f"{sum(1 for i in merged if i['topics'])} tagged with a newsletter topic")
    json.dump(merged, sys.stdout, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
