#!/usr/bin/env python3
"""Collect the last week's candidate stories from arXiv, X and news into one deduplicated, signal-annotated JSON list.

Usage: python3 -I tools/discover.py [--days 7] [--only arxiv,x,news] > <ep>/candidates-<date>.json
       python3 -I tools/discover.py --merge <ep>/candidates-*.json > <ep>/candidates.json

Sources, accounts and queries live in sources.json, plus the P2A digest's six topic profiles in p2a_topics.json
(git-ignored). Every harvest first re-syncs those profiles from prazo5 (tools/sync_p2a_topics.py); if the sync fails it
falls back to the last copy with a warning, and stops if there is none. Methods follow prazo5, the digest engine behind
Pixels2Actions, minus its embedding scorer: relevance is judged by the model reading this file (the
physical-ai-research skill), not by cosine similarity.

| Source | Method                                                                    | Key (env or .env)               |
|--------|---------------------------------------------------------------------------|---------------------------------|
| arXiv  | export.arxiv.org API: cs.RO in full + cs.CV/LG/AI/eess.SY filtered by our terms and every digest keyword, paged | none |
| X      | twitterapi.io advanced_search: batched from: + topic keywords             | TWITTERAPI_IO_KEY (else public) |
| News   | NewsData.io /latest, title match, paged (covers ~36 h only)               | NEWSDATAIO (else skipped)       |
| HN     | Algolia, points floor. Off by default (mostly off-topic); tools/chatter.py searches HN per story instead | none |

The sources run in parallel. News only reaches back about 36 h on the free plan, so for a weekly episode either run
`--only news` daily and --merge the files, or cover older news with web search. A missing key or a failing source
degrades to "no items" with a line on stderr; it never sinks the run.

Every item gets:
- `mentions`: the other items that link to it (a tweet pointing at a paper or press release), the cheapest
  cross-source signal of what people are actually talking about.
- `topics`: digest topics whose keywords appear in its text, or whose account posted it. A lexical hint, not a score.

Papers also get `signals` (alphaXiv organisations, votes and visits; Hugging Face Daily Papers upvotes; code and
stars; demo video; X attention) and a `prior` from 0 to 1 (see paper_priors). X posts get an `engagement` score and
its percentile `eng_pct` (see engagement). Priors are a sort key for reading order, not the editorial ranking.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
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
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


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


def load_topics(sync: bool) -> list[dict]:
    # The digest's topic profiles are required: they steer the X accounts, the arXiv terms and triage. Re-sync them on
    # every harvest; on failure, use the last copy if there is one.
    path = ROOT / "p2a_topics.json"
    if sync:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script's own directory
            import sync_p2a_topics
            sync_p2a_topics.main()
        except BaseException as ex:  # sync exits via sys.exit on a missing key or psql error
            if not path.exists():
                sys.exit(f"topics: sync failed ({ex}) and there is no p2a_topics.json; fix RENDER_API_KEY or psql first")
            age = (time.time() - path.stat().st_mtime) / 86400
            log(f"topics: sync FAILED ({ex}); using the copy from {age:.0f} days ago")
    if not path.exists():
        sys.exit("topics: no p2a_topics.json; run tools/sync_p2a_topics.py")
    return json.loads(path.read_text())


# --- arXiv -----------------------------------------------------------------------------------------------------------

def arxiv(cfg: dict, topics: list[dict], since: datetime, cap: int = 3000) -> list[dict]:
    # Exhaustive on purpose: cs.RO in full, plus papers from the broader ML categories whose title or abstract names
    # an embodied topic (world models, VLAs and robot learning are often filed under cs.LG or cs.CV only). The terms
    # are ours plus every digest keyword, in chunks so each query URL stays short. Newest first, paged past the window.
    terms = list(dict.fromkeys(t.lower() for t in cfg.get("arxiv_terms", []) + [k for tp in topics
                                                                              for k in tp.get("keywords") or []]))
    other = "+OR+".join(f"cat:{c}" for c in cfg.get("arxiv_also", []))
    queries = ["+OR+".join(f"cat:{c}" for c in cfg["arxiv_categories"])]
    for i in range(0, len(terms) if other else 0, 12):
        chunk = "+OR+".join(f'abs:%22{urllib.parse.quote(t)}%22+OR+ti:%22{urllib.parse.quote(t)}%22'
                            for t in terms[i:i + 12])
        queries.append(f"%28{other}%29+AND+%28{chunk}%29")
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
                    "authors": [a.findtext(f"{ATOM}name") for a in e.findall(f"{ATOM}author")],
                    "published": published.isoformat(),
                    "categories": [c.get("term") for c in e.findall(f"{ATOM}category")],
                    # Authors' comment: venue ("Accepted to CoRL 2026"), page count, project page or code link.
                    "comment": " ".join((e.findtext(f"{ARXIV_NS}comment") or "").split()),
                })
            time.sleep(3)  # arXiv asks for one request every 3 s
            oldest = entries[-1].findtext(f"{ATOM}published") if entries else None
            if not oldest or datetime.fromisoformat(oldest.replace("Z", "+00:00")) < since:
                break
    return out


# --- Paper signals ---------------------------------------------------------------------------------------------------
# A paper 0-7 days old has no citations yet, so rank by the early signals the literature finds predictive (see the
# research skill's "Ranking" section for sources): attention from people who read it (alphaXiv votes and visits,
# Hugging Face Daily Papers upvotes, X posts by accounts with reach), released code, and, capped, who wrote it.
# Semantic Scholar is left out: its author profiles for brand-new papers are fragmented (h-index far too low) and its
# affiliations are empty.

def _alphaxiv(pid: str) -> dict:
    try:
        d = json.loads(get(f"https://api.alphaxiv.org/v2/papers/{pid}/metadata"))["data"]
    except Exception:
        return {}
    group = d.get("paper_group") or {}
    m = group.get("metrics") or {}
    researchers = [a.get("researcher") or {} for a in d.get("authors_v2") or []]
    return {
        "orgs": [o["name"] for o in d.get("organization_info") or [] if o.get("name")],
        "ax_votes": m.get("public_total_votes") or 0,
        "ax_visits_7d": (m.get("visits_count") or {}).get("last7Days") or 0,
        "max_h": max([r.get("hIndex") or 0 for r in researchers] or [0]),
        "affiliations": [r["affiliation"] for r in researchers if r.get("affiliation")][:6],
    }


def _hf_daily(since: datetime) -> dict[str, dict]:
    # Hugging Face's curated Daily Papers, by ISO week (paged 100 at a time). Covers only a few % of arXiv, so absence
    # means little; presence and upvotes mean a lot.
    out, day = {}, since
    weeks = []
    while day <= datetime.now(timezone.utc) + timedelta(days=7):
        y, w, _ = day.isocalendar()
        weeks.append(f"{y}-W{w:02d}")
        day += timedelta(days=7)
    for week in dict.fromkeys(weeks):
        for page in range(10):
            try:
                rows = json.loads(get(f"https://huggingface.co/api/daily_papers?week={week}&limit=100&p={page}"))
            except Exception as ex:
                log(f"papers: Hugging Face {week}: FAILED {ex}")
                break
            for r in rows:
                p = r.get("paper") or {}
                out[p.get("id")] = {"hf_upvotes": p.get("upvotes") or 0, "github": p.get("githubRepo"),
                                    "stars": p.get("githubStars"), "video": any(str(u).endswith((".mp4", ".webm"))
                                                                                for u in r.get("mediaUrls") or []),
                                    "project_page": p.get("projectPage")}
            if len(rows) < 100:
                break
    return out


def _stars(repo: str) -> int | None:
    m = re.match(r"https?://github\.com/([\w.-]+/[\w.-]+)", repo or "")
    if not m:
        return None
    try:  # unauthenticated: 60 requests an hour, so only for papers without HF's star count
        return json.loads(get(f"https://api.github.com/repos/{m.group(1).removesuffix('.git')}"))["stargazers_count"]
    except Exception:
        return None


def paper_signals(papers: list[dict]) -> None:
    from concurrent.futures import ThreadPoolExecutor
    hf = _hf_daily(min((datetime.fromisoformat(p["published"]) for p in papers), default=datetime.now(timezone.utc)))
    with ThreadPoolExecutor(4) as pool:
        ax = list(pool.map(lambda p: _alphaxiv(p["id"]), papers))
    lookups = 0
    for p, a in zip(papers, ax):
        h = hf.get(p["id"], {})
        blob = f"{p.get('text', '')} {p.get('comment', '')}"
        repo = h.get("github") or next(iter(re.findall(r"https?://github\.com/[\w.-]+/[\w.-]+", blob)), None)
        stars = h.get("stars")
        if repo and stars is None and lookups < 50:
            lookups += 1
            stars = _stars(repo)
        p["signals"] = {
            **a, "hf_upvotes": h.get("hf_upvotes", 0), "github": repo, "stars": stars,
            "video": bool(h.get("video")) or bool(re.search(r"\bvideos?\b|project page|github\.io", blob, re.I)),
            "project_page": h.get("project_page") or next(iter(re.findall(r"https?://[\w.-]+\.github\.io[\w./-]*", blob)), None),
            "venue": next(iter(re.findall(r"(?:accepted|to appear|published) (?:at|to|in|by) ([^.;,]+)", p.get("comment", ""),
                                          re.I)), None),
        }
    log(f"papers: signals for {len(papers)} ({sum(1 for a in ax if a)} on alphaXiv, "
        f"{sum(1 for p in papers if p['signals']['hf_upvotes'])} on HF Daily Papers, "
        f"{sum(1 for p in papers if p['signals']['github'])} with code)")


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
    # Keyword search finds posts from people we don't follow. The digest's topic keywords are specific enough to
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
    # so only our own short account list, not the digest's.
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


def _pct(values: list[float]):
    # Percentile within this week's pool: share of the pool strictly below v. Zeros stay 0; heavy tails don't dominate.
    ordered = sorted(values)
    n = max(1, len(ordered) - 1)
    return lambda v: bisect.bisect_left(ordered, v) / n


def _lognorm(values: list[float]):
    # log(1+x) / log(1+max) within this week's pool. For sparse signals (most papers have 0 upvotes and no posts) a
    # percentile would hand a single upvote the 95th percentile; this keeps 1 upvote near 0.15 when the week's best has 100.
    top = math.log1p(max(values, default=0))
    return lambda v: math.log1p(v) / top if top else 0.0


def engagement(t: dict) -> float:
    # Absolute reach plus over-performance for the account's size (ln(engagement/followers) is the best virality proxy
    # in the literature): a 2k-like post from a 5k-follower researcher beats a 2k-like post from a 30M brand account.
    e = (t.get("likes") or 0) + 2 * (t.get("retweets") or 0) + 3 * (t.get("quotes") or 0) + 0.5 * (t.get("replies") or 0)
    return 0.6 * math.log10(1 + e) + 0.4 * math.log((e + 1) / ((t.get("followers") or 0) + 100))


def org_tier(names: list[str], cfg: dict) -> float:
    blob = " | ".join(names)
    hit = lambda orgs: any(re.search(rf"(?<!\w){re.escape(o)}(?!\w)", blob, re.I) for o in orgs)
    return 1.0 if hit(cfg.get("orgs_lab", [])) else 0.5 if hit(cfg.get("orgs_university", [])) else 0.0


def paper_priors(papers: list[dict], tweets: dict[str, dict], cfg: dict) -> None:
    """prior = 0.30 N(HF upvotes) + 0.20 N(X attention) + 0.10 N(alphaXiv) + 0.15 code + 0.10 demo + 0.15 prestige.

    N() is log(1+x) / log(1+max) over this week's papers (see _lognorm). HF upvotes and alphaXiv (votes + visits/20) are divided by
    sqrt(days old + 1) so Friday's papers aren't buried. X attention sums, over distinct accounts that posted the
    paper, min(log10(1+followers), 7) + log10(1+likes): who shares a paper matters more than how many. code = 0.5 for a
    repo + 0.5 min(1, log10(1+stars)/3). demo = 1 for a video or project page (this is a video channel). prestige = 1
    for a lab in sources.json orgs_lab, 0.5 for orgs_university; it counts a third as much when nobody is paying
    attention (attention + code < 0.1), because big-lab papers get hype regardless and reviewers already over-reward
    prestige.
    """
    now = datetime.now(timezone.utc)
    rows = []
    for p in papers:
        s = p.setdefault("signals", {})
        age = max(0.0, (now - datetime.fromisoformat(p["published"])).total_seconds() / 86400)
        posts = {}
        for url in p.get("mentions", []):
            t = tweets.get(url)
            if t:
                posts.setdefault(t["handle"].lower(), []).append(t)
        x_att = sum(min(math.log10(1 + (ts[0].get("followers") or 0)), 7) + math.log10(1 + max(t.get("likes") or 0 for t in ts))
                    for ts in posts.values())
        rows.append((p, s, (s.get("hf_upvotes") or 0) / math.sqrt(age + 1),
                     ((s.get("ax_votes") or 0) + (s.get("ax_visits_7d") or 0) / 20) / math.sqrt(age + 1), x_att,
                     len(posts)))
    p_hf, p_ax, p_x = (_lognorm([r[i] for r in rows]) for i in (2, 3, 4))
    for p, s, hf, ax, x_att, n_posts in rows:
        attention = 0.30 * p_hf(hf) + 0.20 * p_x(x_att) + 0.10 * p_ax(ax)
        code = 0.15 * ((0.5 if s.get("github") else 0) + 0.5 * min(1, math.log10(1 + (s.get("stars") or 0)) / 3))
        demo = 0.10 * bool(s.get("video") or s.get("project_page"))
        tier = org_tier((s.get("orgs") or []) + (s.get("affiliations") or []), cfg)
        prestige = (0.15 if attention + code >= 0.1 else 0.05) * tier
        p["signals"]["x_accounts"] = n_posts
        p["signals"]["org_tier"] = tier
        p["prior"] = round(attention + code + demo + prestige, 3)
        p["prior_parts"] = {"attention": round(attention, 3), "code": round(code, 3), "demo": round(demo, 3),
                            "prestige": round(prestige, 3)}


def merge(items: list[dict], topics: list[dict], cfg: dict) -> list[dict]:
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
    tweets = {it["url"]: it for it in by_url.values() if it["type"] == "x"}
    for t in tweets.values():
        t["engagement"] = round(engagement(t), 3)
    p_eng = _pct([t["engagement"] for t in tweets.values()])
    for t in tweets.values():
        t["eng_pct"] = round(p_eng(t["engagement"]), 3)
    paper_priors([it for it in by_url.values() if it["type"] == "paper"], tweets, cfg)
    return sorted(by_url.values(), key=lambda i: i["published"], reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7, help="window, counted back from now (one episode per week)")
    ap.add_argument("--only", default="arxiv,x,news", help="comma list from arxiv,x,news,hn")
    ap.add_argument("--merge", nargs="+", metavar="JSON", help="merge earlier harvests instead of fetching")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "sources.json").read_text())
    topics = load_topics(sync=not args.merge)

    if args.merge:
        items = [it for f in args.merge for it in json.loads(Path(f).read_text())]
    else:
        from concurrent.futures import ThreadPoolExecutor
        since = datetime.now(timezone.utc) - timedelta(days=args.days)
        only = set(args.only.split(","))

        def papers() -> list[dict]:
            got = arxiv(cfg, topics, since)
            paper_signals(got)
            return got

        runs = {
            "arxiv": papers,
            "x": lambda: x_paid(cfg, topics, since, k) if (k := key("TWITTERAPI_IO_KEY")) else (log("x: no "
                         "TWITTERAPI_IO_KEY, falling back to the public timeline (unreliable)") or x_public(cfg, topics, since)),
            "news": lambda: news(cfg, since, pool) if (pool := keys("NEWSDATAIO")) else (log("news: no NEWSDATAIO "
                            "key, skipped; cover news with web search") or []),
            "hn": lambda: hn(cfg, since),
        }

        def safe(name: str) -> list[dict]:
            try:
                got = runs[name]()
                log(f"{name}: {len(got)} items")
                return got
            except Exception as ex:  # one source failing shouldn't sink the run
                log(f"{name}: FAILED {ex}")
                return []

        with ThreadPoolExecutor(len(runs)) as pool_:
            items = [it for got in pool_.map(safe, [n for n in runs if n in only]) for it in got]
    merged = merge(items, topics, cfg)
    log(f"total: {len(merged)} after dedup, {sum(1 for i in merged if i['mentions'])} with cross-source mentions, "
        f"{sum(1 for i in merged if i['topics'])} tagged with a digest topic")
    json.dump(merged, sys.stdout, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
