#!/usr/bin/env python3
"""What people are saying about one story: X, Hacker News and YouTube, for the research skill's context step.

Usage: python3 -I tools/chatter.py "<query>" [--url URL ...] [--arxiv ID] [--details 3] [--transcripts N --out DIR] > chatter.json

Run it for each shortlisted story, after it's picked (not during the harvest). The query is the story's name as people
would write it ("Gemini Robotics 2.5", "Figure 03 BMW"). --url and --arxiv add searches for posts linking the primary
source, which catch the posts that don't name it.

| Source  | Method                                                                  | Key                |
|---------|-------------------------------------------------------------------------|--------------------|
| X       | twitterapi.io advanced_search, queryType=Top (most engaged, any date)   | TWITTERAPI_IO_KEY  |
| HN      | Algolia search over all time (comments are where experts push back)     | none               |
| YouTube | yt-dlp "ytsearch": who has already covered it, and how many views       | none               |

--details N (default 3) adds, for the N most-viewed videos, the upload date, likes, comment count, the full
description (channels often summarise the story there) and the top 20 comments (where viewers push back). --transcripts
N also saves the auto-captions of the N most-viewed videos to DIR as plain text, to read how other channels explained
it. YouTube rate-limits caption downloads (HTTP 429) after a burst of them, e.g. several stories' chatter run in
parallel, and it can stick to one video for hours, so captions get one retry. Videos without usable captions (still
429, or none) are transcribed locally from their audio with whisper.cpp when it's installed (`brew install
whisper-cpp`, model in models/whisper/, see WHISPER_MODEL); otherwise the descriptions and comments are what's left.
Reddit blocks unauthenticated API reads: search it with WebSearch
(`site:reddit.com <query>`) instead.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discover import _tweet, get, key, log  # noqa: E402


def x(api_key: str, queries: list[str]) -> list[dict]:
    seen, out = set(), []
    for q in queries:
        try:
            payload = json.loads(get("https://api.twitterapi.io/twitter/tweet/advanced_search?" + urllib.parse.urlencode(
                {"query": f"{q} -filter:retweets", "queryType": "Top"}), {"X-API-Key": api_key}))
        except Exception as ex:
            log(f"x: {q!r}: FAILED {ex}")
            continue
        for t in filter(None, map(_tweet, payload.get("tweets") or [])):
            if t["id"] not in seen:
                seen.add(t["id"])
                out.append(t)
    return sorted(out, key=lambda t: t.get("likes") or 0, reverse=True)[:25]


def hn(queries: list[str]) -> list[dict]:
    seen, out = set(), []
    for q in queries:
        payload = json.loads(get("https://hn.algolia.com/api/v1/search?" + urllib.parse.urlencode(
            {"query": q, "tags": "story", "hitsPerPage": 10})))
        for h in payload.get("hits") or []:
            if h["objectID"] not in seen and (h.get("points") or 0) >= 5:
                seen.add(h["objectID"])
                out.append({"title": h.get("title"), "points": h.get("points"), "comments": h.get("num_comments"),
                            "url": h.get("url"), "discussion": f"https://news.ycombinator.com/item?id={h['objectID']}",
                            "created": h.get("created_at")})
    return sorted(out, key=lambda h: h["points"] or 0, reverse=True)


def youtube(query: str, n: int = 12) -> list[dict]:
    run = subprocess.run(["yt-dlp", "--flat-playlist", "-j", f"ytsearch{n}:{query}"], capture_output=True, text=True)
    vids = []
    for line in run.stdout.splitlines():
        d = json.loads(line)
        vids.append({"title": d.get("title"), "channel": d.get("channel"), "views": d.get("view_count"),
                     "duration_s": d.get("duration"), "url": d.get("url")})
    return sorted(vids, key=lambda v: v["views"] or 0, reverse=True)


def details(v: dict) -> None:
    run = subprocess.run(["yt-dlp", "-j", "--skip-download", "--ignore-no-formats-error", "--write-comments",
                          "--extractor-args", "youtube:max_comments=20,all,0,0;comment_sort=top", v["url"]],
                         capture_output=True, text=True)
    try:
        d = json.loads(run.stdout)
    except ValueError:
        log(f"youtube: no details for {v['url']}")
        return
    v.update({"uploaded": d.get("upload_date"), "likes": d.get("like_count"), "comment_count": d.get("comment_count"),
              "description": (d.get("description") or "")[:3000],
              "top_comments": [{"likes": c.get("like_count"), "text": (c.get("text") or "")[:400]}
                               for c in sorted(d.get("comments") or [], key=lambda c: -(c.get("like_count") or 0))]})


WHISPER_MODEL = Path(__file__).resolve().parent.parent / "models" / "whisper" / "ggml-base.en.bin"


def _captions(url: str, stem: Path) -> Path | None:
    for wait in (0, 20):  # the 429 often sticks to a video for hours; whisper is the real fallback
        time.sleep(wait)
        run = subprocess.run(["yt-dlp", "--skip-download", "--write-auto-subs", "--write-subs", "--sub-langs", "en",
                              "--sub-format", "vtt", "--sleep-subtitles", "3", "-o", str(stem), url],
                             capture_output=True, text=True)
        vtt = next(iter(sorted(stem.parent.glob(stem.name + "*.vtt"))), None)
        if vtt or "429" not in run.stderr:
            return vtt
        log(f"youtube: captions 429 for {url}, retrying")
    return None


def _whisper(url: str, stem: Path) -> str | None:
    cli = shutil.which("whisper-cli")
    if not cli or not WHISPER_MODEL.exists():
        return None
    audio = stem.with_suffix(".wav")
    subprocess.run(["yt-dlp", "-f", "ba", "-x", "--audio-format", "wav", "--postprocessor-args",
                    "ffmpeg:-ar 16000 -ac 1", "-o", str(stem.with_suffix(".%(ext)s")), url], capture_output=True)
    if not audio.exists():
        return None
    subprocess.run([cli, "-m", str(WHISPER_MODEL), "-f", str(audio), "-otxt", "-of", str(stem), "-np"],
                   capture_output=True)
    audio.unlink()
    txt = stem.with_suffix(".txt")
    return " ".join(txt.read_text().split()) if txt.exists() else None


def transcript(url: str, out: Path) -> Path | None:
    stem = out / re.sub(r"\W+", "_", url.rsplit("=", 1)[-1])
    txt = stem.with_suffix(".txt")
    if txt.exists():
        return txt
    vtt = _captions(url, stem)
    if vtt:
        lines = [re.sub(r"<[^>]+>", "", l).strip() for l in vtt.read_text().splitlines()
                 if l.strip() and "-->" not in l and not l.startswith(("WEBVTT", "Kind:", "Language:"))]
        text = " ".join(dict.fromkeys(lines))  # auto-captions repeat each line as it scrolls
        for f in out.glob(stem.name + "*.vtt"):
            f.unlink()
        source = "captions"
    else:
        text, source = _whisper(url, stem), "whisper"
    if not text:
        log(f"youtube: no transcript for {url} (captions unavailable"
            + ("" if shutil.which("whisper-cli") and WHISPER_MODEL.exists() else ", whisper.cpp not installed") + ")")
        return None
    txt.write_text(text + "\n")
    log(f"youtube: transcript for {url} from {source}")
    return txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--url", nargs="*", default=[], help="primary-source URLs to find posts linking them")
    ap.add_argument("--arxiv", help="arXiv id, to catch posts linking any mirror of the paper")
    ap.add_argument("--details", type=int, default=3, help="description, likes and top comments of the N top videos")
    ap.add_argument("--transcripts", type=int, default=0)
    ap.add_argument("--out", type=Path, help="directory for transcripts")
    args = ap.parse_args()

    names = [f'"{args.query}"'] + [f'url:"{urllib.parse.urlsplit(u).netloc + urllib.parse.urlsplit(u).path}"'
                                    for u in args.url] + ([f'"{args.arxiv}"'] if args.arxiv else [])
    k = key("TWITTERAPI_IO_KEY")
    result = {
        "query": args.query,
        "x": x(k, names) if k else (log("x: no TWITTERAPI_IO_KEY, skipped") or []),
        "hn": hn([args.query] + args.url + ([args.arxiv] if args.arxiv else [])),
        "youtube": youtube(args.query),
    }
    for v in result["youtube"][:args.details]:
        details(v)
    if args.transcripts and args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for v in result["youtube"][:args.transcripts]:
            v["transcript"] = str(transcript(v["url"], args.out) or "")
    log(f"x: {len(result['x'])}, hn: {len(result['hn'])}, youtube: {len(result['youtube'])}")
    json.dump(result, sys.stdout, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
