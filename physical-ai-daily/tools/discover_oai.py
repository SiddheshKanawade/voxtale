"""Fallback harvest: tools/discover.py unchanged except arxiv(), which reads arXiv OAI-PMH (oaipmh.arxiv.org)
instead of the export API. Use it when the export API keeps answering 429 (it did all through the 2026-11-17 backlog
run). Same arguments as discover.py, e.g.:

  python3 tools/discover_oai.py --days 8 --until 2026-10-25 --only arxiv > <ep>/candidates-arxiv.json

Same filter as discover.arxiv(): every cs.RO paper, plus cs.CV/LG/AI/eess.SY papers whose title or abstract contains
any of our terms or digest keywords; kept when v1 ('created') falls in the window. OAI's 'created' is not always the
v1 date, so papers whose ID month (YYMM) is before the window are dropped too; still check each carded paper's v1 date
on its abs page."""
from __future__ import annotations
import re, sys, time, urllib.error, urllib.parse, xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import discover

OAI = "{http://www.openarchives.org/OAI/2.0/}"
AX = "{http://arxiv.org/OAI/arXiv/}"

def fetch(url):
    for i in range(12):
        try:
            return discover.get(url)
        except urllib.error.HTTPError as ex:
            if ex.code in (503, 429):
                wait = int(ex.headers.get("Retry-After") or 30)
                discover.log(f"oai: {ex.code}, waiting {wait}s"); time.sleep(wait); continue
            raise
        except Exception as ex:
            discover.log(f"oai: {ex}, retrying"); time.sleep(20)
    raise RuntimeError("oai: gave up")

def arxiv(cfg, topics, since, until, cap=0):
    terms = list(dict.fromkeys(t.lower() for t in cfg.get("arxiv_terms", []) + [k for tp in topics
                                                                              for k in tp.get("keywords") or []]))
    core, also = set(cfg["arxiv_categories"]), set(cfg.get("arxiv_also", []))
    out, seen, pages = [], set(), 0
    for st in ("cs", "eess"):
        url = (f"https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXiv&set={st}"
               f"&from={since:%Y-%m-%d}&until={datetime.now(timezone.utc):%Y-%m-%d}")
        while url:
            root = ET.fromstring(fetch(url)); pages += 1
            for rec in root.iter(f"{OAI}record"):
                m = rec.find(f"{OAI}metadata/{AX}arXiv")
                if m is None:
                    continue
                pid = m.findtext(f"{AX}id")
                created = datetime.fromisoformat(m.findtext(f"{AX}created")).replace(tzinfo=timezone.utc)
                if not (since <= created < until) or pid in seen or (pid[:4].isdigit() and pid[:4] < f"{since:%y%m}"):
                    continue
                cats = (m.findtext(f"{AX}categories") or "").split()
                title = " ".join((m.findtext(f"{AX}title") or "").split())
                abstract = " ".join((m.findtext(f"{AX}abstract") or "").split())
                blob = (title + " " + abstract).lower()
                if not (core & set(cats) or (also & set(cats) and any(t in blob for t in terms))):
                    continue
                seen.add(pid)
                authors = [" ".join(filter(None, [a.findtext(f"{AX}forenames"), a.findtext(f"{AX}keyname")]))
                           for a in m.iter(f"{AX}author")]
                out.append({"type": "paper", "id": pid, "url": f"https://arxiv.org/abs/{pid}", "title": title,
                            "text": abstract, "authors": authors,
                            "published": created.isoformat(), "categories": cats,
                            "comment": " ".join((m.findtext(f"{AX}comments") or "").split())})
            tok = root.find(f"{OAI}ListRecords/{OAI}resumptionToken")
            url = (f"https://oaipmh.arxiv.org/oai?verb=ListRecords&resumptionToken={urllib.parse.quote(tok.text)}"
                   if tok is not None and tok.text else None)
            discover.log(f"oai: set {st}, page {pages}, kept {len(out)}")
            time.sleep(3)
    return out

discover.arxiv = arxiv
sys.argv = ["discover.py"] + sys.argv[1:]
discover.main()
