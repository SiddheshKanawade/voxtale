#!/usr/bin/env python3
"""Copy the Pixels2Actions newsletter's sub-topic profiles from prazo5's database into p2a_topics.json (git-ignored).

Usage: python3 -I tools/sync_p2a_topics.py

The newsletter's coverage comes from six narrow prazo5 accounts (VLA, world models, teleop data, ...). Each has a
hand-tuned topic prompt, ranked search keywords and ~20 X accounts its finder picked. discover.py reads them for the
weekly harvest, and the research skill uses the prompts as part of its editorial anchor. The file stays out of git
because the repo is public and the prompts are the newsletter's edge.

Needs RENDER_API_KEY in .env (or the environment) and psql. The account list comes from the p2a-shared env group's
PRAZO_ACCOUNTS; the database is opened read-only.
"""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.render.com/v1"
PRAZO_DB = "prazo5-postgres-database"
ENV_GROUP = "p2a-shared"


def render_key() -> str:
    if os.environ.get("RENDER_API_KEY"):
        return os.environ["RENDER_API_KEY"]
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("RENDER_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("RENDER_API_KEY not set (put it in .env)")


def api(path: str, key: str):
    req = urllib.request.Request(API + path, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    key = render_key()
    group = next(g["envGroup"] if "envGroup" in g else g for g in api("/env-groups?limit=50", key)
                 if (g.get("envGroup") or g)["name"] == ENV_GROUP)
    group = api(f"/env-groups/{group['id']}", key)
    emails = next(v["value"] for v in group["envVars"] if v["key"] == "PRAZO_ACCOUNTS").replace(" ", "").split(",")
    db = next(p["postgres"] for p in api("/postgres?limit=50", key) if p["postgres"]["name"] == PRAZO_DB)
    url = api(f"/postgres/{db['id']}/connection-info", key)["externalConnectionString"]
    sql = f"""select json_agg(row_to_json(t)) from (
      select split_part(u.email, '@', 1) as name, p.raw_prompt as prompt,
             p.anchor_json -> 'search_keywords' as keywords,
             (select json_agg(s.handle order by s.relevance_score desc nulls last) from sources s
               where s.profile_id = p.id and s.enabled and s.platform = 'x') as x_accounts
      from users u join topic_profiles p on p.user_id = u.id and p.is_active
      where u.email = any(array[{",".join("'" + e.replace("'", "") + "'" for e in emails)}])
      order by 1) t"""
    out = subprocess.run(["psql", url, "-X", "-q", "-A", "-t", "-c", sql], capture_output=True, text=True,
                         env={**os.environ, "PGOPTIONS": "-c default_transaction_read_only=on"})
    if out.returncode:
        sys.exit(f"psql failed: {out.stderr.strip()}")
    topics = json.loads(out.stdout)
    (ROOT / "p2a_topics.json").write_text(json.dumps(topics, indent=1, ensure_ascii=False) + "\n")
    for t in topics:
        print(f"{t['name']}: {len(t['keywords'] or [])} keywords, {len(t['x_accounts'] or [])} X accounts")


if __name__ == "__main__":
    main()
