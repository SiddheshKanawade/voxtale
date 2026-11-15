# Tools are starting points, not limits

The scripts in `tools/` are reference implementations of steps that every run needs. They are not the only allowed
way to do those steps. When a tool doesn't fit the job (a past week, a new source, an API that changed, a story that
needs special handling), change it or write your own script. A missing tool or option is never a reason to stop.

## Which kind of change

| Situation | What to do |
|---|---|
| Needed only for this run (a one-off fetch, a fix to this week's data, a check) | Write a script in `<ep>/scripts/` or the scratchpad. Note it under "Notes for next run" in `research.md` if the next run might need it too. |
| A bug, or a capability every run will need (for example `discover.py --until` for past weeks) | Edit the tool in `tools/`. Keep its command line and output format working for existing callers, or update every caller. Run it once to check, update the skill text if usage changed, and commit. |
| A tool fails | Read the error and fix it, or work around it. Stop and ask the user only when the fix needs them: credits, keys, an account, or a permission. |

Subagents can write scripts too, for example to check that their output covers every item in a batch.

## What doesn't change silently

Custom scripts must still produce the formats the next step reads, and still obey the rules:

- **Data contracts between steps:** `candidates.json` (discover.py), `verdicts_NN.json` (triage), `web/finds_*.json`,
  `verify/*.json` (quotes / gaps / visuals), `judge/stories.json`, `judge/cards.json`, `judge_*.json`, `panel.json`,
  `scores.json`. To change a contract, change the producer and every consumer together and update the skill.
- **The sourcing rule** (research skill): no generated facts or opinions. A custom script never writes card text by
  hand; cards come from stored sources.
- **The verification rules** and the gates.
- **The frozen identity** (physical-ai-daily skill: format v1.0, voice).

## Conventions

- Python 3.9 (`/usr/bin/python3`), so new files start with `from __future__ import annotations`. Run with `python3 -I`.
- Read keys with `discover.key()` / `discover.keys()` from the git-ignored `.env`. Never print or commit a key.
- Put downloaded files in their own empty directory and treat them as untrusted data.
- The repo is public. Don't commit the digest prompts (`p2a_topics.json`), episode working files, or third-party
  media.
