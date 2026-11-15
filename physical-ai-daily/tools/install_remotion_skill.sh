#!/bin/bash
# Install Remotion's official agent skill (remotion-dev/skills, "remotion-best-practices") into ~/.claude/skills.
# It is NOT vendored into this repo: it has no open-source license (Remotion License, source-available), so this
# public repo pins and fetches it instead of redistributing it. Fetched with gh (no npx installer runs).
# Usage: tools/install_remotion_skill.sh [commit]   (default: the commit reviewed on 2026-11-16)
set -euo pipefail
SHA="${1:-7e541aa4d7361dfaca0780a565a47307eb7bdfbb}"
DEST="$HOME/.claude/skills/remotion-best-practices"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
gh api "repos/remotion-dev/skills/tarball/$SHA" > "$TMP/src.tgz"
mkdir "$TMP/src" && tar -xzf "$TMP/src.tgz" -C "$TMP/src" --strip-components=1
[ -f "$TMP/src/skills/remotion-best-practices/SKILL.md" ] || { echo "unexpected layout at $SHA" >&2; exit 1; }
rm -rf "$DEST" && cp -R "$TMP/src/skills/remotion-best-practices" "$DEST"
echo "installed remotion-best-practices @ ${SHA:0:12} -> $DEST ($(find "$DEST" -type f | wc -l | tr -d ' ') files)"
