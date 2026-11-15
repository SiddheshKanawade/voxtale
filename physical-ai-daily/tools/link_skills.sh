#!/bin/bash
# Make every skill in this repo available to Claude Code on this machine: fetch the pinned submodules
# (vendor/remotion-skills) and symlink each .claude/skills/<name> into ~/.claude/skills/<name>.
# Existing entries that aren't symlinks are left alone and reported.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
git -C "$REPO" submodule update --init --recursive
mkdir -p "$HOME/.claude/skills"
for d in "$REPO"/.claude/skills/*; do
  name="$(basename "$d")"; dest="$HOME/.claude/skills/$name"
  if [ -L "$dest" ] || [ ! -e "$dest" ]; then ln -sfn "$REPO/.claude/skills/$name" "$dest"; echo "linked $name"
  else echo "skipped $name: $dest exists and isn't a symlink"; fi
done
