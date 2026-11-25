#!/usr/bin/env bash
# Render a deep dive in both designs, light (blueprint-light) then dark (blueprint), one render at a time.
# Usage: tools/deepdive/render_both.sh <dd-episode-dir> [--skip-voice] [--stills-only]
# Ends up as out/physical-ai-daily-<dir>-light.mp4 and -dark.mp4. Renders are heavy, so parallel agents share one lock
# with tools/render_both.sh: RENDER_LOCK (default: physical-ai-daily/.render.lock). The script waits for it, holds it per
# render, and releases it on exit. The design is restored to light afterwards.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="$(cd "$1" && pwd)"; shift
LOCK="${RENDER_LOCK:-$ROOT/.render.lock}"
NAME="physical-ai-daily-$(basename "$DIR")"
setdesign() { python3 -c "import json,sys;p=sys.argv[1];e=json.load(open(p));e['design']=sys.argv[2];open(p,'w').write(json.dumps(e,indent=2,ensure_ascii=False)+'\n')" "$1" "$2"; }
lock() { until mkdir "$LOCK" 2>/dev/null; do echo "$(date +%T) waiting for $LOCK"; sleep 20; done; trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT; }
unlock() { rmdir "$LOCK"; trap - EXIT; }

for pair in "blueprint-light light" "blueprint dark"; do
  read -r design suffix <<< "$pair"
  # both files: voice.py copies the design into props.json, but --skip-voice doesn't run it
  setdesign "$DIR/episode.json" "$design"
  if [[ -f "$DIR/props.json" ]]; then setdesign "$DIR/props.json" "$design"; fi
  lock; "$ROOT/tools/deepdive/render.sh" "$DIR" "$@"; unlock
  if [[ " $* " != *" --stills-only "* ]]; then
    mv "$DIR/out/$NAME.mp4" "$DIR/out/$NAME-$suffix.mp4"
    echo "final: $DIR/out/$NAME-$suffix.mp4"
  else
    cp "$DIR/qa/contact_sheet.png" "$DIR/qa/contact_sheet_$suffix.png"
  fi
done
setdesign "$DIR/episode.json" blueprint-light
if [[ -f "$DIR/props.json" ]]; then setdesign "$DIR/props.json" blueprint-light; fi
