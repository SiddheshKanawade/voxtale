#!/usr/bin/env bash
# Render an episode or a Short in both designs, light (blueprint-light) then dark (blueprint), one render at a time.
# Usage: tools/render_both.sh episodes/<ep> | episodes/<ep>/shorts/<slug>   [--skip-voice]
# Episodes end up as out/physical-ai-daily-<ep>-light.mp4 and -dark.mp4; Shorts as render_short.sh names them.
# Renders are heavy, so parallel agents share one lock: RENDER_LOCK (default: physical-ai-daily/.render.lock).
# The script waits for it, holds it per render, and releases it on exit. The design is restored to light afterwards.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$(cd "$1" && pwd)"; shift
LOCK="${RENDER_LOCK:-$ROOT/.render.lock}"
setdesign() { python3 -c "import json,sys;p=sys.argv[1];e=json.load(open(p));e['design']=sys.argv[2];open(p,'w').write(json.dumps(e,indent=2,ensure_ascii=False)+'\n')" "$1" "$2"; }
lock() { until mkdir "$LOCK" 2>/dev/null; do echo "$(date +%T) waiting for $LOCK"; sleep 20; done; trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT; }
unlock() { rmdir "$LOCK"; trap - EXIT; }

if [[ "$DIR" == */shorts/* ]]; then
  lock; "$ROOT/tools/render_short.sh" "$DIR" "$@"; unlock
  lock; "$ROOT/tools/render_short.sh" "$DIR" --skip-voice --design blueprint; unlock
  setdesign "$DIR/props.json" blueprint-light
else
  NAME="physical-ai-daily-$(basename "$DIR")"
  EXTRA=("$@")
  for pair in "blueprint-light light" "blueprint dark"; do
    read -r design suffix <<< "$pair"
    setdesign "$DIR/episode.json" "$design"
    lock; "$ROOT/tools/render.sh" "$DIR" ${EXTRA[@]+"${EXTRA[@]}"}; unlock
    mv "$DIR/out/$NAME.mp4" "$DIR/out/$NAME-$suffix.mp4"
    echo "final: $DIR/out/$NAME-$suffix.mp4"
    EXTRA=()  # render.sh writes the design into props.json only when it voices (cached, so cheap): don't skip it
  done
  setdesign "$DIR/episode.json" blueprint-light
  if [[ -f "$DIR/props.json" ]]; then setdesign "$DIR/props.json" blueprint-light; fi
fi
