#!/usr/bin/env bash
# Render review stills of a deep dive at chosen frames, in one or both designs, without touching episode.json/props.json.
# Usage: tools/deepdive/stills.sh <dd-dir> [--design light|dark|both] [--out <dir>] [--scale 1] <frame | seg:beat[@frac]>...
#   seg:beat picks a beat by segment id and 0-based index; @frac is how far into it (default 0.95, when its reveals are
#   done). Writes <out>/<light|dark>-f<frame>.png (default out: <dd-dir>/qa/stills) and a contact sheet per design.
# Waits for the shared render lock (RENDER_LOCK, as render_both.sh) unless NO_LOCK=1: a still launches Chrome too.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="$(cd "$1" && pwd)"; shift
DESIGN=both; OUT="$DIR/qa/stills"; SCALE=1; SPECS=()
while [ $# -gt 0 ]; do case "$1" in
  --design) DESIGN="$2"; shift;; --out) OUT="$2"; shift;; --scale) SCALE="$2"; shift;; *) SPECS+=("$1");; esac; shift; done
[ ${#SPECS[@]} -gt 0 ] || { echo "no frames given; beats are: $(python3 "$ROOT/tools/deepdive/dd.py" frames "$DIR" x 2>&1 | sed 's/.*beats: //')"; exit 2; }
case "$DESIGN" in light) D=blueprint-light;; dark) D=blueprint;; both) D=blueprint-light,blueprint;; *) D="$DESIGN";; esac
FRAMES=$(python3 "$ROOT/tools/deepdive/dd.py" frames "$DIR" "${SPECS[@]}")
mkdir -p "$OUT"
"$ROOT/.venv/bin/python" "$ROOT/tools/brand_audio.py" "$DIR/brand" >/dev/null  # the composition references the sonic logo

LOCK="${RENDER_LOCK:-$ROOT/.render.lock}"
if [ "${NO_LOCK:-0}" != 1 ]; then
  until mkdir "$LOCK" 2>/dev/null; do echo "$(date +%T) waiting for $LOCK"; sleep 20; done
  trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
fi
node "$ROOT/tools/deepdive/stills.mjs" "$DIR" "$OUT" "$D" "$SCALE" $FRAMES
for d in ${D//,/ }; do
  tag=$([ "$d" = blueprint ] && echo dark || ([ "$d" = blueprint-light ] && echo light || echo "$d"))
  files=(); for f in $FRAMES; do files+=("$OUT/$tag-f$(printf %05d "$f").png"); done
  [ ${#files[@]} -gt 1 ] || continue
  ins=(); for f in "${files[@]}"; do ins+=(-i "$f"); done
  cols=$(( ${#files[@]} < 3 ? ${#files[@]} : 3 ))
  ffmpeg -v error -y "${ins[@]}" -filter_complex "$(for i in "${!files[@]}"; do printf '[%d]scale=640:-1[v%d];' "$i" "$i"; done)$(for i in "${!files[@]}"; do printf '[v%d]' "$i"; done)xstack=inputs=${#files[@]}:layout=$(python3 -c "print('|'.join(f'{(i%$cols)*640}_{(i//$cols)*360}' for i in range(${#files[@]})))"):fill=black" \
    -frames:v 1 "$OUT/sheet-$tag.png"
  echo "contact sheet: $OUT/sheet-$tag.png"
done
