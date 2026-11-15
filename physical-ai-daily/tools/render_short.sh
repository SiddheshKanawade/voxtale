#!/usr/bin/env bash
# Render a 9:16 promotional Short: narration -> cover frame + QA contact sheet -> MP4 -> loudness-normalised final.
# Usage: tools/render_short.sh <episode>/shorts/<slug> [--stills-only] [--skip-voice] [--design blueprint|blueprint-light]
# The Short's folder holds its own episode.json (one story segment, plus "hook" and "cta") and assets/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SH="$(cd "$1" && pwd)"; shift
STILLS=0; VOICE=1; DESIGN=""
while [ $# -gt 0 ]; do case "$1" in
  --stills-only) STILLS=1;; --skip-voice) VOICE=0;; --design) DESIGN="$2"; shift;; esac; shift; done
NAME="$(basename "$(dirname "$(dirname "$SH")")")-$(basename "$SH")"

[ $VOICE = 1 ] && "$ROOT/.venv/bin/python" "$ROOT/tools/voice.py" "$SH"
if [ -n "$DESIGN" ]; then  # render a design without editing episode.json
  python3 -c "import json,sys;p=json.load(open('$SH/props.json'));p['design']='$DESIGN';json.dump(p,open('$SH/props.json','w'),indent=2)"
fi
D="$(python3 -c "import json;print(json.load(open('$SH/props.json')).get('design') or 'blueprint-light')")"
cd "$ROOT/video"; mkdir -p "$SH/qa" "$SH/out"; rm -f "$SH"/qa/f*.png
TOTAL=$(python3 -c "import json;print(int((sum(s['duration'] for s in json.load(open('$SH/props.json'))['segments'])+2.2)*30))")
# Frame 0 is the cover in the Shorts grid and the first thing a viewer sees: always check it.
for f in 0 $(python3 -c "print(' '.join(str(int($TOTAL*k/6)) for k in range(1,6)))"); do
  npx remotion still src/shorts/index.ts Short "$SH/qa/f$(printf %05d $f).png" --frame="$f" --props="$SH/props.json" --public-dir="$SH" >/dev/null
done
ffmpeg -v error -y -pattern_type glob -i "$SH/qa/f*.png" -vf "scale=360:-1,tile=6x1" -frames:v 1 "$SH/qa/contact_sheet_$D.png"
echo "contact sheet: $SH/qa/contact_sheet_$D.png"
[ $STILLS = 1 ] && exit 0
npx remotion render src/shorts/index.ts Short "$SH/out/$NAME-$D.raw.mp4" --props="$SH/props.json" --public-dir="$SH" --concurrency=4
ffmpeg -v error -y -i "$SH/out/$NAME-$D.raw.mp4" -c:v copy -af "loudnorm=I=-14:TP=-2:LRA=11" -c:a aac -b:a 192k "$SH/out/$NAME-$D.mp4"
rm "$SH/out/$NAME-$D.raw.mp4"
echo "final: $SH/out/$NAME-$D.mp4"
