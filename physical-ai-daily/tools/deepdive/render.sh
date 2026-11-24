#!/usr/bin/env bash
# Render a deep dive end to end, as tools/render.sh does for the digest: narration -> cue check -> QA stills -> MP4 ->
# loudness-normalized final. Composition DeepDive (video/src/deepdive/index.ts).
# Usage: tools/deepdive/render.sh <dd-episode-dir> [--stills-only] [--skip-voice]   (--skip-voice reuses audio/ + props.json)
# 4K crf 15 by default; SCALE=1 CRF=23 CONCURRENCY=2 for a quick draft on a low-memory machine.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EP="$(cd "$1" && pwd)"
NAME="physical-ai-daily-$(basename "$EP")"
ENTRY=src/deepdive/index.ts

[[ " $* " == *" --skip-voice "* ]] || "$ROOT/.venv/bin/python" "$ROOT/tools/voice.py" "$EP"
# Every reveal waits for its cue word: a cue that isn't in its beat's narration would fall back to a guessed time.
python3 "$ROOT/tools/deepdive/dd.py" check "$EP" || { echo "fix the errors above (video/src/deepdive/DeepDive.tsx documents the types)"; exit 1; }

# Sonic logo (generated, identical every episode), served from the episode dir like other media.
"$ROOT/.venv/bin/python" "$ROOT/tools/brand_audio.py" "$EP/brand" >/dev/null

cd "$ROOT/video"
mkdir -p "$EP/qa" "$EP/out"
rm -f "$EP"/qa/f*.png
TOTAL=$(python3 "$ROOT/tools/deepdive/dd.py" total "$EP")  # sting and end tail included
for f in $(python3 -c "print(' '.join(str(int($TOTAL*k/7)) for k in range(1,7)))"); do
  npx remotion still $ENTRY DeepDive "$EP/qa/f$(printf %05d "$f").png" --frame="$f" --props="$EP/props.json" --public-dir="$EP" >/dev/null
done
ffmpeg -v error -y -pattern_type glob -i "$EP/qa/f*.png" -vf "scale=960:-1,tile=2x3" -frames:v 1 "$EP/qa/contact_sheet.png"
echo "contact sheet: $EP/qa/contact_sheet.png"
[[ " $* " == *" --stills-only "* ]] && exit 0

# 4K by default, as the digest (tools/render.sh): --scale=2 renders 1920x1080 at 3840x2160 so diagrams and text stay sharp.
npx remotion render $ENTRY DeepDive "$EP/out/$NAME.raw.mp4" --props="$EP/props.json" --public-dir="$EP" \
  --concurrency="${CONCURRENCY:-4}" --scale="${SCALE:-2}" --crf="${CRF:-15}"
# YouTube normalizes to about -14 LUFS; TP -2 leaves room for the true-peak overshoot AAC adds (see tools/render.sh).
ffmpeg -v error -y -i "$EP/out/$NAME.raw.mp4" -c:v copy -af "loudnorm=I=-14:TP=-2:LRA=11" -c:a aac -b:a 192k "$EP/out/$NAME.mp4"
rm "$EP/out/$NAME.raw.mp4"
# Shots under 1.5 s flash past as glitches; warn (a deliberate fast cut is fine, say why in the hand-off).
python3 "$ROOT/tools/shot_check.py" "$EP/out/$NAME.mp4" || echo "WARN: short shots above; fix the clip unless the fast cut is intended"
echo "final: $EP/out/$NAME.mp4"
