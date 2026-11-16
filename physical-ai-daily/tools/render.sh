#!/usr/bin/env bash
# Render an episode end to end: narration -> QA stills -> MP4 -> loudness-normalized final.
# Usage: tools/render.sh episodes/<episode-dir> [--stills-only] [--skip-voice]   (--skip-voice reuses audio/ + props.json)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EP="$(cd "$1" && pwd)"
NAME="physical-ai-daily-$(basename "$EP")"

[[ " $* " == *" --skip-voice "* ]] || "$ROOT/.venv/bin/python" "$ROOT/tools/voice.py" "$EP"

# Sonic logo + section tick (generated, identical every episode), served from the episode dir like other media.
"$ROOT/.venv/bin/python" "$ROOT/tools/brand_audio.py" "$EP/brand" >/dev/null

cd "$ROOT/video"
mkdir -p "$EP/qa" "$EP/out"
rm -f "$EP"/qa/f*.png
TOTAL=$(python3 -c "import json;print(int((sum(s['duration'] for s in json.load(open('$EP/props.json'))['segments'])+3.2)*30))")  # + sting + end tail
for f in $(python3 -c "print(' '.join(str(int($TOTAL*k/7)) for k in range(1,7)))"); do
  npx remotion still src/index.ts Episode "$EP/qa/f$f.png" --frame="$f" --props="$EP/props.json" --public-dir="$EP" >/dev/null
done
ffmpeg -v error -y -pattern_type glob -i "$EP/qa/f*.png" -vf "scale=960:-1,tile=2x3" -frames:v 1 "$EP/qa/contact_sheet.png"
echo "contact sheet: $EP/qa/contact_sheet.png"
[[ " $* " == *" --stills-only "* ]] && exit 0

# 4K by default (user, 2026-11-16: output quality): --scale=2 renders the 1920x1080 composition at 3840x2160
# (Shorts 2160x3840) so text and graphics stay sharp, and YouTube serves 4K uploads with a better codec even at 1080p.
# --crf 15 (lower = better; Remotion's default is 18). SCALE=1 CRF=18 for a quick draft.
npx remotion render src/index.ts Episode "$EP/out/$NAME.raw.mp4" --props="$EP/props.json" --public-dir="$EP" --concurrency=4 --scale="${SCALE:-2}" --crf="${CRF:-15}"
# YouTube normalizes to about -14 LUFS; master there so narration isn't turned up with added noise.
# TP -2 (not -1.5): AAC encoding adds ~0.2-0.5 dB of true peak after loudnorm, and the target is <= -1.5 dBTP.
ffmpeg -v error -y -i "$EP/out/$NAME.raw.mp4" -c:v copy -af "loudnorm=I=-14:TP=-2:LRA=11" -c:a aac -b:a 192k "$EP/out/$NAME.mp4"
rm "$EP/out/$NAME.raw.mp4"
echo "final: $EP/out/$NAME.mp4"
