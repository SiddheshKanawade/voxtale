#!/usr/bin/env bash
# Export the Pixels2Actions logo files into brand/: YouTube profile picture (800x800, circle-safe), the mark on a
# transparent background (1024x1024), and the mark + wordmark lockup on the light page colour (1600x400).
# Usage: tools/brand_assets.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/brand"
cd "$ROOT/video"
for id in ChannelIcon LogoMark LockupLight; do
  npx remotion still src/index.ts "$id" "$ROOT/brand/$id.png" >/dev/null
  echo "$ROOT/brand/$id.png"
done

# Channel banner (2560x1440, key content inside YouTube's 1546x423 safe area) with the three newest episode thumbnails.
mkdir -p "$ROOT/brand/thumbs"
THUMBS=()
for t in $(ls -d "$ROOT"/episodes/*-weekly/out/thumbnail.png 2>/dev/null | sort -r | head -3); do
  n="$(basename "$(dirname "$(dirname "$t")")").png"; cp "$t" "$ROOT/brand/thumbs/$n"; THUMBS+=("\"thumbs/$n\"")
done
if [ ${#THUMBS[@]} -gt 0 ]; then
  echo "{\"thumbs\": [$(IFS=,; echo "${THUMBS[*]}")]}" > "$ROOT/brand/banner_props.json"
  npx remotion still src/index.ts ChannelBanner "$ROOT/brand/ChannelBanner.png" --props="$ROOT/brand/banner_props.json" --public-dir="$ROOT/brand" >/dev/null
  echo "$ROOT/brand/ChannelBanner.png"
fi
