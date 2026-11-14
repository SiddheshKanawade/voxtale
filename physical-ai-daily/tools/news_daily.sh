#!/bin/bash
# Scheduled NewsData pull (launchd, twice a day; install with tools/install_news_schedule.sh). NewsData's free plan
# only reaches back ~36 h, so the weekly harvest relies on these saved files for the rest of the week. Keeps 21 days.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p news
stamp=$(date -u +%Y-%m-%dT%H%M)
/usr/bin/python3 -I tools/discover.py --only news --no-sync --days 2 > "news/.news-$stamp.json" 2>> news/pull.log
mv "news/.news-$stamp.json" "news/news-$stamp.json"
find news -name 'news-*.json' -mtime +21 -delete
echo "$(date -u +%FT%TZ) saved news-$stamp.json" >> news/pull.log
