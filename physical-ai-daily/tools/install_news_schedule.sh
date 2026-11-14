#!/bin/bash
# Install (or with --remove, uninstall) the launchd job that runs tools/news_daily.sh at 08:00 and 20:00 local time.
# The job runs only while this Mac is awake; launchd runs a missed slot once the Mac wakes. Logs: news/pull.log.
set -euo pipefail
label=com.pixels2actions.news
plist="$HOME/Library/LaunchAgents/$label.plist"
script="$(cd "$(dirname "$0")" && pwd)/news_daily.sh"
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
if [ "${1:-}" = "--remove" ]; then rm -f "$plist"; echo "removed $label"; exit 0; fi
chmod +x "$script"
cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$script</string></array>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardErrorPath</key><string>/tmp/$label.err</string>
</dict>
</plist>
PLIST
launchctl bootstrap "gui/$(id -u)" "$plist"
echo "installed $label: $script at 08:00 and 20:00"
