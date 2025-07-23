#!/usr/bin/env bash
set -euo pipefail

# ensure output dir exists
mkdir -p segments

# for each line in your CSV (skip header), cut out the labelled clip
tail -n +2 annotations/Highlights.csv | \
while IFS=, read -r video start end label; do
  out="segments/${video%.*}_${start//:/-}.webm"
  if [[ -f "$out" ]]; then
    echo "⏭  Skipping existing $out"
    continue
  fi
  ffmpeg -hide_banner -loglevel error \
    -i "raw_videos/$video" \
    -ss "$start" -to "$end" \
    -c copy \
    "$out"
done
