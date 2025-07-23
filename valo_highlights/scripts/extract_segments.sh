#!/usr/bin/env bash
set -euo pipefail

# ensure top‐level segments/ exists
mkdir -p segments

# skip header, loop over CSV
tail -n +2 annotations/Highlights.csv | while IFS=, read -r video start end label; do
  in="raw_videos/$video"
  out="segments/${video%.*}_${start//:/-}.webm"

  # skip if already done
  if [[ -f "$out" ]]; then
    echo "Skipping existing $out"
    continue
  fi

  ffmpeg -hide_banner -loglevel error \
    -i "$in" \
    -ss "$start" -to "$end" \
    -c:v libvpx-vp9 -b:v 1M \
    -c:a libopus \
    "$out"
done
