#!/usr/bin/env bash
set -euo pipefail

mkdir -p segments

tail -n +2 annotations/Highlights.csv | while IFS=, read -r video start end label; do
  out="segments/${video%.*}_${start//:/-}.webm"
  if [[ -f "$out" ]]; then
    echo "⏭  Skipping existing $out"
    continue
  fi

  ffmpeg -nostdin -hide_banner -loglevel error \
         -i "raw_videos/$video" \
         -ss "$start" -to "$end" \
         -c:v libvpx-vp9 -crf 30 -b:v 0 \
         -c:a libopus \
         "$out"
done
