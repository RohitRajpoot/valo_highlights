#!/usr/bin/env bash
set -euo pipefail

# make sure the output dir exists
mkdir -p segments

# skip header, then for each line: video_file,start_time,end_time,label
tail -n +2 annotations/Highlights.csv | while IFS=, read -r video start end label; do
  out="segments/${video%.*}_${start//:/-}.webm"
  if [ -f "$out" ]; then
    echo "⏭  Skipping existing $out"
    continue
  fi

  ffmpeg -hide_banner -loglevel error \
    -i "raw_videos/$video" \
    -ss "$start" -to "$end" \
    -c:v libvpx-vp9 -crf 30 -b:v 0 \
    -c:a libopus \
    "$out"
done
