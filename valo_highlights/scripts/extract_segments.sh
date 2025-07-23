#!/usr/bin/env bash
mkdir -p raw_videos/segments
tail -n +2 annotations/Highlights.csv | \
while IFS=, read -r video start end label; do
  out="raw_videos/segments/${video%.*}_${start//:/-}.webm"
  [ -f "$out" ] && continue
  ffmpeg -hide_banner -loglevel error \
    -i "raw_videos/$video" \
    -ss "$start" -to "$end" \
    -c copy "$out"
done
