#!/usr/bin/env bash
set -euo pipefail

mkdir -p segments

# segments.tsv has lines: <path>\t<start>\t<end>
while IFS=$'\t' read -r path start end; do
  name=$(basename "$path" .mp4)
  out="segments/${name}_${start//:/-}-${end//:/-}.webm"
  if [[ -f "$out" ]]; then
    echo "Skipping $out"
    continue
  fi

  # (we’ll override this ffmpeg call below)
  ffmpeg -hide_banner -loglevel error \
    -i "$path" -ss "$start" -to "$end" \
    -c:v libvpx-vp9 -cpu-used 4 -threads 4 \
    -c:a libopus -b:a 64k \
    "$out"

done < segments.tsv
