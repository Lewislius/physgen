#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 OUTPUT_ROOT AUDIT_DIR [STEP]" >&2
  exit 2
fi

output_root=$1
audit_dir=$2
step=${3:-6}

if [[ ! -d "$output_root" ]]; then
  echo "missing output root: $output_root" >&2
  exit 2
fi
if ! [[ "$step" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEP must be a positive integer" >&2
  exit 2
fi

mkdir -p "$audit_dir"

while IFS= read -r -d '' video; do
  relative=${video#"$output_root"/}
  run=${relative%%/*}
  sample=${relative#*/}
  sample=${sample%%/*}
  target_dir="$audit_dir/$run"
  target="$target_dir/${sample}_dense.jpg"
  mkdir -p "$target_dir"
  ffmpeg -hide_banner -loglevel error -y -i "$video" \
    -vf "select='not(mod(n,$step))',scale=320:176:force_original_aspect_ratio=decrease,pad=320:196:0:20:black,drawtext=text='frame %{eif\\:n*$step\\:d}  |  %{pts\\:hms}':x=8:y=2:fontsize=14:fontcolor=white,tile=5x4:nb_frames=17:padding=3:margin=3:color=black" \
    -frames:v 1 -q:v 2 "$target"
done < <(find "$output_root" -name video.mp4 -print0 | sort -z)
