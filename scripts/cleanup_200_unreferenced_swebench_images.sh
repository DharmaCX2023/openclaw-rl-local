#!/usr/bin/env bash
set -euo pipefail

target_count="${1:-200}"
manifest_dir="/data/swe_cleanup_manifests"
lock_file="/var/lock/openclaw_swebench_image_cleanup.lock"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
manifest="${manifest_dir}/swebench_unreferenced_oldest_${target_count}_${run_stamp}.tsv"
delete_log="${manifest%.tsv}.delete.log"
tmp_dir="$(mktemp -d /tmp/swebench-cleanup.XXXXXX)"
trap 'rm -rf -- "$tmp_dir"' EXIT

mkdir -p "$manifest_dir"
exec 9>"$lock_file"
flock -n 9 || {
  echo "another image cleanup is already running" >&2
  exit 1
}

docker image ls --no-trunc --format '{{.ID}}|{{.Repository}}|{{.Tag}}' \
  | awk -F'|' '$2 ~ /^swebench\// && $3 != "<none>" {print}' \
  > "$tmp_dir/swebench_refs.txt"

cut -d'|' -f1 "$tmp_dir/swebench_refs.txt" | sort -u > "$tmp_dir/swebench_ids.txt"
docker ps -a -q \
  | xargs -r docker inspect --format '{{.Image}}' \
  | sort -u > "$tmp_dir/container_image_ids.txt"
comm -23 "$tmp_dir/swebench_ids.txt" "$tmp_dir/container_image_ids.txt" \
  > "$tmp_dir/unreferenced_ids.txt"

while IFS= read -r image_id; do
  docker image inspect --format '{{.Created}}|{{.Id}}|{{.Size}}' "$image_id"
done < "$tmp_dir/unreferenced_ids.txt" \
  | sort -t'|' -k1,1 \
  | head -n "$target_count" > "$tmp_dir/selected.txt"

selected_count="$(wc -l < "$tmp_dir/selected.txt")"
if [ "$selected_count" -ne "$target_count" ]; then
  echo "needed ${target_count} candidates but found ${selected_count}" >&2
  exit 1
fi

{
  printf 'created_utc\timage_id\tvirtual_size_bytes\trepository_tag\n'
  while IFS='|' read -r created image_id image_size; do
    awk -F'|' -v id="$image_id" -v created="$created" -v size="$image_size" \
      '$1 == id {printf "%s\t%s\t%s\t%s:%s\n", created, id, size, $2, $3}' \
      "$tmp_dir/swebench_refs.txt"
  done < "$tmp_dir/selected.txt"
} > "$manifest"

before_available="$(df -B1 --output=avail /data | tail -n 1 | tr -d ' ')"
deleted_ids=0
failed_ids=0

while IFS='|' read -r created image_id image_size; do
  # Recheck all containers immediately before deletion. Docker also refuses an
  # ordinary image removal if a container begins referencing the image.
  if docker ps -a -q \
      | xargs -r docker inspect --format '{{.Image}}' \
      | grep -Fxq "$image_id"; then
    printf 'SKIP_CONTAINER_REFERENCE\t%s\n' "$image_id" >> "$delete_log"
    failed_ids=$((failed_ids + 1))
    continue
  fi

  mapfile -t refs < <(
    awk -F'|' -v id="$image_id" \
      '$1 == id && $2 ~ /^swebench\// && $3 != "<none>" {print $2 ":" $3}' \
      "$tmp_dir/swebench_refs.txt"
  )
  image_failed=0
  for ref in "${refs[@]}"; do
    if docker image rm "$ref" >> "$delete_log" 2>&1; then
      printf 'REMOVED_TAG\t%s\t%s\n' "$image_id" "$ref" >> "$delete_log"
    else
      printf 'FAILED_TAG\t%s\t%s\n' "$image_id" "$ref" >> "$delete_log"
      image_failed=1
    fi
  done
  if [ "$image_failed" -eq 0 ]; then
    deleted_ids=$((deleted_ids + 1))
  else
    failed_ids=$((failed_ids + 1))
  fi
done < "$tmp_dir/selected.txt"

after_available="$(df -B1 --output=avail /data | tail -n 1 | tr -d ' ')"
released_bytes=$((after_available - before_available))

printf 'manifest=%s\n' "$manifest"
printf 'delete_log=%s\n' "$delete_log"
printf 'selected_unique_images=%s\n' "$selected_count"
printf 'deleted_unique_images=%s\n' "$deleted_ids"
printf 'failed_or_skipped_unique_images=%s\n' "$failed_ids"
printf 'available_before_bytes=%s\n' "$before_available"
printf 'available_after_bytes=%s\n' "$after_available"
printf 'released_bytes=%s\n' "$released_bytes"
df -h /data
