#!/usr/bin/env bash
# User-run download helper. No extraction, project imports or inference.
set -euo pipefail
umask 077

ucf_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ucf_feature_dir="$ucf_project_root/data/features/official/ucf"
ucf_setup_dir="$ucf_project_root/.local/setup/ucf"
ucf_partial="$ucf_feature_dir/UCFClipFeatures.shared.zip.part"
ucf_complete="$ucf_feature_dir/UCFClipFeatures.shared.zip"
ucf_share_url='https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/pengwu_stu_xidian_edu_cn/Ea86YOcp5z9KhRFDQm9a8zwBcGiGGg5BuBJtgmCVByazBQ?e=tqHLHt'
ucf_download_url='https://stuxidianeducn-my.sharepoint.com/personal/pengwu_stu_xidian_edu_cn/_layouts/15/download.aspx?SourceUrl=https%3A%2F%2Fstuxidianeducn-my.sharepoint.com%2Fpersonal%2Fpengwu_stu_xidian_edu_cn%2FDocuments%2FVadCLIP-AAAI2024%2Ffeatures%2FUCF-Crime%2FUCFClipFeatures.zip'

for ucf_command in wget file zipinfo; do
  if ! command -v "$ucf_command" >/dev/null 2>&1; then
    printf 'Required command is missing: %s\n' "$ucf_command" >&2
    exit 1
  fi
done
mkdir -p "$ucf_feature_dir" "$ucf_setup_dir"
if [ -e "$ucf_complete" ]; then
  printf 'A final file already exists; leaving it unchanged: %s\n' "$ucf_complete" >&2
  exit 1
fi
if [ -s "$ucf_partial" ]; then
  case "$(file -b "$ucf_partial")" in
    *"Zip archive"*|*"ZIP archive"*) ;;
    *)
      printf 'Existing partial file is not identified as ZIP; do not resume it: %s\n' "$ucf_partial" >&2
      exit 1
      ;;
  esac
fi

ucf_cookie_file="$(mktemp "$ucf_setup_dir/ucf-share-cookies.XXXXXX")"
trap 'rm -f -- "$ucf_cookie_file"' EXIT

printf 'Opening the original public sharing link to obtain session cookies...\n'
if ! wget --no-verbose --user-agent='Mozilla/5.0' \
    --timeout=60 --tries=2 --max-redirect=20 \
    --save-cookies "$ucf_cookie_file" --keep-session-cookies \
    -O /dev/null "$ucf_share_url"; then
  printf 'The sharing-link request failed. Report the HTTP error; no feature extraction was performed.\n' >&2
  exit 1
fi

printf 'Downloading with cookies from the sharing link...\n'
if ! wget -c --user-agent='Mozilla/5.0' \
    --timeout=60 --tries=5 --max-redirect=20 \
    --load-cookies "$ucf_cookie_file" \
    -O "$ucf_partial" "$ucf_download_url"; then
  printf 'Download failed. The sharing cookies may not be sufficient; no ZIP was marked complete.\n' >&2
  exit 1
fi

file "$ucf_partial"
if ! zipinfo -t "$ucf_partial"; then
  printf 'ZIP directory validation failed. Do not extract or resume an HTML/error response.\n' >&2
  exit 1
fi
mv -n -- "$ucf_partial" "$ucf_complete"
if [ -e "$ucf_partial" ]; then
  printf 'Final destination appeared during download; partial file preserved: %s\n' "$ucf_partial" >&2
  exit 1
fi
printf 'Saved: %s\n' "$ucf_complete"
printf 'ZIP directory checked; full member CRC, contents and feature mapping remain unverified.\n'
