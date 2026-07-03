#!/bin/bash
# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# geniex-bench entry script for QDC Linux IoT (BASH framework).
#
# QDC extracts the artifact zip to /data/local/tmp/TestContent/ and runs this
# via /bin/bash. Anything under /data/local/tmp/QDC_logs/ is auto-uploaded.
# run_qdc_jobs.py substitutes:
#   {MODELS}  → `name|plugin|csv_devices|model_id|vlm|image` lines
#   {CHIPSET} → AI Hub chipset slug (e.g. qualcomm-qcs9075)
# Each cell's column-4 model_id is resolved on the device by the model-
# manager C API (multi-connection HTTPS, byte-range resume) on first
# reference; the cached copy is reused across the ctx sweep.
#
# We sweep ctx in {512, 1024, 4096} per cell to align with test-llama.cpp's
# PERFORMANCE SESSION. Two prefill modes coexist:
#   - llama_cpp cells use random-ids prefill (`-p N`, mirrors llama-bench
#     `pp{N}`), so reported pp is exactly the ctx value;
#   - qairt cells go through prompt_utf8 (the plugin doesn't accept
#     pre-tokenized input_ids — see issue #1008), with a pre-trimmed
#     `sample_prompt_${ctx}.txt` per ctx so prompt length is bounded.
# Each plugin gets its own per-ctx TSV so the two invocations don't mix.

set +e
umask 022

LOG=/data/local/tmp/QDC_logs
OUT=$LOG/results
MM_CACHE=/data/local/tmp/geniex-cache
TC=/data/local/tmp/TestContent
BUNDLE=$TC/pkg-geniex
PROMPTS=$TC/prompts

mkdir -p "$LOG" "$OUT" "$MM_CACHE"
exec > "$LOG/script.log" 2>&1
date -u
uname -a

cd "$BUNDLE" || { echo "FATAL: missing $BUNDLE"; exit 1; }
chmod +x bin/* 2>/dev/null
export LD_LIBRARY_PATH="$BUNDLE/lib:$BUNDLE/lib/llama_cpp:$BUNDLE/lib/qairt:$LD_LIBRARY_PATH"
export GENIEX_PLUGIN_PATH="$BUNDLE/lib"

IMG=$TC/test.png

for ctx in 512 1024 4096; do
  : > "/data/local/tmp/matrix-llama-${ctx}.tsv"
  : > "/data/local/tmp/matrix-qairt-${ctx}.tsv"
done

while IFS='|' read -r name plugin devs model_id vlm image; do
  [ -z "$name" ] && continue
  echo "=== plan $name id=$model_id ==="
  imgpath=""
  [ "$image" = "1" ] && imgpath="$IMG"
  case "$plugin" in
    qairt)     bucket=qairt ;;
    llama_cpp) bucket=llama ;;
    *) echo "WARN: unknown plugin $plugin in $name, skipping"; continue ;;
  esac
  IFS=','
  for d in $devs; do
    for ctx in 512 1024 4096; do
      # Columns 5/6 (tokenizer/mmproj) intentionally blank: the model
      # manager fills both from the resolved manifest.
      printf '%s-%s-%s-c%s\t%s\t%s\t%s\t\t\t%s\t%s\n' \
        "$name" "$plugin" "$d" "$ctx" "$plugin" "$d" "$model_id" "$imgpath" "$vlm" \
        >> "/data/local/tmp/matrix-${bucket}-${ctx}.tsv"
    done
  done
  IFS='|'
done <<'EOF'
{MODELS}
EOF

for ctx in 512 1024 4096; do
  llama_tsv="/data/local/tmp/matrix-llama-${ctx}.tsv"
  qairt_tsv="/data/local/tmp/matrix-qairt-${ctx}.tsv"

  if [ -s "$llama_tsv" ]; then
    echo "=== matrix llama_cpp ctx=$ctx (random-ids prefill) ==="
    cat "$llama_tsv"
    ./bin/geniex-bench --matrix-file "$llama_tsv" --output-json-dir "$OUT" -r 3 \
      -c "$ctx" -p "$ctx" \
      --mm-data-dir "$MM_CACHE" --chipset "{CHIPSET}"
    echo "rc=$?  ($(ls "$OUT" | wc -l) cell json files so far)"
  fi

  if [ -s "$qairt_tsv" ]; then
    echo "=== matrix qairt ctx=$ctx (prompt-file) ==="
    cat "$qairt_tsv"
    ./bin/geniex-bench --matrix-file "$qairt_tsv" --output-json-dir "$OUT" -r 3 \
      -c "$ctx" --prompt-file "$PROMPTS/sample_prompt_${ctx}.txt" \
      --mm-data-dir "$MM_CACHE" --chipset "{CHIPSET}"
    echo "rc=$?  ($(ls "$OUT" | wc -l) cell json files so far)"
  fi
done
echo "=== done ==="
exit 0
