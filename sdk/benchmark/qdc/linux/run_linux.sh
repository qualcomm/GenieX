#!/bin/bash
# geniex_benchmark entry script for QDC Linux IoT (BASH framework).
#
# QDC extracts the artifact zip to /data/local/tmp/TestContent/ and runs this
# via /bin/bash. Anything under /data/local/tmp/QDC_logs/ is auto-uploaded.
# run_qdc_jobs.py substitutes {MODELS} with `name|plugin|csv_devices|url|kind`
# lines before upload.

set +e
umask 022

LOG=/data/local/tmp/QDC_logs
OUT=$LOG/results
MODELS=/data/local/tmp/models
BUNDLE=/data/local/tmp/TestContent/pkg-geniex
TSV=/data/local/tmp/matrix.tsv

mkdir -p "$LOG" "$OUT" "$MODELS"
exec > "$LOG/script.log" 2>&1
date -u
uname -a

cd "$BUNDLE" || { echo "FATAL: missing $BUNDLE"; exit 1; }
chmod +x bin/* 2>/dev/null
export LD_LIBRARY_PATH="$BUNDLE/lib:$BUNDLE/lib/llama_cpp:$BUNDLE/lib/qairt:$LD_LIBRARY_PATH"
export GENIEX_PLUGIN_PATH="$BUNDLE/lib"

IMG=/data/local/tmp/TestContent/test.png

: > "$TSV"
while IFS='|' read -r name plugin devs url kind mmproj_url vlm image; do
  [ -z "$name" ] && continue
  dir="$MODELS/$name"
  mkdir -p "$dir"
  echo "=== fetch $name ($kind) ==="
  if [ "$kind" = "bundle" ]; then
    mpath="$dir/bundle"
    if [ ! -d "$mpath" ]; then
      curl -L -fS --retry 3 --retry-delay 5 -o "$dir/b.zip" "$url" \
        && python3 -c "import zipfile,os,shutil
d='$mpath'
zipfile.ZipFile('$dir/b.zip').extractall(d)
e=os.listdir(d)
if len(e)==1 and os.path.isdir(os.path.join(d,e[0])):
    inner=os.path.join(d,e[0])
    for n in os.listdir(inner): shutil.move(os.path.join(inner,n),os.path.join(d,n))
    os.rmdir(inner)"
      if [ $? -ne 0 ]; then echo "WARNING: $name fetch failed, skipping"; rm -rf "$mpath"; continue; fi
    fi
  else
    mpath="$dir/model.gguf"
    if [ ! -f "$mpath" ]; then
      curl -L -fS --retry 3 --retry-delay 5 -o "$mpath" "$url"
      if [ $? -ne 0 ]; then echo "WARNING: $name fetch failed, skipping"; continue; fi
    fi
  fi
  mmpath=""
  if [ -n "$mmproj_url" ]; then
    mmpath="$dir/mmproj.gguf"
    if [ ! -f "$mmpath" ]; then
      curl -L -fS --retry 3 --retry-delay 5 -o "$mmpath" "$mmproj_url"
      if [ $? -ne 0 ]; then echo "WARNING: $name mmproj fetch failed, skipping"; continue; fi
    fi
  fi
  imgpath=""
  [ "$image" = "1" ] && imgpath="$IMG"
  IFS=','
  for d in $devs; do
    printf '%s-%s-%s\t%s\t%s\t%s\t\t%s\t%s\t%s\n' \
      "$name" "$plugin" "$d" "$plugin" "$d" "$mpath" "$mmpath" "$imgpath" "$vlm" >> "$TSV"
  done
  IFS='|'
done <<'EOF'
{MODELS}
EOF

echo "=== matrix ==="
cat "$TSV"
./bin/geniex_benchmark --matrix-file "$TSV" --output-json-dir "$OUT" -r 3
echo "rc=$?  ($(ls "$OUT" | wc -l) cell json files)"
echo "=== done ==="
exit 0
