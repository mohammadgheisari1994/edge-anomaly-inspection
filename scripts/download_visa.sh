#!/usr/bin/env bash
# Downloads VisA (Amazon Science, CC BY 4.0; large download) into data/VisA.
set -euo pipefail
DEST="${1:-data/VisA}"
URL="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"

if [ -d "$DEST" ] && [ -n "$(find "$DEST" -name 1cls.csv -print -quit)" ]; then
  echo "VisA already present in $DEST"; exit 0
fi
mkdir -p "$DEST"
echo "Downloading VisA (large, may take a while)..."
curl -L --fail -C - -o "$DEST/VisA.tar" "$URL"
echo "Extracting..."
tar -xf "$DEST/VisA.tar" -C "$DEST"
rm "$DEST/VisA.tar"
echo "Done: $DEST"
