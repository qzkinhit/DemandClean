#!/usr/bin/env bash
# Recompute the two-player attribution of the twelve canonical deployed outputs.
# Usage: bash results/attribution_operation/attr_op_all.sh [output-directory]
set -euo pipefail
ATTR_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ATTR_SOURCE_DIR/../.." && pwd)"
ATTR_OUTPUT_DIR="${1:-$REPO_ROOT/results/runs/attribution_operation}"
DC_PYTHON="${DC_PYTHON:-python3}"
mkdir -p "$ATTR_OUTPUT_DIR"
ATTR_OUTPUT_DIR="$(cd "$ATTR_OUTPUT_DIR" && pwd)"
if [[ "$ATTR_OUTPUT_DIR" == "$ATTR_SOURCE_DIR" ]]; then
  printf '%s\n' 'Choose a separate output directory for recomputed results.' >&2
  exit 2
fi
ATTR_TEMP="$(mktemp "$ATTR_OUTPUT_DIR/.rows.XXXXXX")"
trap 'rm -f "$ATTR_TEMP"' EXIT
for DATASET in adult beers breast_cancer smartfactory bike mercedes nasa soilmoisture har flights soccer hospitals; do
  "$DC_PYTHON" "$ATTR_SOURCE_DIR/attr_op.py" \
    "$REPO_ROOT/results/main_table/$DATASET/demandclean" >> "$ATTR_TEMP"
done
mv "$ATTR_TEMP" "$ATTR_OUTPUT_DIR/rows.jsonl"
trap - EXIT
printf '%s\n' "Saved twelve attribution records to $ATTR_OUTPUT_DIR/rows.jsonl"
