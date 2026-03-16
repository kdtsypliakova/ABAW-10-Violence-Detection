#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash test_set_processing/run_all.sh \
#     --models-suite test_set_processing/models_suite.example.json \
#     --test-frames-root /path/to/DVD/Test/frames

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOTEBOOK_REL="exp_backbones_v2.ipynb"
OUT_REL="test_set_processing/submissions_generated"
TEMPLATES_REL="__NO_TEMPLATES__"
DEVICE="cuda"
ZIP_FLAG="--zip"

python3 "$PROJECT_ROOT/test_set_processing/generate_submissions.py" \
  --project-root "$PROJECT_ROOT" \
  --notebook "$NOTEBOOK_REL" \
  --out-root "$OUT_REL" \
  --templates-root "$TEMPLATES_REL" \
  --device "$DEVICE" \
  $ZIP_FLAG \
  "$@"
