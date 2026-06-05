#!/usr/bin/env bash
set -euo pipefail

python -B -m scripts.traceflow experiment \
  --suite configs/suites/paper_smoke.yml \
  --training-figures
