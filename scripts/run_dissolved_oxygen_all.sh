#!/usr/bin/env bash
# Run scripts/run_dissolved_oxygen.py for every (group, WY, SLR) combination
# available under /Users/dillonragar/data/cpra/AMP_INPUT for G400 and G900.
# Each invocation is written out explicitly so the args are easy to review.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_dissolved_oxygen.py"

failed=()

run_one() {
  local label="$1"; shift
  echo
  echo "========================================================"
  echo "  ${label}"
  echo "========================================================"
  if python "${RUNNER}" "$@"; then
    echo "  OK: ${label}"
  else
    echo "  FAILED: ${label}" >&2
    failed+=("${label}")
  fi
}

# --- G400 ---------------------------------------------------------------
run_one "G400 WY06 SLR000" \
  --data-dir /Users/dillonragar/data/cpra \
  --group G400 --wy 06 --slr 000 \
  --input-version V2 --output-version V4

run_one "G400 WY06 SLR328" \
  --data-dir /Users/dillonragar/data/cpra \
  --group G400 --wy 06 --slr 328 \
  --input-version V2 --output-version V4

run_one "G400 WY20 SLR000" \
  --data-dir /Users/dillonragar/data/cpra \
  --group G400 --wy 20 --slr 000 \
  --input-version V2 --output-version V4

run_one "G400 WY20 SLR328" \
  --data-dir /Users/dillonragar/data/cpra \
  --group G400 --wy 20 --slr 328 \
  --input-version V2 --output-version V4

run_one "G400 WY22 SLR000" \
  --data-dir /Users/dillonragar/data/cpra \
  --group G400 --wy 22 --slr 000 \
  --input-version V2 --output-version V4

run_one "G400 WY22 SLR328" \
  --data-dir /Users/dillonragar/data/cpra \
  --group G400 --wy 22 --slr 328 \
  --input-version V2 --output-version V4

# --- G900 ---------------------------------------------------------------
run_one "G900 WY06 SLR000" \
  --data-dir /Users/dillonragar/data/cpra \
  --group G900 --wy 06 --slr 000 \
  --input-version V2 --output-version V4

run_one "G900 WY06 SLR328" \
  --data-dir /Users/dillonragar/data/cpra \
  --group G900 --wy 06 --slr 328 \
  --input-version V2 --output-version V4

run_one "G900 WY20 SLR000" \
  --data-dir /Users/dillonragar/data/cpra \
  --group G900 --wy 20 --slr 000 \
  --input-version V2 --output-version V4

run_one "G900 WY20 SLR328" \
  --data-dir /Users/dillonragar/data/cpra \
  --group G900 --wy 20 --slr 328 \
  --input-version V2 --output-version V4

run_one "G900 WY22 SLR000" \
  --data-dir /Users/dillonragar/data/cpra \
  --group G900 --wy 22 --slr 000 \
  --input-version V2 --output-version V4

run_one "G900 WY22 SLR328" \
  --data-dir /Users/dillonragar/data/cpra \
  --group G900 --wy 22 --slr 328 \
  --input-version V2 --output-version V4

# --- Summary ------------------------------------------------------------
echo
if ((${#failed[@]} == 0)); then
  echo "All 12 combinations completed successfully."
else
  echo "Completed with ${#failed[@]} failure(s):" >&2
  printf '  - %s\n' "${failed[@]}" >&2
  exit 1
fi
