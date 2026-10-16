#!/usr/bin/env bash
# Run scripts/run_dissolved_oxygen.py for every (group, WY, SLR) combination
# available under /Users/dillonragar/data/cpra/AMP_INPUT for G400 and G900.
# Each invocation is written out explicitly so the args are easy to review.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_dissolved_oxygen.py"

failed=()
ran=0

run_one() {
  local label="$1"; shift
  echo
  echo "========================================================"
  echo "  ${label}"
  echo "========================================================"
  # -u keeps stdout unbuffered: on a SIGKILL the buffer is never flushed, so a
  # buffered run loses exactly the lines that say where it died.
  local rc=0
  uv run python -u "${RUNNER}" "$@" || rc=$?
  if ((rc == 0)); then
    ran=$((ran + 1))
    echo "  OK: ${label}"
  else
    local detail=""
    if ((rc == 137)); then
      detail=" — SIGKILL, almost certainly the OOM killer."
      detail+=" Confirm with: dmesg -T | grep -i 'killed process'"
    elif ((rc > 128)); then
      detail=" — killed by signal $((rc - 128))"
    fi
    echo "  FAILED: ${label} (exit ${rc})${detail}" >&2
    ran=$((ran + 1))
    failed+=("${label} (exit ${rc})")
  fi
}


# # ------ G085 -----------------------------------------------------------
# run_one "G085 WY06 SLR000" \
#   --data-dir '/Volumes/dmr-disk/cpra/G085' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G085 --wy 06 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G085 WY06 SLR328" \
#   --data-dir '/Volumes/dmr-disk/cpra/G085' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G085 --wy 06 --slr 328 \
#   --input-version V2 --output-version V4

# run_one "G085 WY20 SLR000" \
#   --data-dir '/Volumes/dmr-disk/cpra/G085' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G085 --wy 20 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G085 WY20 SLR328" \
#   --data-dir '/Volumes/dmr-disk/cpra/G085' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G085 --wy 20 --slr 328 \
#   --input-version V2 --output-version V4

# run_one "G085 WY22 SLR000" \
#   --data-dir '/Volumes/dmr-disk/cpra/G085' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G085 --wy 22 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G085 WY22 SLR328" \
#   --data-dir '/Volumes/dmr-disk/cpra/G085' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G085 --wy 22 --slr 328 \
#   --input-version V2 --output-version V4


# ------ G401 -----------------------------------------------------------
# run_one "G401 WY06 SLR000" \
#   --data-dir '/Volumes/dmr-disk/cpra/G401' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G401 --wy 06 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G401 WY06 SLR328" \
#   --data-dir '/Volumes/dmr-disk/cpra/G401' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G401 --wy 06 --slr 328 \
#   --input-version V2 --output-version V4

# run_one "G401 WY20 SLR000" \
#   --data-dir '/Volumes/dmr-disk/cpra/G401' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G401 --wy 20 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G401 WY20 SLR328" \
#   --data-dir '/Volumes/dmr-disk/cpra/G401' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G401 --wy 20 --slr 328 \
#   --input-version V2 --output-version V4

# run_one "G401 WY22 SLR000" \
#   --data-dir '/Volumes/dmr-disk/cpra/G401' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G401 --wy 22 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G401 WY22 SLR328" \
#   --data-dir '/Volumes/dmr-disk/cpra/G401' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G401 --wy 22 --slr 328 \
#   --input-version V2 --output-version V4


# ------ G407 -----------------------------------------------------------
run_one "G407 WY06 SLR000" \
  --data-dir '/Volumes/dmr-disk/cpra/G407' \
  --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
  --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
  --group G407 --wy 06 --slr 000 \
  --input-version V2 --output-version V4

run_one "G407 WY06 SLR328" \
  --data-dir '/Volumes/dmr-disk/cpra/G407' \
  --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
  --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
  --group G407 --wy 06 --slr 328 \
  --input-version V2 --output-version V4

run_one "G407 WY20 SLR000" \
  --data-dir '/Volumes/dmr-disk/cpra/G407' \
  --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
  --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
  --group G407 --wy 20 --slr 000 \
  --input-version V2 --output-version V4

run_one "G407 WY20 SLR328" \
  --data-dir '/Volumes/dmr-disk/cpra/G407' \
  --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
  --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
  --group G407 --wy 20 --slr 328 \
  --input-version V2 --output-version V4

run_one "G407 WY22 SLR000" \
  --data-dir '/Volumes/dmr-disk/cpra/G407' \
  --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
  --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
  --group G407 --wy 22 --slr 000 \
  --input-version V2 --output-version V4

run_one "G407 WY22 SLR328" \
  --data-dir '/Volumes/dmr-disk/cpra/G407' \
  --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
  --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
  --group G407 --wy 22 --slr 328 \
  --input-version V2 --output-version V4


# # ------ G408 -----------------------------------------------------------
# run_one "G408 WY06 SLR000" \
#   --data-dir '/Volumes/dmr-disk/cpra/G408' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G408 --wy 06 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G408 WY06 SLR328" \
#   --data-dir '/Volumes/dmr-disk/cpra/G408' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G408 --wy 06 --slr 328 \
#   --input-version V2 --output-version V4

# run_one "G408 WY20 SLR000" \
#   --data-dir '/Volumes/dmr-disk/cpra/G408' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G408 --wy 20 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G408 WY20 SLR328" \
#   --data-dir '/Volumes/dmr-disk/cpra/G408' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G408 --wy 20 --slr 328 \
#   --input-version V2 --output-version V4

# run_one "G408 WY22 SLR000" \
#   --data-dir '/Volumes/dmr-disk/cpra/G408' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G408 --wy 22 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G408 WY22 SLR328" \
#   --data-dir '/Volumes/dmr-disk/cpra/G408' \
#   --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
#   --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
#   --group G408 --wy 22 --slr 328 \
#   --input-version V2 --output-version V4

# ------ G411 -----------------------------------------------------------
run_one "G411 WY06 SLR000" \
  --data-dir '/Volumes/dmr-disk/cpra/G411' \
  --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
  --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
  --group G411 --wy 06 --slr 000 \
  --input-version V2 --output-version V4

run_one "G411 WY06 SLR328" \
  --data-dir '/Volumes/dmr-disk/cpra/G411' \
  --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
  --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
  --group G411 --wy 06 --slr 328 \
  --input-version V2 --output-version V4

run_one "G411 WY20 SLR000" \
  --data-dir '/Volumes/dmr-disk/cpra/G411' \
  --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
  --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
  --group G411 --wy 20 --slr 000 \
  --input-version V2 --output-version V4

run_one "G411 WY20 SLR328" \
  --data-dir '/Volumes/dmr-disk/cpra/G411' \
  --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
  --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
  --group G411 --wy 20 --slr 328 \
  --input-version V2 --output-version V4

run_one "G411 WY22 SLR000" \
  --data-dir '/Volumes/dmr-disk/cpra/G411' \
  --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
  --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
  --group G411 --wy 22 --slr 000 \
  --input-version V2 --output-version V4

run_one "G411 WY22 SLR328" \
  --data-dir '/Volumes/dmr-disk/cpra/G411' \
  --dem-path /Users/dillonragar/data/cpra/60m_dem_1280_3200_padded.tif \
  --domain-path /Users/dillonragar/data/cpra/D3D_model_domain.tif \
  --group G411 --wy 22 --slr 328 \
  --input-version V2 --output-version V4


# # ------ G414 -----------------------------------------------------------
# run_one "G414 WY06 SLR000" \
#   --data-dir /home/dillon/cpra/AMP_INPUT/ \
#   --model-path /home/dillon/cpra/run_resources/xgb_dissolved_oxygen.json \
#   --device cuda \
#   --group G414 --wy 06 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G414 WY06 SLR328" \
#   --data-dir /home/dillon/cpra/AMP_INPUT/ \
#   --model-path /home/dillon/cpra/run_resources/xgb_dissolved_oxygen.json \
#   --device cuda \
#   --group G414 --wy 06 --slr 328 \
#   --input-version V2 --output-version V4

# run_one "G414 WY20 SLR000" \
#   --data-dir /home/dillon/cpra/AMP_INPUT/ \
#   --model-path /home/dillon/cpra/run_resources/xgb_dissolved_oxygen.json \
#   --device cuda \
#   --group G414 --wy 20 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G414 WY20 SLR328" \
#   --data-dir /home/dillon/cpra/AMP_INPUT/ \
#   --model-path /home/dillon/cpra/run_resources/xgb_dissolved_oxygen.json \
#   --device cuda \
#   --group G414 --wy 20 --slr 328 \
#   --input-version V2 --output-version V4

# run_one "G414 WY22 SLR000" \
#   --data-dir /home/dillon/cpra/AMP_INPUT/ \
#   --model-path /home/dillon/cpra/run_resources/xgb_dissolved_oxygen.json \
#   --device cuda \
#   --group G414 --wy 22 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G414 WY22 SLR328" \
#   --data-dir /home/dillon/cpra/AMP_INPUT/ \
#   --model-path /home/dillon/cpra/run_resources/xgb_dissolved_oxygen.json \
#   --device cuda \
#   --group G414 --wy 22 --slr 328 \
#   --input-version V2 --output-version V4


# ------ G417 -----------------------------------------------------------
run_one "G417 WY06 SLR000" \
  --data-dir '/Volumes/dmr-disk/cpra/G417' \
  --model-path /home/dillon/cpra/run_resources/xgb_dissolved_oxygen.json \
  --device cuda \
  --group G417 --wy 06 --slr 000 \
  --input-version V2 --output-version V4

run_one "G417 WY06 SLR328" \
  --data-dir '/Volumes/dmr-disk/cpra/G417' \
  --model-path /home/dillon/cpra/run_resources/xgb_dissolved_oxygen.json \
  --device cuda \
  --group G417 --wy 06 --slr 328 \
  --input-version V2 --output-version V4

run_one "G417 WY20 SLR000" \
  --data-dir '/Volumes/dmr-disk/cpra/G417' \
  --model-path /home/dillon/cpra/run_resources/xgb_dissolved_oxygen.json \
  --device cuda \
  --group G417 --wy 20 --slr 000 \
  --input-version V2 --output-version V4

run_one "G417 WY20 SLR328" \
  --data-dir '/Volumes/dmr-disk/cpra/G417' \
  --model-path /home/dillon/cpra/run_resources/xgb_dissolved_oxygen.json \
  --device cuda \
  --group G417 --wy 20 --slr 328 \
  --input-version V2 --output-version V4

run_one "G417 WY22 SLR000" \
  --data-dir '/Volumes/dmr-disk/cpra/G417' \
  --model-path /home/dillon/cpra/run_resources/xgb_dissolved_oxygen.json \
  --device cuda \
  --group G417 --wy 22 --slr 000 \
  --input-version V2 --output-version V4

run_one "G417 WY22 SLR328" \
  --data-dir '/Volumes/dmr-disk/cpra/G417' \
  --model-path /home/dillon/cpra/run_resources/xgb_dissolved_oxygen.json \
  --device cuda \
  --group G417 --wy 22 --slr 328 \
  --input-version V2 --output-version V4



# # --- G400 ---------------------------------------------------------------
# run_one "G400 WY06 SLR000" \
#   --data-dir /Users/dillonragar/data/cpra \
#   --group G400 --wy 06 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G400 WY06 SLR328" \
#   --data-dir /Users/dillonragar/data/cpra \
#   --group G400 --wy 06 --slr 328 \
#   --input-version V2 --output-version V4

# run_one "G400 WY20 SLR000" \
#   --data-dir /Users/dillonragar/data/cpra \
#   --group G400 --wy 20 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G400 WY20 SLR328" \
#   --data-dir /Users/dillonragar/data/cpra \
#   --group G400 --wy 20 --slr 328 \
#   --input-version V2 --output-version V4

# run_one "G400 WY22 SLR000" \
#   --data-dir /Users/dillonragar/data/cpra \
#   --group G400 --wy 22 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G400 WY22 SLR328" \
#   --data-dir /Users/dillonragar/data/cpra \
#   --group G400 --wy 22 --slr 328 \
#   --input-version V2 --output-version V4

# # --- G900 ---------------------------------------------------------------
# run_one "G900 WY06 SLR000" \
#   --data-dir /Users/dillonragar/data/cpra \
#   --group G900 --wy 06 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G900 WY06 SLR328" \
#   --data-dir /Users/dillonragar/data/cpra \
#   --group G900 --wy 06 --slr 328 \
#   --input-version V2 --output-version V4

# run_one "G900 WY20 SLR000" \
#   --data-dir /Users/dillonragar/data/cpra \
#   --group G900 --wy 20 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G900 WY20 SLR328" \
#   --data-dir /Users/dillonragar/data/cpra \
#   --group G900 --wy 20 --slr 328 \
#   --input-version V2 --output-version V4

# run_one "G900 WY22 SLR000" \
#   --data-dir /Users/dillonragar/data/cpra \
#   --group G900 --wy 22 --slr 000 \
#   --input-version V2 --output-version V4

# run_one "G900 WY22 SLR328" \
#   --data-dir /Users/dillonragar/data/cpra \
#   --group G900 --wy 22 --slr 328 \
#   --input-version V2 --output-version V4

# --- Summary ------------------------------------------------------------
echo
if ((ran == 0)); then
  echo "No combinations ran: every run_one call is commented out." >&2
  exit 1
elif ((${#failed[@]} == 0)); then
  echo "All ${ran} combination(s) completed successfully."
else
  echo "Completed ${ran} combination(s) with ${#failed[@]} failure(s):" >&2
  printf '  - %s\n' "${failed[@]}" >&2
  exit 1
fi
