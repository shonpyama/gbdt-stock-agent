#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SKIP_SMOKE=0
SYMBOLS="AAPL,MSFT,NVDA,AMZN,META"

for arg in "$@"; do
  case "$arg" in
    --skip-smoke) SKIP_SMOKE=1 ;;
    --symbols=*) SYMBOLS="${arg#--symbols=}" ;;
  esac
done

load_api_key() {
  if [[ -n "${FMP_API_KEY:-}" ]]; then
    if [[ "${FMP_API_KEY}" == FMP_API_KEY=* ]]; then
      printf "%s" "${FMP_API_KEY#FMP_API_KEY=}" | sed "s/^['\"]//;s/['\"]$//"
    else
      printf "%s" "${FMP_API_KEY}"
    fi
    return 0
  fi
  if [[ -f /content/.env_fmp ]]; then
    while IFS= read -r raw || [[ -n "$raw" ]]; do
      line="$(printf "%s" "$raw" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
      [[ -z "$line" || "$line" == \#* ]] && continue
      if [[ "$line" == FMP_API_KEY=* ]]; then
        printf "%s" "${line#FMP_API_KEY=}" | sed "s/^['\"]//;s/['\"]$//"
        return 0
      fi
      if [[ "$line" != *=* ]]; then
        printf "%s" "$line"
        return 0
      fi
    done < /content/.env_fmp
  fi
  return 1
}

echo "[setup] Installing dependencies from requirements.txt"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if KEY="$(load_api_key)"; then
  export FMP_API_KEY="$KEY"
fi

if [[ -z "${FMP_API_KEY:-}" ]]; then
  echo "[setup] ERROR: FMP_API_KEY not found. Set env var or /content/.env_fmp." >&2
  exit 1
fi
echo "[setup] API key loaded."

if [[ "$SKIP_SMOKE" -eq 1 ]]; then
  echo "[setup] Skip smoke test (--skip-smoke)."
  exit 0
fi

echo "[setup] Running smoke test with symbols: $SYMBOLS"
./run_all_local.sh "$SYMBOLS"
echo "[setup] Completed."
