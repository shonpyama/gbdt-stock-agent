#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SYMBOLS="${1:-AAPL,MSFT,NVDA,AMZN,META}"
MODE="${2:-quick}"  # quick | improve

echo "GBDT-only mode"
echo "symbols: $SYMBOLS"
echo "mode: $MODE"
echo "note: This entrypoint uses only GBDT scripts (no GNN training)."

case "$MODE" in
  quick)
    ./run_all_local.sh "$SYMBOLS"
    ;;
  improve)
    ./auto_improve_loop.sh "$SYMBOLS"
    ;;
  *)
    echo "ERROR: mode must be 'quick' or 'improve'" >&2
    exit 1
    ;;
esac

if [[ -x ./save_workspace_checkpoint.sh ]]; then
  ./save_workspace_checkpoint.sh "gbdt_only_${MODE}"
fi
