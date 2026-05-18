#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_mac_portable.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$MAC_ATTENTION_ROOT"

"$PYTHON_BIN" -m pytest \
  tests/test_mac_persistent_decode.py \
  tests/test_sglang_q_preserve.py \
  tests/test_sglang_plugin_config.py \
  -q
