#!/usr/bin/env bash
# Start the CFA Streamlit study app. Extra args are passed to streamlit,
# e.g. ./run_app.sh --server.port 8502
set -euo pipefail
cd "$(dirname "$0")"
exec uv run streamlit run app.py "$@"
