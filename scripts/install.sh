#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config (override via env)
# -----------------------------
REPO="${REPO:-https://github.com/AalbatrossGuy/Skynet.git}"
BRANCH="${BRANCH:-main}"
# DIR="${DIR:-$HOME/.local/share/federated_secure_agg}"
DIR="${DIR:-$HOME/Skynet}"

# These envs are forwarded to fedctl.sh automatically if you export them
# ROUNDS, MIN_CLIENTS, CLIENTS, CLIENT_SAMPLES, CLIENT_ROUNDS, CLIENT_LR

# -----------------------------
# Helpers
# -----------------------------
need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[-] '$1' is required but not found in PATH." >&2
    exit 1
  }
}

info(){ echo "[*] $*"; }
ok(){ echo "✅ $*"; }

# -----------------------------
# Prerequisite checks
# -----------------------------
need git
need python3
# pip might be 'pip3' on some systems; we rely on venv's pip anyway.

# -----------------------------
# Fetch or update repo
# -----------------------------
info "Installing to: $DIR"
if [[ -d "$DIR/.git" ]]; then
  info "Repo exists. Pulling updates from $BRANCH ..."
  git -C "$DIR" fetch --depth=1 origin "$BRANCH"
  git -C "$DIR" checkout -q "$BRANCH"
  git -C "$DIR" pull --ff-only origin "$BRANCH"
else
  info "Cloning $REPO ($BRANCH) ..."
  git clone --depth=1 --branch "$BRANCH" "$REPO" "$DIR"
fi

cd "$DIR"

# -----------------------------
# Python venv & deps
# -----------------------------
if [[ ! -x ".venv/bin/python" ]]; then
  info "Creating virtual environment ..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

info "Upgrading pip ..."
python -m pip install --upgrade pip

info "Installing requirements ..."
pip install -r requirements.txt

# -----------------------------
# Run controller
# -----------------------------
chmod +x scripts/skynet.sh
# info "Starting stack via scripts/skynet.sh start ..."
# ./scripts/skynet.sh start

ok "Install complete."

cat <<'EOS'

Next steps:
  - Use './scripts/skynet.sh status' to check processes
  - Use './scripts/skynet.sh logs'   to tail logs
  - Use './scripts/skynet.sh stop'   to stop everything

Customize the run by exporting environment variables BEFORE running:
  export CLIENTS="A B C"
  export ROUNDS=8
  export MIN_CLIENTS=3
  export CLIENT_SAMPLES=400
  export CLIENT_ROUNDS=6
  export CLIENT_LR=0.4

Run this command to start Skynet:
  ./scripts/skynet.sh start
EOS

