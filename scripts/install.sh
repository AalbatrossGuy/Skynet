#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-https://github.com/AalbatrossGuy/Skynet.git}"
BRANCH="${BRANCH:-main}"
# DIR="${DIR:-$HOME/.local/share/Skynet}"
DIR="${DIR:-$HOME/Skynet}"


need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[-] '$1' is required but not found in PATH." >&2
    exit 1
  }
}

info(){ echo "[*] $*"; }
ok(){ echo "✅ $*"; }


need git
need python3

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


if [[ ! -x ".venv/bin/python" ]]; then
  info "Creating virtual environment ..."
  python3 -m venv .venv
fi

source .venv/bin/activate

info "Upgrading pip ..."
python -m pip install --upgrade pip

info "Installing requirements ..."
pip install -r requirements.txt


chmod +x scripts/skynet.sh


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

