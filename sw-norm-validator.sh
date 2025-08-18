#!/bin/bash

VENV_DIR="venv"
APP_FILE="app.py"
REQUIREMENTS_FILE="requirements.txt"

# ───────────────────────────────
# 🎨 Color Functions
info()    { echo -e "\033[1;34m[INFO]\033[0m $1"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m $1"; }
err()     { echo -e "\033[1;31m[ERROR]\033[0m $1"; }
success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }

# ⏳ Spinner (static line)
spinner() {
  local pid=$!
  local delay=0.1
  local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
  local i=0
  while kill -0 $pid 2>/dev/null; do
    i=$(( (i+1) % 10 ))
    printf "\r\033[1;34m[INFO]\033[0m Installing dependencies... ${spinstr:$i:1}"
    sleep $delay
  done
  printf "\r\033[1;32m[SUCCESS]\033[0m Dependencies installed.     \n"
}

# ───────────────────────────────
# ✅ Check Python
if ! command -v python3 &>/dev/null; then
  err "Python3 not found. Please install Python 3."
  exit 1
fi

# 🧪 Handle venv
if [ ! -d "$VENV_DIR" ]; then
  info "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
  if [ $? -eq 0 ]; then
    success "Virtual environment created at '$VENV_DIR'."
  else
    err "Failed to create virtual environment."
    exit 1
  fi
else
  info "Using existing virtual environment at '$VENV_DIR'."
fi

# 🧬 Activate venv
source "$VENV_DIR/bin/activate"

# 📦 Install dependencies (with spinner)
(pip install --upgrade pip &>/dev/null && pip install -r "$REQUIREMENTS_FILE" &>/dev/null) &
spinner

# 🔄 Git update check
if command -v git &>/dev/null; then
  info "Checking for updates from Git..."
  git fetch &>/dev/null
  LOCAL=$(git rev-parse @)
  REMOTE=$(git rev-parse @{u})
  if [ "$LOCAL" != "$REMOTE" ]; then
    read -p $'\033[1;33m[UPDATE]\033[0m New version available. Pull now? (y/n): ' yn
    if [ "$yn" == "y" ]; then
      git pull
      success "Project updated."
    else
      warn "Update skipped."
    fi
  else
    success "Project is up to date."
  fi
else
  warn "Git not found. Skipping update check."
fi

# 🚀 Launch Streamlit
info "Launching Streamlit app..."
streamlit run "$APP_FILE"