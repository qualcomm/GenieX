#!/bin/bash
# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Self-contained installer for the GenieX-CLI on Linux.
# This script includes an embedded binary payload.
#
# Usage:
#   chmod +x ./install.sh
#   sudo ./install.sh              # Full installation
#   ./install.sh --extract-only     # Extract only (no installation)

set -eu

# --- Shell UI Helper Functions ---
red="$( (tput bold 2>/dev/null || :) && (tput setaf 1 2>/dev/null || :) )"
plain="$( tput sgr0 2>/dev/null || : )"

status() { echo ">>> $*" >&2; }
error() { echo "${red}ERROR:${plain} $*" >&2; exit 1; }
warning() { echo "${red}WARNING:${plain} $*" >&2; }

# --- Cleanup handler ---
# Ensures temporary files are removed on exit
TEMP_DIR=$(mktemp -d)
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# --- Global Variables ---
SUDO=""
GENIEX_INSTALL_DIR="/opt/geniex"
BINDIR="/usr/local/bin"
EXTRACT_ONLY=false

# --- Prerequisite and Environment Checks ---

# Checks if a command is available on the system
available() {
    command -v "$1" >/dev/null 2>&1
}

# Checks for a list of required tools and reports missing ones
require_tools() {
    local missing_tools=""
    for tool in "$@"; do
        if ! available "$tool"; then
            missing_tools="${missing_tools} ${tool}"
        fi
    done
    echo "$missing_tools"
}


# Sets up the SUDO variable if not running as root
setup_sudo() {
    if [ "$(id -u)" -ne 0 ]; then
        if ! available sudo; then
            error "This script requires superuser permissions, but 'sudo' is not available. Please re-run as root."
        fi
        SUDO="sudo"
        status "Using 'sudo' for privileged operations."
    fi
}

# Validates all system requirements before proceeding
validate_requirements() {
    local needs
    needs=$(require_tools tar cat awk)
    if [ -n "$needs" ]; then
        error "The following required tools are missing:$needs"
    fi
    status "All required tools are available."
}

# --- Core Installation Logic ---

# Extract-only function
extract_only() {
    local target_dir="./geniex"

    status "Starting GenieX-CLI extraction..."

    # --- 1. Locate and extract the embedded payload ---
    local payload_line
    payload_line=$(awk '/^__PAYLOAD_BELOW__/ {print NR + 1}' "$0")
    if [ -z "$payload_line" ]; then
        error "Could not find payload in the script. The installer appears to be corrupted."
    fi

    status "Creating extraction directory: $target_dir"
    mkdir -p "$target_dir"

    status "Extracting embedded payload to $target_dir..."
    tail -n "+$payload_line" "$0" | tar -xzf - -C "$target_dir"
    if [ $? -ne 0 ]; then
        error "Failed to extract payload. The installer might be corrupted or incomplete."
    fi

    # Make binaries executable
    chmod +x "$target_dir/geniex" "$target_dir/geniex-cli" 2>/dev/null || true

    # Get absolute path for display
    local abs_dir
    abs_dir=$(cd "$target_dir" && pwd)

    status "${plain}Extraction complete! Files extracted to: $abs_dir"
    echo ""
    status "To use the extracted binaries, add the following to your PATH:"
    echo "  export PATH=\"$abs_dir:\$PATH\""
    echo ""
    status "Or add it permanently to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
    echo "  echo 'export PATH=\"$abs_dir:\$PATH\"' >> ~/.bashrc"
    echo ""
    status "Then you can use 'geniex' commands directly."
}

# Main installation function
install_geniex() {
    status "Starting GenieX-CLI installation..."

    # --- 1. Locate and extract the embedded payload ---
    local payload_line
    payload_line=$(awk '/^__PAYLOAD_BELOW__/ {print NR + 1}' "$0")
    if [ -z "$payload_line" ]; then
        error "Could not find payload in the script. The installer appears to be corrupted."
    fi

    local temp_extract_dir
    temp_extract_dir=$(mktemp -d)

    status "Extracting embedded payload to a temporary directory..."
    tail -n "+$payload_line" "$0" | tar -xzf - -C "$temp_extract_dir"
    if [ $? -ne 0 ]; then
        rm -rf "$temp_extract_dir"
        error "Failed to extract payload. The installer might be corrupted or incomplete."
    fi

    # --- 2. Clean up previous installations ---
    if [ -d "$GENIEX_INSTALL_DIR" ]; then
        status "Removing existing installation at $GENIEX_INSTALL_DIR"
        $SUDO rm -rf "$GENIEX_INSTALL_DIR"
    fi

    # --- 3. Install new files ---
    status "Creating installation directory: $GENIEX_INSTALL_DIR"
    $SUDO install -o root -g root -m 755 -d "$GENIEX_INSTALL_DIR"

    status "Installing GenieX-CLI files..."
    $SUDO mv "$temp_extract_dir"/* "$GENIEX_INSTALL_DIR/"
    $SUDO chmod a+x "$GENIEX_INSTALL_DIR/geniex" "$GENIEX_INSTALL_DIR/geniex-cli"

    # --- 4. Create symbolic links ---
    status "Creating symbolic links in $BINDIR..."
    $SUDO mkdir -p "$BINDIR"
    $SUDO ln -sf "$GENIEX_INSTALL_DIR/geniex" "$BINDIR/geniex"

    # --- 5. Clean up ---
    rm -rf "$temp_extract_dir"
    status "GenieX-CLI files installed successfully."
}


# Parse command line arguments
parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --extract-only)
                EXTRACT_ONLY=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --extract-only    Extract files only, do not install (extracts to ./geniex)"
                echo "  -h, --help        Show this help message"
                echo ""
                echo "Examples:"
                echo "  sudo $0              # Full installation to /opt/geniex"
                echo "  $0 --extract-only     # Extract to ./geniex"
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use -h or --help for usage information."
                ;;
        esac
    done
}

# Main function to orchestrate the installation
main() {
    if [ "$(uname -s)" != "Linux" ]; then
        error "This script is intended to run on Linux only."
    fi

    parse_args "$@"

    if [ "$EXTRACT_ONLY" = true ]; then
        validate_requirements
        extract_only
    else
        status "Starting GenieX-CLI installer..."

        setup_sudo
        validate_requirements

        install_geniex

        status "${plain}Install complete! The GenieX-CLI is now installed."
        status "You can use the 'geniex' commands from your terminal."
    fi

    # warning for missing libgomp1
    warning "libgomp1 is required for GenieX-CLI to function properly. make sure it is installed on your system."
    warning "You can install it using your package manager, e.g., 'sudo apt-get install libgomp1' on Debian-based systems."
}

# Run the main function with all arguments passed to the script
main "$@"

# --- IMPORTANT ---
# The script MUST exit before the payload marker.
# The CI/CD process will append the base64 encoded payload below this line.
exit 0
__PAYLOAD_BELOW__
