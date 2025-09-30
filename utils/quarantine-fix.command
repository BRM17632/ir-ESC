#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Removing quarantine attribute recursively in folder:"
echo "$SCRIPT_DIR"

xattr -r -d com.apple.quarantine "$SCRIPT_DIR"

echo "Done! You can now close this window and try running your executable."
read -p "Press Enter to close this window..."
