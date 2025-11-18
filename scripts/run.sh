#!/usr/bin/env bash

set -e

BIN_DIR="../build/bin"

if [ ! -d "$BIN_DIR" ]; then
    echo "Error: $BIN_DIR not found. Please run build.sh first."
    exit 1
fi

echo "==== Running all executables in $BIN_DIR ===="

for file in "$BIN_DIR"/*; do
    if [ -f "$file" ] && [ -x "$file" ]; then
        echo "------------------------------------------"
        echo "Running: $(basename "$file")"
        echo "------------------------------------------"
        "$file"
        echo
    fi
done

echo "==== All programs executed ===="
