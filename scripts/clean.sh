#!/usr/bin/env bash

set -e

BUILD_DIR="../build"

echo "==== Cleaning project ===="

if [ -d "$BUILD_DIR" ]; then
    echo "Removing build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    echo "Build directory removed."
else
    echo "No build directory found, nothing to remove."
fi

echo "==== Clean Done ===="
