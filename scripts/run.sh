#!/usr/bin/env bash

set -e

BIN_DIR="../build/bin"

if [ ! -d "$BIN_DIR" ]; then
    echo "Error: $BIN_DIR not found. Please run build.sh first."
    exit 1
fi

# 默认值：空，表示运行所有
CASE_NAME=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -case)
            CASE_NAME="$2"
            shift # past argument
            shift # past value
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-case <executable_name>]"
            exit 1
            ;;
    esac
done

echo "==== Running executables in $BIN_DIR ===="

if [[ -n "$CASE_NAME" ]]; then
    # 指定运行某个可执行文件
    TARGET="$BIN_DIR/$CASE_NAME"
    if [[ -f "$TARGET" && -x "$TARGET" ]]; then
        echo "------------------------------------------"
        echo "Running: $CASE_NAME"
        echo "------------------------------------------"
        "$TARGET"
        echo "==== Done ===="
    else
        echo "Error: executable '$CASE_NAME' not found in $BIN_DIR"
        exit 1
    fi
else
    # 运行所有可执行文件
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
fi
