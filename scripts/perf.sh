#!/usr/bin/env bash

set -e

BIN_DIR="../build/bin"
PROFILE_DIR="../profiles"

mkdir -p "$PROFILE_DIR"

if [ ! -d "$BIN_DIR" ]; then
    echo "Error: $BIN_DIR not found. Please run build.sh first."
    exit 1
fi

CASE_NAME=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -case)
            CASE_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-case <executable_name>]"
            exit 1
            ;;
    esac
done

echo "==== Nsight Compute profiling (ncu) ===="

run_ncu () {
    local exe_path="$1"
    local exe_name
    exe_name=$(basename "$exe_path")

    echo "------------------------------------------"
    echo "Profiling: $exe_name"
    echo "------------------------------------------"

    # 每个可执行程序单独子文件夹
    local exe_profile_dir="$PROFILE_DIR/$exe_name"
    mkdir -p "$exe_profile_dir"

    local rep_file="$exe_profile_dir/${exe_name}.ncu-rep"
    local summary_file="$exe_profile_dir/${exe_name}_import_summary.txt"
    local output_file="$exe_profile_dir/${exe_name}_output.txt"

    # 1. ncu profiling，生成 .ncu-rep
    ncu -f \
        --set full \
        --export "$rep_file" \
        --print-summary per-gpu \
        "$exe_path" > /dev/null 2>&1

    echo "Generated ncu-rep: $rep_file"

    # 2. 读取 ncu-rep，生成 CLI summary
    ncu --import "$rep_file" --print-summary none > "$summary_file" 2>&1
    echo "Generated summary: $summary_file"

    # 3. 运行原程序，保留输出到文件
    "$exe_path" > "$output_file" 2>&1
    echo "Program output saved to: $output_file"
    echo
}

# 如果指定单个可执行文件
if [[ -n "$CASE_NAME" ]]; then
    TARGET="$BIN_DIR/$CASE_NAME"
    if [[ -f "$TARGET" && -x "$TARGET" ]]; then
        run_ncu "$TARGET"
    else
        echo "Error: executable '$CASE_NAME' not found in $BIN_DIR"
        exit 1
    fi
else
    # 执行所有可执行文件
    for file in "$BIN_DIR"/*; do
        if [[ -f "$file" && -x "$file" ]]; then
            run_ncu "$file"
        fi
    done
fi

echo "==== Profiling complete ===="
