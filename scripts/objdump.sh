#!/usr/bin/env bash

set -e  # exit on error

# =========================================
# dump_sass.sh
# 反汇编 build/bin/ 下的所有 CUDA 可执行文件
# 支持参数:  -case <program>  -mode <sass|ptx|both>
# 每个程序生成独立目录
# =========================================

BUILD_BIN_DIR="../build/bin"
OUT_DIR="../objdump"

# ========== 解析参数 ==========
TARGET_CASE=""
MODE="both"  # 默认输出 SASS + PTX
while [[ $# -gt 0 ]]; do
    case $1 in
        -case)
            TARGET_CASE="$2"
            shift 2
            ;;
        -mode)
            MODE="$2"
            if [[ "$MODE" != "sass" && "$MODE" != "ptx" && "$MODE" != "both" ]]; then
                echo "[Error] Invalid mode: $MODE"
                echo "Usage: $0 [-case <program>] [-mode <sass|ptx|both>]"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [-case <program>] [-mode <sass|ptx|both>]"
            exit 1
            ;;
    esac
done

# ========== 检查 cuobjdump ==========
if ! command -v cuobjdump &> /dev/null; then
    echo "[Error] cuobjdump not found. Please install CUDA toolkit."
    exit 1
fi

# ========== 输出目录 ==========
mkdir -p "$OUT_DIR"

echo "==== CUDA Dump Script ===="

# ========== 获取反汇编目标列表 ==========
if [ -z "$TARGET_CASE" ]; then
    echo "[1/3] Scanning $BUILD_BIN_DIR for executables..."
    EXEC_LIST=($(find "$BUILD_BIN_DIR" -maxdepth 1 -type f -executable))
else
    EXEC_LIST=("$BUILD_BIN_DIR/$TARGET_CASE")
    if [ ! -f "${EXEC_LIST[0]}" ]; then
        echo "[Error] Program not found: $BUILD_BIN_DIR/$TARGET_CASE"
        exit 1
    fi
    echo "[1/3] Selected program via -case: ${EXEC_LIST[0]}"
fi

# ========== 反汇编所有目标 ==========
echo "[2/3] Dumping into $OUT_DIR/"

for exe in "${EXEC_LIST[@]}"; do
    name=$(basename "$exe")
    exe_out_dir="$OUT_DIR/$name"
    mkdir -p "$exe_out_dir"

    echo "   → Processing $name ..."

    case "$MODE" in
        sass)
            cuobjdump --dump-sass "$exe" > "$exe_out_dir/$name.sass"
            echo "     Saved SASS: $exe_out_dir/$name.sass"
            ;;
        ptx)
            cuobjdump --dump-ptx "$exe" > "$exe_out_dir/$name.ptx"
            echo "     Saved PTX: $exe_out_dir/$name.ptx"
            ;;
        both)
            cuobjdump --dump-sass "$exe" > "$exe_out_dir/$name.sass"
            cuobjdump --dump-ptx "$exe" > "$exe_out_dir/$name.ptx"
            echo "     Saved SASS: $exe_out_dir/$name.sass"
            echo "     Saved PTX: $exe_out_dir/$name.ptx"
            ;;
    esac
done

# ========== 完成 ==========
echo "[3/3] Done."
echo "All files are in: $OUT_DIR/"
echo "==============================="
