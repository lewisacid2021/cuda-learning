#!/usr/bin/env bash

set -e  # 遇到错误立即退出

# 构建目录
BUILD_DIR="../build"

echo "==== CUDA Project Build Script ===="

# 创建 build 目录
if [ ! -d "$BUILD_DIR" ]; then
    echo "[1/3] Creating build directory..."
    mkdir -p $BUILD_DIR
fi

cd $BUILD_DIR

# 运行 CMake
echo "[2/3] Running CMake ..."
cmake ..

# 自动检测 CPU 线程数
CORES=$(nproc --all)

echo "[3/3] Building with $CORES threads ..."
make -j$CORES

echo "==== Build Done ===="
echo "Executables are located in: $BUILD_DIR/bin/"
