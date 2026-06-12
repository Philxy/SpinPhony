#!/bin/bash

# Navigate to the directory where this script is located (project root)
cd "$(dirname "$0")" || exit

BUILD_DIR="build"
EXECUTABLE="spinphony_run"

# Ensure the build directory exists
mkdir -p "$BUILD_DIR"

case "$1" in
    dev)
        echo "[SpinPhony] Running iterative dev loop..."
        cd "$BUILD_DIR" || exit
        
        # Check if Makefile exists; if not, run CMake automatically
        if [ ! -f Makefile ]; then
            echo "--> No Makefile found. Running initial CMake configuration..."
            cmake -DCMAKE_BUILD_TYPE=Debug ..
        fi
        
        make -j4 && ./"$EXECUTABLE"
        ;;
    debug)
        echo "[SpinPhony] Configuring a fresh DEBUG build..."
        cd "$BUILD_DIR" || exit
        rm -rf *
        cmake -DCMAKE_BUILD_TYPE=Debug ..
        make -j4
        echo ""
        echo "--> Debug build complete. Ready for gdb or cuda-gdb."
        echo "--> To debug, run: cd build && cuda-gdb ./$EXECUTABLE"
        ;;
    release)
        echo "[SpinPhony] Configuring a fresh RELEASE build..."
        cd "$BUILD_DIR" || exit
        rm -rf *
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j4
        echo ""
        echo "--> Release build complete. Maximum optimization applied."
        echo "--> To execute, run: cd build && ./$EXECUTABLE"
        ;;
    clean)
        echo "[SpinPhony] Wiping build directory..."
        rm -rf "$BUILD_DIR"/*
        echo "--> Clean complete."
        ;;
    *)
        echo "Usage: ./spin.sh {dev|debug|release|clean}"
        echo "  dev     : Fast incremental recompile and run"
        echo "  debug   : Clean cache and rebuild with debug symbols"
        echo "  release : Clean cache and rebuild with -O3 optimizations"
        echo "  clean   : Wipe the build directory entirely"
        ;;
esac
