#!/bin/bash
#
# setup_tflm.sh
# 
# This script downloads the complete TensorFlow Lite Micro source files
# from the official repository and integrates them into the AIduino library.
#
# Usage: ./setup_tflm.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM_VERSION="main"  # or specific commit hash
TEMP_DIR="${SCRIPT_DIR}/.tflm_temp"

echo "=========================================="
echo "AIduino TFLite Micro Setup Script"
echo "=========================================="
echo ""

# Check for git
if ! command -v git &> /dev/null; then
    echo "ERROR: git is required but not installed."
    exit 1
fi

# Create temp directory
echo "Creating temporary directory..."
mkdir -p "${TEMP_DIR}"
cd "${TEMP_DIR}"

# Clone TFLM repository (shallow clone for speed)
echo "Downloading TensorFlow Lite Micro source..."
if [ -d "tflite-micro" ]; then
    echo "Existing clone found, pulling latest..."
    cd tflite-micro
    git pull
    cd ..
else
    git clone --depth 1 https://github.com/tensorflow/tflite-micro.git
fi

# Clone FlatBuffers
echo "Downloading FlatBuffers..."
if [ ! -d "flatbuffers" ]; then
    git clone --depth 1 --branch v23.5.26 https://github.com/google/flatbuffers.git
fi

# Create target directories
echo "Setting up library structure..."
mkdir -p "${SCRIPT_DIR}/src/tensorflow/lite/micro/kernels"
mkdir -p "${SCRIPT_DIR}/src/tensorflow/lite/c"
mkdir -p "${SCRIPT_DIR}/src/tensorflow/lite/core/c"
mkdir -p "${SCRIPT_DIR}/src/tensorflow/lite/core/api"
mkdir -p "${SCRIPT_DIR}/src/tensorflow/lite/kernels/internal/reference"
mkdir -p "${SCRIPT_DIR}/src/tensorflow/lite/schema"
mkdir -p "${SCRIPT_DIR}/src/third_party/flatbuffers/include/flatbuffers"
mkdir -p "${SCRIPT_DIR}/src/third_party/gemmlowp"
mkdir -p "${SCRIPT_DIR}/src/third_party/ruy"

# Copy TFLM core files
echo "Copying TensorFlow Lite Micro core files..."

# Core micro files
cp -r tflite-micro/tensorflow/lite/micro/*.h "${SCRIPT_DIR}/src/tensorflow/lite/micro/"
cp -r tflite-micro/tensorflow/lite/micro/*.cc "${SCRIPT_DIR}/src/tensorflow/lite/micro/"

# Kernels
cp -r tflite-micro/tensorflow/lite/micro/kernels/*.h "${SCRIPT_DIR}/src/tensorflow/lite/micro/kernels/"
cp -r tflite-micro/tensorflow/lite/micro/kernels/*.cc "${SCRIPT_DIR}/src/tensorflow/lite/micro/kernels/"

# C API
cp -r tflite-micro/tensorflow/lite/c/*.h "${SCRIPT_DIR}/src/tensorflow/lite/c/"
cp -r tflite-micro/tensorflow/lite/c/*.c "${SCRIPT_DIR}/src/tensorflow/lite/c/" 2>/dev/null || true

# Core API
cp -r tflite-micro/tensorflow/lite/core/c/*.h "${SCRIPT_DIR}/src/tensorflow/lite/core/c/" 2>/dev/null || true
cp -r tflite-micro/tensorflow/lite/core/api/*.h "${SCRIPT_DIR}/src/tensorflow/lite/core/api/" 2>/dev/null || true
cp -r tflite-micro/tensorflow/lite/core/api/*.cc "${SCRIPT_DIR}/src/tensorflow/lite/core/api/" 2>/dev/null || true

# Kernels internal
cp -r tflite-micro/tensorflow/lite/kernels/internal/*.h "${SCRIPT_DIR}/src/tensorflow/lite/kernels/internal/" 2>/dev/null || true
cp -r tflite-micro/tensorflow/lite/kernels/internal/reference/*.h "${SCRIPT_DIR}/src/tensorflow/lite/kernels/internal/reference/" 2>/dev/null || true

# Schema
cp tflite-micro/tensorflow/lite/schema/schema_generated.h "${SCRIPT_DIR}/src/tensorflow/lite/schema/"

# Copy FlatBuffers
echo "Copying FlatBuffers..."
cp flatbuffers/include/flatbuffers/*.h "${SCRIPT_DIR}/src/third_party/flatbuffers/include/flatbuffers/"

# Copy third-party headers
echo "Copying third-party dependencies..."

# Try to get gemmlowp and ruy if available
if [ -d "tflite-micro/third_party/gemmlowp" ]; then
    cp -r tflite-micro/third_party/gemmlowp/* "${SCRIPT_DIR}/src/third_party/gemmlowp/" 2>/dev/null || true
fi

if [ -d "tflite-micro/third_party/ruy" ]; then
    cp -r tflite-micro/third_party/ruy/* "${SCRIPT_DIR}/src/third_party/ruy/" 2>/dev/null || true
fi

# Rename .cc files to .cpp for Arduino compatibility
echo "Converting file extensions for Arduino compatibility..."
find "${SCRIPT_DIR}/src" -name "*.cc" -exec sh -c 'mv "$1" "${1%.cc}.cpp"' _ {} \;

# Create Arduino-specific compatibility header
echo "Creating Arduino compatibility layer..."
cat > "${SCRIPT_DIR}/src/tensorflow/lite/micro/arduino_compat.h" << 'EOF'
/*
 * arduino_compat.h
 * 
 * Arduino compatibility layer for TensorFlow Lite Micro
 */

#ifndef TENSORFLOW_LITE_MICRO_ARDUINO_COMPAT_H_
#define TENSORFLOW_LITE_MICRO_ARDUINO_COMPAT_H_

#ifdef ARDUINO
  #include <Arduino.h>
  
  // Use Arduino's Serial for debug output
  #define TF_LITE_MICRO_DEBUG_LOG(...) \
    do { \
      char _buf[128]; \
      snprintf(_buf, sizeof(_buf), __VA_ARGS__); \
      Serial.print(_buf); \
    } while(0)
  
  // Memory allocation using Arduino heap
  #define TF_LITE_MICRO_MALLOC(size) malloc(size)
  #define TF_LITE_MICRO_FREE(ptr) free(ptr)
  
  // Timing using Arduino's micros()
  #define TF_LITE_MICRO_GET_TIMESTAMP_US() micros()
  
#else
  // Standard C/C++ fallbacks
  #include <stdio.h>
  #include <stdlib.h>
  #include <time.h>
  
  #define TF_LITE_MICRO_DEBUG_LOG(...) printf(__VA_ARGS__)
  #define TF_LITE_MICRO_MALLOC(size) malloc(size)
  #define TF_LITE_MICRO_FREE(ptr) free(ptr)
  #define TF_LITE_MICRO_GET_TIMESTAMP_US() (clock() * 1000000 / CLOCKS_PER_SEC)
#endif

// STM32-specific optimizations
#if defined(STM32F4xx) || defined(STM32F407xx)
  #define TF_LITE_MICRO_USE_CMSIS_NN 1
  #define TF_LITE_MICRO_HAS_FPU 1
#endif

#endif  // TENSORFLOW_LITE_MICRO_ARDUINO_COMPAT_H_
EOF

# Clean up
echo "Cleaning up..."
cd "${SCRIPT_DIR}"
rm -rf "${TEMP_DIR}"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "The AIduino TFLite Micro library is now ready to use."
echo ""
echo "To use in Arduino IDE:"
echo "1. Copy this folder to your Arduino libraries directory"
echo "2. Restart Arduino IDE"
echo "3. Go to File > Examples > AIduino_TFLiteMicro"
echo ""
echo "For more information, see README.md"
