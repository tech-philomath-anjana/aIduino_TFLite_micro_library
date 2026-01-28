/*
 * AIduino_TFLiteMicro.h
 * 
 * TensorFlow Lite Micro Library for STM32F407VGT6 (AIduino Board)
 * 
 * This library provides TensorFlow Lite Micro inference capabilities
 * optimized for the ARM Cortex-M4 with FPU and CMSIS-NN acceleration.
 * 
 * Hardware: AIduino Board (STM32F407VGT6)
 * - 1MB Flash, 192KB SRAM + 64KB CCM
 * - ARM Cortex-M4 @ 168MHz with FPU
 * - INMP441 I2S Microphone
 * - LIS3DH/MPU6050 Accelerometer
 * 
 * Author: AIduino Project
 * License: Apache 2.0
 */

#ifndef AIDUINO_TFLITEMICRO_H
#define AIDUINO_TFLITEMICRO_H

#include <Arduino.h>
#include <stdint.h>
#include <stddef.h>

// ============================================================================
// Platform Detection and Configuration
// ============================================================================

#if defined(STM32F4xx) || defined(STM32F407xx) || defined(STM32F407VG)
  #define AIDUINO_PLATFORM_STM32F4
  #define AIDUINO_HAS_FPU 1
  #define AIDUINO_HAS_DSP 1
  #define AIDUINO_FLASH_SIZE (1024 * 1024)  // 1MB
  #define AIDUINO_SRAM_SIZE  (192 * 1024)   // 192KB main SRAM
  #define AIDUINO_CCM_SIZE   (64 * 1024)    // 64KB CCM
#else
  #warning "AIduino_TFLiteMicro: Non-STM32F4 platform detected. Some optimizations may not be available."
  #define AIDUINO_HAS_FPU 0
  #define AIDUINO_HAS_DSP 0
#endif

// ============================================================================
// TensorFlow Lite Micro Core Includes
// ============================================================================

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ============================================================================
// Memory Configuration
// ============================================================================

// Default tensor arena size (adjust based on your model)
#ifndef AIDUINO_TENSOR_ARENA_SIZE
  #define AIDUINO_TENSOR_ARENA_SIZE (50 * 1024)  // 50KB default
#endif

// Maximum supported tensor arena
#define AIDUINO_MAX_TENSOR_ARENA_SIZE (100 * 1024)  // 100KB max

// ============================================================================
// Error Codes
// ============================================================================

typedef enum {
  AIDUINO_OK = 0,
  AIDUINO_ERROR_MODEL_INVALID,
  AIDUINO_ERROR_MEMORY_ALLOC,
  AIDUINO_ERROR_INTERPRETER_INIT,
  AIDUINO_ERROR_TENSOR_ALLOC,
  AIDUINO_ERROR_INVOKE_FAILED,
  AIDUINO_ERROR_INPUT_SIZE_MISMATCH,
  AIDUINO_ERROR_OUTPUT_SIZE_MISMATCH,
  AIDUINO_ERROR_NOT_INITIALIZED
} AIduino_Error_t;

// ============================================================================
// AIduino ML Model Class
// ============================================================================

class AIduinoModel {
public:
  /**
   * Constructor
   * @param tensor_arena_size Size of tensor arena in bytes (default: 50KB)
   */
  AIduinoModel(size_t tensor_arena_size = AIDUINO_TENSOR_ARENA_SIZE);
  
  /**
   * Destructor - frees allocated memory
   */
  ~AIduinoModel();

  /**
   * Initialize the model from a TFLite flatbuffer
   * @param model_data Pointer to the model data (typically from .h file)
   * @param model_size Size of the model data in bytes
   * @return AIDUINO_OK on success, error code otherwise
   */
  AIduino_Error_t begin(const uint8_t* model_data, size_t model_size);
  
  /**
   * Initialize the model (overload for header file models)
   * @param model_data Pointer to the model data array
   * @return AIDUINO_OK on success, error code otherwise
   */
  AIduino_Error_t begin(const uint8_t* model_data);

  /**
   * Run inference on the model
   * @return AIDUINO_OK on success, error code otherwise
   */
  AIduino_Error_t predict();

  /**
   * Set input data for the model
   * @param data Pointer to input data
   * @param size Size of input data in bytes
   * @param input_index Index of the input tensor (default: 0)
   * @return AIDUINO_OK on success, error code otherwise
   */
  AIduino_Error_t setInput(const void* data, size_t size, int input_index = 0);
  
  /**
   * Set input data (float version)
   * @param data Pointer to float input data
   * @param num_elements Number of float elements
   * @param input_index Index of the input tensor (default: 0)
   * @return AIDUINO_OK on success, error code otherwise
   */
  AIduino_Error_t setInputFloat(const float* data, size_t num_elements, int input_index = 0);
  
  /**
   * Set input data (int8 quantized version)
   * @param data Pointer to int8 input data
   * @param num_elements Number of int8 elements
   * @param input_index Index of the input tensor (default: 0)
   * @return AIDUINO_OK on success, error code otherwise
   */
  AIduino_Error_t setInputInt8(const int8_t* data, size_t num_elements, int input_index = 0);

  /**
   * Get output data from the model
   * @param data Pointer to buffer for output data
   * @param size Size of the buffer
   * @param output_index Index of the output tensor (default: 0)
   * @return AIDUINO_OK on success, error code otherwise
   */
  AIduino_Error_t getOutput(void* data, size_t size, int output_index = 0);
  
  /**
   * Get output data (float version)
   * @param data Pointer to buffer for float output data
   * @param num_elements Number of elements to read
   * @param output_index Index of the output tensor (default: 0)
   * @return AIDUINO_OK on success, error code otherwise
   */
  AIduino_Error_t getOutputFloat(float* data, size_t num_elements, int output_index = 0);
  
  /**
   * Get pointer to raw output tensor
   * @param output_index Index of the output tensor (default: 0)
   * @return Pointer to output tensor data, or nullptr if not available
   */
  void* getOutputPtr(int output_index = 0);
  
  /**
   * Get the index of the highest probability output (for classification)
   * @param output_index Index of the output tensor (default: 0)
   * @return Index of the highest value, or -1 on error
   */
  int getTopClass(int output_index = 0);
  
  /**
   * Get the confidence of the highest probability output
   * @param output_index Index of the output tensor (default: 0)
   * @return Confidence value (0.0 to 1.0), or -1.0 on error
   */
  float getTopConfidence(int output_index = 0);

  /**
   * Get input tensor information
   * @param input_index Index of the input tensor (default: 0)
   * @return Pointer to input tensor, or nullptr if not available
   */
  TfLiteTensor* getInputTensor(int input_index = 0);
  
  /**
   * Get output tensor information
   * @param output_index Index of the output tensor (default: 0)
   * @return Pointer to output tensor, or nullptr if not available
   */
  TfLiteTensor* getOutputTensor(int output_index = 0);

  /**
   * Get the number of input tensors
   * @return Number of inputs
   */
  size_t getInputCount();
  
  /**
   * Get the number of output tensors
   * @return Number of outputs
   */
  size_t getOutputCount();

  /**
   * Get model arena usage (bytes used)
   * @return Arena bytes used
   */
  size_t getArenaUsed();
  
  /**
   * Get last inference time in microseconds
   * @return Inference time in microseconds
   */
  uint32_t getInferenceTimeUs();

  /**
   * Print model information to Serial
   */
  void printModelInfo();
  
  /**
   * Get error string for error code
   * @param error Error code
   * @return Human-readable error string
   */
  static const char* getErrorString(AIduino_Error_t error);

  /**
   * Check if model is initialized
   * @return true if initialized, false otherwise
   */
  bool isInitialized() const { return _initialized; }

private:
  // TensorFlow Lite Micro components
  const tflite::Model* _model;
  tflite::MicroInterpreter* _interpreter;
  tflite::MicroMutableOpResolver<20>* _resolver;  // Support up to 20 ops
  
  // Memory management
  uint8_t* _tensor_arena;
  size_t _tensor_arena_size;
  
  // State
  bool _initialized;
  uint32_t _last_inference_time_us;
  
  // Internal methods
  AIduino_Error_t _registerOps();
  void _cleanup();
};

// ============================================================================
// AIduino Sensor Helpers (for common ML applications)
// ============================================================================

/**
 * MFCC Feature Extractor for audio
 * Converts raw audio samples to MFCC features for keyword spotting
 */
class AIduinoAudioFeatures {
public:
  AIduinoAudioFeatures();
  
  /**
   * Initialize the feature extractor
   * @param sample_rate Audio sample rate (typically 16000)
   * @param window_size_ms Window size in milliseconds (typically 30)
   * @param window_stride_ms Window stride in milliseconds (typically 20)
   * @param num_mfcc Number of MFCC coefficients (typically 13)
   * @return true on success
   */
  bool begin(uint32_t sample_rate = 16000, 
             uint32_t window_size_ms = 30,
             uint32_t window_stride_ms = 20,
             uint8_t num_mfcc = 13);
  
  /**
   * Process audio samples and extract MFCC features
   * @param audio_samples Pointer to audio samples
   * @param num_samples Number of samples
   * @param features Output buffer for MFCC features
   * @param max_features Maximum number of feature frames
   * @return Number of feature frames generated
   */
  int extractFeatures(const int16_t* audio_samples, 
                      size_t num_samples,
                      float* features,
                      size_t max_features);

private:
  uint32_t _sample_rate;
  uint32_t _window_size;
  uint32_t _window_stride;
  uint8_t _num_mfcc;
  bool _initialized;
};

/**
 * Accelerometer Feature Processor for gesture recognition
 */
class AIduinoAccelFeatures {
public:
  AIduinoAccelFeatures();
  
  /**
   * Initialize the feature processor
   * @param sample_rate Accelerometer sample rate in Hz
   * @param window_samples Number of samples per window
   * @return true on success
   */
  bool begin(uint32_t sample_rate = 100, uint32_t window_samples = 128);
  
  /**
   * Add accelerometer sample to buffer
   * @param x X-axis acceleration
   * @param y Y-axis acceleration
   * @param z Z-axis acceleration
   * @return true if window is complete and ready for inference
   */
  bool addSample(float x, float y, float z);
  
  /**
   * Get the current feature window
   * @param features Output buffer (must be at least window_samples * 3 floats)
   * @return true if features are available
   */
  bool getFeatures(float* features);
  
  /**
   * Reset the feature buffer
   */
  void reset();
  
  /**
   * Check if window is ready
   * @return true if enough samples collected
   */
  bool isReady() const { return _ready; }

private:
  float* _buffer;
  uint32_t _sample_rate;
  uint32_t _window_samples;
  uint32_t _current_sample;
  bool _initialized;
  bool _ready;
};

// ============================================================================
// Utility Functions
// ============================================================================

namespace AIduinoUtils {
  /**
   * Softmax function for converting logits to probabilities
   * @param input Input logits array
   * @param output Output probabilities array
   * @param size Number of elements
   */
  void softmax(const float* input, float* output, size_t size);
  
  /**
   * Quantize float to int8
   * @param value Float value to quantize
   * @param scale Quantization scale
   * @param zero_point Quantization zero point
   * @return Quantized int8 value
   */
  int8_t quantizeInt8(float value, float scale, int32_t zero_point);
  
  /**
   * Dequantize int8 to float
   * @param value Quantized int8 value
   * @param scale Quantization scale
   * @param zero_point Quantization zero point
   * @return Dequantized float value
   */
  float dequantizeInt8(int8_t value, float scale, int32_t zero_point);
  
  /**
   * Get system tick in microseconds
   * @return Current tick in microseconds
   */
  uint32_t getMicros();
  
  /**
   * Print memory usage information
   */
  void printMemoryInfo();
}

// ============================================================================
// Platform-Specific Optimizations
// ============================================================================

#ifdef AIDUINO_PLATFORM_STM32F4

namespace AIduinoHAL {
  /**
   * Initialize hardware acceleration (FPU, caches, etc.)
   */
  void initHardwareAcceleration();
  
  /**
   * Get free heap memory
   * @return Free heap bytes
   */
  size_t getFreeHeap();
  
  /**
   * Get CPU frequency
   * @return CPU frequency in Hz
   */
  uint32_t getCPUFrequency();
}

#endif // AIDUINO_PLATFORM_STM32F4

#endif // AIDUINO_TFLITEMICRO_H
