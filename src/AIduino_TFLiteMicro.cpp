/*
 * AIduino_TFLiteMicro.cpp
 * 
 * TensorFlow Lite Micro Library Implementation for STM32F407VGT6
 * 
 * Author: AIduino Project
 * License: Apache 2.0
 */

#include "AIduino_TFLiteMicro.h"
#include <string.h>

// TensorFlow Lite Micro operator includes
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

// For timing
#ifdef AIDUINO_PLATFORM_STM32F4
  extern "C" {
    #include "stm32f4xx_hal.h"
  }
#endif

// ============================================================================
// AIduinoModel Implementation
// ============================================================================

AIduinoModel::AIduinoModel(size_t tensor_arena_size) 
  : _model(nullptr),
    _interpreter(nullptr),
    _resolver(nullptr),
    _tensor_arena(nullptr),
    _tensor_arena_size(tensor_arena_size),
    _initialized(false),
    _last_inference_time_us(0) {
  
  // Clamp arena size
  if (_tensor_arena_size > AIDUINO_MAX_TENSOR_ARENA_SIZE) {
    _tensor_arena_size = AIDUINO_MAX_TENSOR_ARENA_SIZE;
  }
}

AIduinoModel::~AIduinoModel() {
  _cleanup();
}

void AIduinoModel::_cleanup() {
  if (_interpreter) {
    delete _interpreter;
    _interpreter = nullptr;
  }
  if (_resolver) {
    delete _resolver;
    _resolver = nullptr;
  }
  if (_tensor_arena) {
    delete[] _tensor_arena;
    _tensor_arena = nullptr;
  }
  _model = nullptr;
  _initialized = false;
}

AIduino_Error_t AIduinoModel::_registerOps() {
  // Register commonly used ops for TinyML applications
  // Add more ops as needed for your specific models
  
  TfLiteStatus status;
  
  // Fully connected (Dense) layers
  status = _resolver->AddFullyConnected();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  // Activation functions
  status = _resolver->AddRelu();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  status = _resolver->AddRelu6();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  status = _resolver->AddSoftmax();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  status = _resolver->AddLogistic();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  status = _resolver->AddTanh();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  // Convolution operations (for CNN models)
  status = _resolver->AddConv2D();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  status = _resolver->AddDepthwiseConv2D();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  // Pooling operations
  status = _resolver->AddMaxPool2D();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  status = _resolver->AddAveragePool2D();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  // Reshape operations
  status = _resolver->AddReshape();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  status = _resolver->AddFlatten();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  // Quantization operations
  status = _resolver->AddQuantize();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  status = _resolver->AddDequantize();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  // Other common operations
  status = _resolver->AddMean();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  status = _resolver->AddAdd();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  status = _resolver->AddMul();
  if (status != kTfLiteOk) return AIDUINO_ERROR_INTERPRETER_INIT;
  
  return AIDUINO_OK;
}

AIduino_Error_t AIduinoModel::begin(const uint8_t* model_data, size_t model_size) {
  (void)model_size;  // Size is encoded in the flatbuffer
  return begin(model_data);
}

AIduino_Error_t AIduinoModel::begin(const uint8_t* model_data) {
  // Clean up any previous initialization
  _cleanup();
  
  // Initialize TensorFlow Lite Micro system
  tflite::InitializeTarget();
  
  // Map the model
  _model = tflite::GetModel(model_data);
  if (_model == nullptr) {
    return AIDUINO_ERROR_MODEL_INVALID;
  }
  
  // Verify model schema version
  if (_model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version %d does not match %d",
                _model->version(), TFLITE_SCHEMA_VERSION);
    return AIDUINO_ERROR_MODEL_INVALID;
  }
  
  // Allocate tensor arena
  _tensor_arena = new uint8_t[_tensor_arena_size];
  if (_tensor_arena == nullptr) {
    return AIDUINO_ERROR_MEMORY_ALLOC;
  }
  
  // Create op resolver
  _resolver = new tflite::MicroMutableOpResolver<20>();
  if (_resolver == nullptr) {
    _cleanup();
    return AIDUINO_ERROR_MEMORY_ALLOC;
  }
  
  // Register operations
  AIduino_Error_t err = _registerOps();
  if (err != AIDUINO_OK) {
    _cleanup();
    return err;
  }
  
  // Create interpreter
  _interpreter = new tflite::MicroInterpreter(
    _model, *_resolver, _tensor_arena, _tensor_arena_size);
  
  if (_interpreter == nullptr) {
    _cleanup();
    return AIDUINO_ERROR_INTERPRETER_INIT;
  }
  
  // Allocate tensors
  TfLiteStatus status = _interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    _cleanup();
    return AIDUINO_ERROR_TENSOR_ALLOC;
  }
  
  _initialized = true;
  return AIDUINO_OK;
}

AIduino_Error_t AIduinoModel::predict() {
  if (!_initialized) {
    return AIDUINO_ERROR_NOT_INITIALIZED;
  }
  
  // Measure inference time
  uint32_t start_time = AIduinoUtils::getMicros();
  
  TfLiteStatus status = _interpreter->Invoke();
  
  _last_inference_time_us = AIduinoUtils::getMicros() - start_time;
  
  if (status != kTfLiteOk) {
    return AIDUINO_ERROR_INVOKE_FAILED;
  }
  
  return AIDUINO_OK;
}

AIduino_Error_t AIduinoModel::setInput(const void* data, size_t size, int input_index) {
  if (!_initialized) {
    return AIDUINO_ERROR_NOT_INITIALIZED;
  }
  
  TfLiteTensor* input = _interpreter->input(input_index);
  if (input == nullptr) {
    return AIDUINO_ERROR_INPUT_SIZE_MISMATCH;
  }
  
  if (size > input->bytes) {
    return AIDUINO_ERROR_INPUT_SIZE_MISMATCH;
  }
  
  memcpy(input->data.raw, data, size);
  return AIDUINO_OK;
}

AIduino_Error_t AIduinoModel::setInputFloat(const float* data, size_t num_elements, int input_index) {
  if (!_initialized) {
    return AIDUINO_ERROR_NOT_INITIALIZED;
  }
  
  TfLiteTensor* input = _interpreter->input(input_index);
  if (input == nullptr) {
    return AIDUINO_ERROR_INPUT_SIZE_MISMATCH;
  }
  
  // Handle both float and quantized input
  if (input->type == kTfLiteFloat32) {
    size_t size = num_elements * sizeof(float);
    if (size > input->bytes) {
      return AIDUINO_ERROR_INPUT_SIZE_MISMATCH;
    }
    memcpy(input->data.f, data, size);
  } else if (input->type == kTfLiteInt8) {
    // Quantize the input
    float scale = input->params.scale;
    int32_t zero_point = input->params.zero_point;
    
    if (num_elements > input->bytes) {
      return AIDUINO_ERROR_INPUT_SIZE_MISMATCH;
    }
    
    for (size_t i = 0; i < num_elements; i++) {
      input->data.int8[i] = AIduinoUtils::quantizeInt8(data[i], scale, zero_point);
    }
  } else {
    return AIDUINO_ERROR_INPUT_SIZE_MISMATCH;
  }
  
  return AIDUINO_OK;
}

AIduino_Error_t AIduinoModel::setInputInt8(const int8_t* data, size_t num_elements, int input_index) {
  if (!_initialized) {
    return AIDUINO_ERROR_NOT_INITIALIZED;
  }
  
  TfLiteTensor* input = _interpreter->input(input_index);
  if (input == nullptr || input->type != kTfLiteInt8) {
    return AIDUINO_ERROR_INPUT_SIZE_MISMATCH;
  }
  
  if (num_elements > input->bytes) {
    return AIDUINO_ERROR_INPUT_SIZE_MISMATCH;
  }
  
  memcpy(input->data.int8, data, num_elements);
  return AIDUINO_OK;
}

AIduino_Error_t AIduinoModel::getOutput(void* data, size_t size, int output_index) {
  if (!_initialized) {
    return AIDUINO_ERROR_NOT_INITIALIZED;
  }
  
  TfLiteTensor* output = _interpreter->output(output_index);
  if (output == nullptr) {
    return AIDUINO_ERROR_OUTPUT_SIZE_MISMATCH;
  }
  
  size_t copy_size = (size < output->bytes) ? size : output->bytes;
  memcpy(data, output->data.raw, copy_size);
  
  return AIDUINO_OK;
}

AIduino_Error_t AIduinoModel::getOutputFloat(float* data, size_t num_elements, int output_index) {
  if (!_initialized) {
    return AIDUINO_ERROR_NOT_INITIALIZED;
  }
  
  TfLiteTensor* output = _interpreter->output(output_index);
  if (output == nullptr) {
    return AIDUINO_ERROR_OUTPUT_SIZE_MISMATCH;
  }
  
  if (output->type == kTfLiteFloat32) {
    size_t copy_elements = (num_elements < (output->bytes / sizeof(float))) 
                            ? num_elements : (output->bytes / sizeof(float));
    memcpy(data, output->data.f, copy_elements * sizeof(float));
  } else if (output->type == kTfLiteInt8) {
    // Dequantize the output
    float scale = output->params.scale;
    int32_t zero_point = output->params.zero_point;
    
    size_t copy_elements = (num_elements < output->bytes) ? num_elements : output->bytes;
    for (size_t i = 0; i < copy_elements; i++) {
      data[i] = AIduinoUtils::dequantizeInt8(output->data.int8[i], scale, zero_point);
    }
  } else {
    return AIDUINO_ERROR_OUTPUT_SIZE_MISMATCH;
  }
  
  return AIDUINO_OK;
}

void* AIduinoModel::getOutputPtr(int output_index) {
  if (!_initialized) {
    return nullptr;
  }
  
  TfLiteTensor* output = _interpreter->output(output_index);
  if (output == nullptr) {
    return nullptr;
  }
  
  return output->data.raw;
}

int AIduinoModel::getTopClass(int output_index) {
  if (!_initialized) {
    return -1;
  }
  
  TfLiteTensor* output = _interpreter->output(output_index);
  if (output == nullptr) {
    return -1;
  }
  
  int num_classes;
  float max_val = -1e9f;
  int max_idx = 0;
  
  if (output->type == kTfLiteFloat32) {
    num_classes = output->bytes / sizeof(float);
    for (int i = 0; i < num_classes; i++) {
      if (output->data.f[i] > max_val) {
        max_val = output->data.f[i];
        max_idx = i;
      }
    }
  } else if (output->type == kTfLiteInt8) {
    num_classes = output->bytes;
    for (int i = 0; i < num_classes; i++) {
      float val = AIduinoUtils::dequantizeInt8(output->data.int8[i], 
                                                output->params.scale, 
                                                output->params.zero_point);
      if (val > max_val) {
        max_val = val;
        max_idx = i;
      }
    }
  } else {
    return -1;
  }
  
  return max_idx;
}

float AIduinoModel::getTopConfidence(int output_index) {
  if (!_initialized) {
    return -1.0f;
  }
  
  TfLiteTensor* output = _interpreter->output(output_index);
  if (output == nullptr) {
    return -1.0f;
  }
  
  int num_classes;
  float max_val = -1e9f;
  
  if (output->type == kTfLiteFloat32) {
    num_classes = output->bytes / sizeof(float);
    for (int i = 0; i < num_classes; i++) {
      if (output->data.f[i] > max_val) {
        max_val = output->data.f[i];
      }
    }
  } else if (output->type == kTfLiteInt8) {
    num_classes = output->bytes;
    for (int i = 0; i < num_classes; i++) {
      float val = AIduinoUtils::dequantizeInt8(output->data.int8[i],
                                                output->params.scale,
                                                output->params.zero_point);
      if (val > max_val) {
        max_val = val;
      }
    }
  } else {
    return -1.0f;
  }
  
  return max_val;
}

TfLiteTensor* AIduinoModel::getInputTensor(int input_index) {
  if (!_initialized) {
    return nullptr;
  }
  return _interpreter->input(input_index);
}

TfLiteTensor* AIduinoModel::getOutputTensor(int output_index) {
  if (!_initialized) {
    return nullptr;
  }
  return _interpreter->output(output_index);
}

size_t AIduinoModel::getInputCount() {
  if (!_initialized) {
    return 0;
  }
  return _interpreter->inputs_size();
}

size_t AIduinoModel::getOutputCount() {
  if (!_initialized) {
    return 0;
  }
  return _interpreter->outputs_size();
}

size_t AIduinoModel::getArenaUsed() {
  if (!_initialized) {
    return 0;
  }
  return _interpreter->arena_used_bytes();
}

uint32_t AIduinoModel::getInferenceTimeUs() {
  return _last_inference_time_us;
}

void AIduinoModel::printModelInfo() {
  if (!_initialized) {
    Serial.println("Model not initialized");
    return;
  }
  
  Serial.println("=== AIduino Model Info ===");
  Serial.print("Arena size: ");
  Serial.print(_tensor_arena_size);
  Serial.println(" bytes");
  Serial.print("Arena used: ");
  Serial.print(getArenaUsed());
  Serial.println(" bytes");
  Serial.print("Inputs: ");
  Serial.println(getInputCount());
  Serial.print("Outputs: ");
  Serial.println(getOutputCount());
  
  // Print input tensor details
  for (size_t i = 0; i < getInputCount(); i++) {
    TfLiteTensor* input = getInputTensor(i);
    Serial.print("  Input ");
    Serial.print(i);
    Serial.print(": ");
    Serial.print(input->bytes);
    Serial.print(" bytes, type=");
    Serial.println(input->type);
  }
  
  // Print output tensor details
  for (size_t i = 0; i < getOutputCount(); i++) {
    TfLiteTensor* output = getOutputTensor(i);
    Serial.print("  Output ");
    Serial.print(i);
    Serial.print(": ");
    Serial.print(output->bytes);
    Serial.print(" bytes, type=");
    Serial.println(output->type);
  }
  
  Serial.println("========================");
}

const char* AIduinoModel::getErrorString(AIduino_Error_t error) {
  switch (error) {
    case AIDUINO_OK: return "OK";
    case AIDUINO_ERROR_MODEL_INVALID: return "Invalid model";
    case AIDUINO_ERROR_MEMORY_ALLOC: return "Memory allocation failed";
    case AIDUINO_ERROR_INTERPRETER_INIT: return "Interpreter init failed";
    case AIDUINO_ERROR_TENSOR_ALLOC: return "Tensor allocation failed";
    case AIDUINO_ERROR_INVOKE_FAILED: return "Invoke failed";
    case AIDUINO_ERROR_INPUT_SIZE_MISMATCH: return "Input size mismatch";
    case AIDUINO_ERROR_OUTPUT_SIZE_MISMATCH: return "Output size mismatch";
    case AIDUINO_ERROR_NOT_INITIALIZED: return "Not initialized";
    default: return "Unknown error";
  }
}

// ============================================================================
// AIduinoAudioFeatures Implementation
// ============================================================================

AIduinoAudioFeatures::AIduinoAudioFeatures()
  : _sample_rate(16000),
    _window_size(480),
    _window_stride(320),
    _num_mfcc(13),
    _initialized(false) {
}

bool AIduinoAudioFeatures::begin(uint32_t sample_rate, 
                                  uint32_t window_size_ms,
                                  uint32_t window_stride_ms,
                                  uint8_t num_mfcc) {
  _sample_rate = sample_rate;
  _window_size = (sample_rate * window_size_ms) / 1000;
  _window_stride = (sample_rate * window_stride_ms) / 1000;
  _num_mfcc = num_mfcc;
  _initialized = true;
  return true;
}

int AIduinoAudioFeatures::extractFeatures(const int16_t* audio_samples,
                                           size_t num_samples,
                                           float* features,
                                           size_t max_features) {
  if (!_initialized) {
    return 0;
  }
  
  // Simple MFCC extraction - in production, use a proper DSP library
  // This is a placeholder that should be replaced with actual MFCC computation
  // using libraries like CMSIS-DSP or a dedicated MFCC implementation
  
  int num_frames = 0;
  size_t offset = 0;
  
  while (offset + _window_size <= num_samples && num_frames < (int)max_features) {
    // Compute simple energy-based feature as placeholder
    float energy = 0;
    for (size_t i = 0; i < _window_size; i++) {
      float sample = audio_samples[offset + i] / 32768.0f;
      energy += sample * sample;
    }
    
    // Normalize and store (placeholder for actual MFCC)
    for (int j = 0; j < _num_mfcc; j++) {
      features[num_frames * _num_mfcc + j] = energy / _window_size;
    }
    
    offset += _window_stride;
    num_frames++;
  }
  
  return num_frames;
}

// ============================================================================
// AIduinoAccelFeatures Implementation
// ============================================================================

AIduinoAccelFeatures::AIduinoAccelFeatures()
  : _buffer(nullptr),
    _sample_rate(100),
    _window_samples(128),
    _current_sample(0),
    _initialized(false),
    _ready(false) {
}

bool AIduinoAccelFeatures::begin(uint32_t sample_rate, uint32_t window_samples) {
  _sample_rate = sample_rate;
  _window_samples = window_samples;
  
  if (_buffer) {
    delete[] _buffer;
  }
  
  _buffer = new float[_window_samples * 3];  // x, y, z for each sample
  if (_buffer == nullptr) {
    return false;
  }
  
  _current_sample = 0;
  _ready = false;
  _initialized = true;
  return true;
}

bool AIduinoAccelFeatures::addSample(float x, float y, float z) {
  if (!_initialized) {
    return false;
  }
  
  if (_current_sample < _window_samples) {
    size_t idx = _current_sample * 3;
    _buffer[idx] = x;
    _buffer[idx + 1] = y;
    _buffer[idx + 2] = z;
    _current_sample++;
  }
  
  if (_current_sample >= _window_samples) {
    _ready = true;
    return true;
  }
  
  return false;
}

bool AIduinoAccelFeatures::getFeatures(float* features) {
  if (!_initialized || !_ready) {
    return false;
  }
  
  memcpy(features, _buffer, _window_samples * 3 * sizeof(float));
  return true;
}

void AIduinoAccelFeatures::reset() {
  _current_sample = 0;
  _ready = false;
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================

namespace AIduinoUtils {

void softmax(const float* input, float* output, size_t size) {
  float max_val = input[0];
  for (size_t i = 1; i < size; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }
  
  float sum = 0;
  for (size_t i = 0; i < size; i++) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }
  
  for (size_t i = 0; i < size; i++) {
    output[i] /= sum;
  }
}

int8_t quantizeInt8(float value, float scale, int32_t zero_point) {
  int32_t quantized = static_cast<int32_t>(roundf(value / scale)) + zero_point;
  if (quantized < -128) quantized = -128;
  if (quantized > 127) quantized = 127;
  return static_cast<int8_t>(quantized);
}

float dequantizeInt8(int8_t value, float scale, int32_t zero_point) {
  return scale * (static_cast<int32_t>(value) - zero_point);
}

uint32_t getMicros() {
#ifdef AIDUINO_PLATFORM_STM32F4
  return micros();
#else
  return micros();
#endif
}

void printMemoryInfo() {
#ifdef AIDUINO_PLATFORM_STM32F4
  Serial.println("=== Memory Info ===");
  Serial.print("Free heap: ");
  Serial.print(AIduinoHAL::getFreeHeap());
  Serial.println(" bytes");
  Serial.print("CPU freq: ");
  Serial.print(AIduinoHAL::getCPUFrequency() / 1000000);
  Serial.println(" MHz");
  Serial.println("==================");
#else
  Serial.println("Memory info not available for this platform");
#endif
}

} // namespace AIduinoUtils

// ============================================================================
// Platform-Specific HAL Implementation
// ============================================================================

#ifdef AIDUINO_PLATFORM_STM32F4

namespace AIduinoHAL {

void initHardwareAcceleration() {
  // Enable FPU
  SCB->CPACR |= ((3UL << 10*2) | (3UL << 11*2));  // Set CP10 and CP11 Full Access
  
  // Enable I-Cache and D-Cache if available (F7 and later, not F4)
  // For F4, these are not available, but we keep the structure for portability
  
  // Configure system tick for accurate timing
  // This is typically already done by Arduino core
}

size_t getFreeHeap() {
  // Estimate free heap - this is a simplified version
  // For more accurate measurement, integrate with your memory allocator
  extern char _ebss;
  extern char _estack;
  
  char* heap_end = &_ebss;
  char* stack_ptr = (char*)__get_MSP();
  
  return stack_ptr - heap_end;
}

uint32_t getCPUFrequency() {
  return HAL_RCC_GetSysClockFreq();
}

} // namespace AIduinoHAL

#endif // AIDUINO_PLATFORM_STM32F4
