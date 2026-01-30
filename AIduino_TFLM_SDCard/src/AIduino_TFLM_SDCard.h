/*
 * AIduino_TFLM_SDCard.h
 * 
 * TensorFlow Lite Micro Runtime with SD Card Model Loading
 * Load .tflite models directly from SD card instead of flash
 * 
 * USAGE:
 * ------
 * #include <AIduino_TFLM_SDCard.h>
 * 
 * TFLiteRuntime_SD runtime;
 * 
 * void setup() {
 *   SD.begin();
 *   runtime.beginFromSD("model.tflite", 30*1024);
 * }
 * 
 * void loop() {
 *   float input[] = {1.0, 2.0, 3.0};
 *   runtime.setInput(input, 3);
 *   runtime.invoke();
 *   int result = runtime.getOutputMaxIndex();
 * }
 */

#ifndef AIDUINO_TFLM_SDCARD_H
#define AIDUINO_TFLM_SDCARD_H

#include <Arduino.h>
#include <STM32SD.h>  // For STM32 SDIO
#include "tflite_types.h"
#include "tflite_model_parser.h"
#include "tflite_kernels.h"

// ============================================================================
// Configuration
// ============================================================================

#define TFLM_MAX_TENSORS 64
#define TFLM_MAX_OPERATORS 32
#define TFLM_MAX_DIMS 8
#define TFLM_MAX_MODEL_SIZE (100 * 1024)  // 100KB max model size

// ============================================================================
// Internal Tensor Storage
// ============================================================================

struct TensorInfo {
  TfLiteType type;
  int dims[TFLM_MAX_DIMS];
  int num_dims;
  size_t bytes;
  float scale;
  int32_t zero_point;
  void* data;
  int buffer_index;
  bool is_variable;
};

// ============================================================================
// TFLite Runtime with SD Card Support
// ============================================================================

class TFLiteRuntime_SD {
public:
  TFLiteRuntime_SD() : 
    initialized_(false),
    modelBuffer_(nullptr),
    modelSize_(0),
    arena_(nullptr),
    arena_size_(0),
    arena_used_(0),
    num_tensors_(0),
    num_inputs_(0),
    num_outputs_(0),
    inference_time_us_(0) {}
  
  ~TFLiteRuntime_SD() {
    if (modelBuffer_) {
      delete[] modelBuffer_;
    }
    if (arena_) {
      delete[] arena_;
    }
  }

  /**
   * Load model from SD card
   * 
   * @param filename    Model file on SD card (e.g., "model.tflite")
   * @param arena_size  Memory for tensors in bytes (start with 30*1024)
   * @return true on success
   */
  bool beginFromSD(const char* filename, size_t arena_size) {
    Serial.print("Loading model from SD: ");
    Serial.println(filename);
    
    // Open file
    File modelFile = SD.open(filename, FILE_READ);
    if (!modelFile) {
      Serial.println("ERROR: Cannot open model file");
      return false;
    }
    
    // Get file size
    modelSize_ = modelFile.size();
    Serial.print("Model size: ");
    Serial.print(modelSize_);
    Serial.println(" bytes");
    
    if (modelSize_ > TFLM_MAX_MODEL_SIZE) {
      Serial.println("ERROR: Model too large");
      modelFile.close();
      return false;
    }
    
    // Allocate buffer for model
    modelBuffer_ = new uint8_t[modelSize_];
    if (!modelBuffer_) {
      Serial.println("ERROR: Cannot allocate model buffer");
      modelFile.close();
      return false;
    }
    
    // Read model from SD card into RAM
    Serial.println("Reading model into RAM...");
    size_t bytesRead = modelFile.read(modelBuffer_, modelSize_);
    modelFile.close();
    
    if (bytesRead != modelSize_) {
      Serial.println("ERROR: Failed to read complete model");
      return false;
    }
    
    Serial.println("Model loaded into RAM");
    
    // Now initialize with the model in RAM
    return begin(modelBuffer_, arena_size);
  }

  /**
   * Initialize with model already in memory
   * (Can also be used directly if model is in flash)
   */
  bool begin(const unsigned char* model_data, size_t arena_size) {
    // Allocate arena
    arena_size_ = arena_size;
    arena_ = new uint8_t[arena_size_];
    if (!arena_) {
      Serial.println("ERROR: Cannot allocate arena");
      return false;
    }
    arena_used_ = 0;
    
    // Parse model
    if (!modelReader_.load(model_data)) {
      Serial.println("ERROR: Invalid model file");
      return false;
    }
    
    // Check version
    if (modelReader_.getVersion() != TFLITE_SCHEMA_VERSION) {
      Serial.print("WARNING: Model version ");
      Serial.print(modelReader_.getVersion());
      Serial.print(" != expected ");
      Serial.println(TFLITE_SCHEMA_VERSION);
    }
    
    // Get first subgraph
    subgraph_ = modelReader_.getSubgraph(0);
    if (!subgraph_) {
      Serial.println("ERROR: No subgraph found");
      return false;
    }
    
    // Allocate tensors
    if (!allocateTensors()) {
      Serial.println("ERROR: Failed to allocate tensors");
      return false;
    }
    
    // Get input/output info
    num_inputs_ = modelReader_.getNumInputs(subgraph_);
    num_outputs_ = modelReader_.getNumOutputs(subgraph_);
    
    for (size_t i = 0; i < num_inputs_ && i < 8; i++) {
      input_indices_[i] = modelReader_.getInput(subgraph_, i);
    }
    
    for (size_t i = 0; i < num_outputs_ && i < 8; i++) {
      output_indices_[i] = modelReader_.getOutput(subgraph_, i);
    }
    
    initialized_ = true;
    
    // Print info
    Serial.println("=== Model Loaded ===");
    Serial.print("Arena: ");
    Serial.print(arena_used_);
    Serial.print(" / ");
    Serial.print(arena_size_);
    Serial.println(" bytes");
    Serial.print("Tensors: ");
    Serial.println(num_tensors_);
    Serial.print("Inputs: ");
    Serial.println(num_inputs_);
    Serial.print("Outputs: ");
    Serial.println(num_outputs_);
    Serial.println("====================");
    
    return true;
  }

  /**
   * Set input data (float)
   */
  bool setInput(const float* data, size_t count, int input_index = 0) {
    if (!initialized_ || input_index >= (int)num_inputs_) return false;
    
    TensorInfo* input = &tensors_[input_indices_[input_index]];
    
    if (input->type == kTfLiteFloat32) {
      memcpy(input->data, data, count * sizeof(float));
    } 
    else if (input->type == kTfLiteInt8) {
      int8_t* dst = (int8_t*)input->data;
      for (size_t i = 0; i < count; i++) {
        int32_t q = (int32_t)roundf(data[i] / input->scale) + input->zero_point;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        dst[i] = (int8_t)q;
      }
    }
    return true;
  }

  /**
   * Run inference
   */
  bool invoke() {
    if (!initialized_) return false;
    
    uint32_t start = micros();
    
    size_t numOps = modelReader_.getNumOperators(subgraph_);
    
    for (size_t i = 0; i < numOps; i++) {
      const uint8_t* op = modelReader_.getOperator(subgraph_, i);
      
      if (!executeOperator(op)) {
        Serial.print("ERROR: Op ");
        Serial.print(i);
        Serial.println(" failed");
        return false;
      }
    }
    
    inference_time_us_ = micros() - start;
    return true;
  }

  /**
   * Get output as float array
   */
  float* getOutput(int output_index = 0) {
    if (!initialized_ || output_index >= (int)num_outputs_) return nullptr;
    
    TensorInfo* output = &tensors_[output_indices_[output_index]];
    
    if (output->type == kTfLiteFloat32) {
      return (float*)output->data;
    } else if (output->type == kTfLiteInt8) {
      static float dequantized[256];
      int8_t* src = (int8_t*)output->data;
      size_t count = output->bytes;
      if (count > 256) count = 256;
      
      for (size_t i = 0; i < count; i++) {
        dequantized[i] = output->scale * (src[i] - output->zero_point);
      }
      return dequantized;
    }
    return nullptr;
  }

  /**
   * Get index of maximum output value
   */
  int getOutputMaxIndex(int output_index = 0) {
    float* out = getOutput(output_index);
    if (!out) return -1;
    
    int count = getOutputSize(output_index);
    int maxIdx = 0;
    float maxVal = out[0];
    
    for (int i = 1; i < count; i++) {
      if (out[i] > maxVal) {
        maxVal = out[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  /**
   * Get maximum output value
   */
  float getOutputMaxValue(int output_index = 0) {
    float* out = getOutput(output_index);
    if (!out) return -1.0f;
    
    int count = getOutputSize(output_index);
    float maxVal = out[0];
    
    for (int i = 1; i < count; i++) {
      if (out[i] > maxVal) maxVal = out[i];
    }
    return maxVal;
  }

  int getInputSize(int input_index = 0) {
    if (!initialized_ || input_index >= (int)num_inputs_) return 0;
    TensorInfo* t = &tensors_[input_indices_[input_index]];
    if (t->type == kTfLiteFloat32) return t->bytes / sizeof(float);
    return t->bytes;
  }

  int getOutputSize(int output_index = 0) {
    if (!initialized_ || output_index >= (int)num_outputs_) return 0;
    TensorInfo* t = &tensors_[output_indices_[output_index]];
    if (t->type == kTfLiteFloat32) return t->bytes / sizeof(float);
    return t->bytes;
  }

  uint32_t getInferenceTimeUs() { return inference_time_us_; }
  size_t getArenaUsed() { return arena_used_; }
  size_t getModelSize() { return modelSize_; }
  bool isReady() { return initialized_; }

private:
  bool initialized_;
  
  // Model buffer (loaded from SD)
  uint8_t* modelBuffer_;
  size_t modelSize_;
  
  // Memory arena
  uint8_t* arena_;
  size_t arena_size_;
  size_t arena_used_;
  
  // Model
  TfLiteModelReader modelReader_;
  const uint8_t* subgraph_;
  
  // Tensors
  TensorInfo tensors_[TFLM_MAX_TENSORS];
  int num_tensors_;
  
  // Inputs/outputs
  int input_indices_[8];
  int output_indices_[8];
  size_t num_inputs_;
  size_t num_outputs_;
  
  // Timing
  uint32_t inference_time_us_;

  void* allocateArena(size_t bytes) {
    bytes = (bytes + 3) & ~3;
    if (arena_used_ + bytes > arena_size_) return nullptr;
    void* ptr = arena_ + arena_used_;
    arena_used_ += bytes;
    return ptr;
  }

  TfLiteType convertType(int8_t fbType) {
    switch (fbType) {
      case 0: return kTfLiteFloat32;
      case 1: return kTfLiteFloat16;
      case 2: return kTfLiteInt32;
      case 3: return kTfLiteUInt8;
      case 4: return kTfLiteInt64;
      case 6: return kTfLiteBool;
      case 7: return kTfLiteInt16;
      case 9: return kTfLiteInt8;
      default: return kTfLiteNoType;
    }
  }

  bool allocateTensors() {
    num_tensors_ = modelReader_.getNumTensors(subgraph_);
    
    if (num_tensors_ > TFLM_MAX_TENSORS) {
      Serial.println("ERROR: Too many tensors");
      return false;
    }
    
    for (int i = 0; i < num_tensors_; i++) {
      const uint8_t* tensor = modelReader_.getTensor(subgraph_, i);
      TensorInfo* info = &tensors_[i];
      
      info->type = convertType(modelReader_.getTensorType(tensor));
      
      const uint8_t* shape = modelReader_.getTensorShape(tensor);
      info->num_dims = shape ? modelReader_.vectorLength(shape) : 0;
      
      size_t totalElements = 1;
      for (int d = 0; d < info->num_dims && d < TFLM_MAX_DIMS; d++) {
        info->dims[d] = modelReader_.vectorGet<int32_t>(shape, d);
        totalElements *= info->dims[d];
      }
      
      info->bytes = totalElements * TfLiteTypeGetSize(info->type);
      
      const uint8_t* quant = modelReader_.getTensorQuantization(tensor);
      info->scale = modelReader_.getQuantizationScale(quant);
      info->zero_point = (int32_t)modelReader_.getQuantizationZeroPoint(quant);
      
      info->buffer_index = modelReader_.getTensorBuffer(tensor);
      
      size_t bufferSize = 0;
      const uint8_t* bufferData = modelReader_.getBufferData(info->buffer_index, &bufferSize);
      
      if (bufferData && bufferSize > 0) {
        info->data = (void*)bufferData;
        info->is_variable = false;
      } else {
        info->data = allocateArena(info->bytes);
        if (!info->data) {
          Serial.print("ERROR: Tensor ");
          Serial.println(i);
          return false;
        }
        memset(info->data, 0, info->bytes);
        info->is_variable = true;
      }
    }
    return true;
  }

  bool executeOperator(const uint8_t* op) {
    uint32_t opcodeIdx = modelReader_.getOperatorOpcodeIndex(op);
    int32_t builtinCode = modelReader_.getBuiltinCode(opcodeIdx);
    
    size_t numInputs = modelReader_.getNumOpInputs(op);
    size_t numOutputs = modelReader_.getNumOpOutputs(op);
    
    int32_t inputIdx[8], outputIdx[8];
    for (size_t i = 0; i < numInputs && i < 8; i++) {
      inputIdx[i] = modelReader_.getOpInput(op, i);
    }
    for (size_t i = 0; i < numOutputs && i < 8; i++) {
      outputIdx[i] = modelReader_.getOpOutput(op, i);
    }
    
    switch (builtinCode) {
      case kTfLiteBuiltinFullyConnected:
        return executeFullyConnected(inputIdx, outputIdx, numInputs);
      case kTfLiteBuiltinRelu:
        return executeRelu(inputIdx[0], outputIdx[0]);
      case kTfLiteBuiltinRelu6:
        return executeRelu6(inputIdx[0], outputIdx[0]);
      case kTfLiteBuiltinSoftmax:
        return executeSoftmax(inputIdx[0], outputIdx[0]);
      case kTfLiteBuiltinReshape:
        return executeReshape(inputIdx[0], outputIdx[0]);
      case kTfLiteBuiltinLogistic:
        return executeLogistic(inputIdx[0], outputIdx[0]);
      case kTfLiteBuiltinTanh:
        return executeTanh(inputIdx[0], outputIdx[0]);
      case kTfLiteBuiltinAdd:
        return executeAdd(inputIdx[0], inputIdx[1], outputIdx[0]);
      case kTfLiteBuiltinMul:
        return executeMul(inputIdx[0], inputIdx[1], outputIdx[0]);
      case kTfLiteBuiltinSub:
        return executeSub(inputIdx[0], inputIdx[1], outputIdx[0]);
      default:
        Serial.print("Unknown op: ");
        Serial.println(builtinCode);
        if (numInputs > 0 && numOutputs > 0) {
          TensorInfo* in = &tensors_[inputIdx[0]];
          TensorInfo* out = &tensors_[outputIdx[0]];
          if (in->bytes == out->bytes && out->is_variable) {
            memcpy(out->data, in->data, in->bytes);
          }
        }
        return true;
    }
  }

  // Operator implementations
  bool executeFullyConnected(int32_t* inputs, int32_t* outputs, size_t numInputs) {
    TensorInfo* input = &tensors_[inputs[0]];
    TensorInfo* weights = &tensors_[inputs[1]];
    TensorInfo* bias = numInputs > 2 && inputs[2] >= 0 ? &tensors_[inputs[2]] : nullptr;
    TensorInfo* output = &tensors_[outputs[0]];
    
    int input_size = input->bytes / TfLiteTypeGetSize(input->type);
    int output_size = output->bytes / TfLiteTypeGetSize(output->type);
    
    if (input->type == kTfLiteFloat32) {
      kernels::FullyConnected(
        (float*)input->data,
        (float*)weights->data,
        bias ? (float*)bias->data : nullptr,
        (float*)output->data,
        input_size, output_size, 0);
    }
    return true;
  }

  bool executeRelu(int32_t inputIdx, int32_t outputIdx) {
    TensorInfo* input = &tensors_[inputIdx];
    TensorInfo* output = &tensors_[outputIdx];
    size_t count = input->bytes / TfLiteTypeGetSize(input->type);
    if (input->type == kTfLiteFloat32) {
      kernels::Relu((float*)input->data, (float*)output->data, count);
    }
    return true;
  }

  bool executeRelu6(int32_t inputIdx, int32_t outputIdx) {
    TensorInfo* input = &tensors_[inputIdx];
    TensorInfo* output = &tensors_[outputIdx];
    size_t count = input->bytes / sizeof(float);
    if (input->type == kTfLiteFloat32) {
      kernels::Relu6((float*)input->data, (float*)output->data, count);
    }
    return true;
  }

  bool executeSoftmax(int32_t inputIdx, int32_t outputIdx) {
    TensorInfo* input = &tensors_[inputIdx];
    TensorInfo* output = &tensors_[outputIdx];
    if (input->type == kTfLiteFloat32) {
      size_t count = input->bytes / sizeof(float);
      kernels::Softmax((float*)input->data, (float*)output->data, count);
    }
    return true;
  }

  bool executeReshape(int32_t inputIdx, int32_t outputIdx) {
    TensorInfo* input = &tensors_[inputIdx];
    TensorInfo* output = &tensors_[outputIdx];
    kernels::Reshape(input->data, output->data, input->bytes);
    return true;
  }

  bool executeLogistic(int32_t inputIdx, int32_t outputIdx) {
    TensorInfo* input = &tensors_[inputIdx];
    TensorInfo* output = &tensors_[outputIdx];
    size_t count = input->bytes / sizeof(float);
    if (input->type == kTfLiteFloat32) {
      kernels::Logistic((float*)input->data, (float*)output->data, count);
    }
    return true;
  }

  bool executeTanh(int32_t inputIdx, int32_t outputIdx) {
    TensorInfo* input = &tensors_[inputIdx];
    TensorInfo* output = &tensors_[outputIdx];
    size_t count = input->bytes / sizeof(float);
    if (input->type == kTfLiteFloat32) {
      kernels::Tanh((float*)input->data, (float*)output->data, count);
    }
    return true;
  }

  bool executeAdd(int32_t aIdx, int32_t bIdx, int32_t outIdx) {
    TensorInfo* a = &tensors_[aIdx];
    TensorInfo* b = &tensors_[bIdx];
    TensorInfo* out = &tensors_[outIdx];
    if (a->type == kTfLiteFloat32) {
      size_t count = a->bytes / sizeof(float);
      kernels::Add((float*)a->data, (float*)b->data, (float*)out->data, count);
    }
    return true;
  }

  bool executeMul(int32_t aIdx, int32_t bIdx, int32_t outIdx) {
    TensorInfo* a = &tensors_[aIdx];
    TensorInfo* b = &tensors_[bIdx];
    TensorInfo* out = &tensors_[outIdx];
    if (a->type == kTfLiteFloat32) {
      size_t count = a->bytes / sizeof(float);
      kernels::Mul((float*)a->data, (float*)b->data, (float*)out->data, count);
    }
    return true;
  }

  bool executeSub(int32_t aIdx, int32_t bIdx, int32_t outIdx) {
    TensorInfo* a = &tensors_[aIdx];
    TensorInfo* b = &tensors_[bIdx];
    TensorInfo* out = &tensors_[outIdx];
    if (a->type == kTfLiteFloat32) {
      size_t count = a->bytes / sizeof(float);
      kernels::Sub((float*)a->data, (float*)b->data, (float*)out->data, count);
    }
    return true;
  }
};

#endif // AIDUINO_TFLM_SDCARD_H
