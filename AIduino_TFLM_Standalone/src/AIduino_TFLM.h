/*
 * AIduino_TFLM.h
 * 
 * SELF-CONTAINED TensorFlow Lite Micro Runtime for STM32F407VGT6
 * 
 * NO external TensorFlow downloads required!
 * All dependencies are included.
 * 
 * USAGE:
 * ------
 * #include <AIduino_TFLM.h>
 * #include "your_model.h"  // Convert with: xxd -i model.tflite > your_model.h
 * 
 * TFLiteRuntime runtime;
 * 
 * void setup() {
 *   Serial.begin(115200);
 *   runtime.begin(your_model_tflite, 30*1024);  // model, arena size
 * }
 * 
 * void loop() {
 *   float input[] = {1.0, 2.0, 3.0};
 *   runtime.setInput(input, 3);
 *   runtime.invoke();
 *   int result = runtime.getOutputMaxIndex();
 *   Serial.println(result);
 * }
 */

#ifndef AIDUINO_TFLM_H
#define AIDUINO_TFLM_H

#include <Arduino.h>
#include "tflite_types.h"
#include "tflite_model_parser.h"
#include "tflite_kernels.h"

// ============================================================================
// Configuration
// ============================================================================

#define TFLM_MAX_TENSORS 64
#define TFLM_MAX_OPERATORS 32
#define TFLM_MAX_DIMS 8

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
// TFLite Runtime Class
// ============================================================================

class TFLiteRuntime {
public:
  TFLiteRuntime() : 
    initialized_(false),
    arena_(nullptr),
    arena_size_(0),
    arena_used_(0),
    num_tensors_(0),
    num_inputs_(0),
    num_outputs_(0),
    inference_time_us_(0) {}
  
  ~TFLiteRuntime() {
    if (arena_) {
      delete[] arena_;
    }
  }

  /**
   * Initialize the runtime with a TFLite model
   * 
   * @param model_data   Model bytes from xxd -i conversion
   * @param arena_size   Memory arena size in bytes (start with 20*1024, increase if needed)
   * @return true on success
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
    
    // Get first subgraph (main graph)
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
    Serial.println("=== TFLite Model Loaded ===");
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
    
    // Print input info
    if (num_inputs_ > 0) {
      TensorInfo* input = &tensors_[input_indices_[0]];
      Serial.print("Input shape: [");
      for (int i = 0; i < input->num_dims; i++) {
        Serial.print(input->dims[i]);
        if (i < input->num_dims - 1) Serial.print(", ");
      }
      Serial.print("] type=");
      Serial.println(input->type == kTfLiteFloat32 ? "float32" : "int8");
    }
    
    // Print output info
    if (num_outputs_ > 0) {
      TensorInfo* output = &tensors_[output_indices_[0]];
      Serial.print("Output shape: [");
      for (int i = 0; i < output->num_dims; i++) {
        Serial.print(output->dims[i]);
        if (i < output->num_dims - 1) Serial.print(", ");
      }
      Serial.print("] type=");
      Serial.println(output->type == kTfLiteFloat32 ? "float32" : "int8");
    }
    
    Serial.println("===========================");
    
    return true;
  }

  /**
   * Set input tensor data (float)
   */
  bool setInput(const float* data, size_t count, int input_index = 0) {
    if (!initialized_ || input_index >= (int)num_inputs_) return false;
    
    TensorInfo* input = &tensors_[input_indices_[input_index]];
    
    if (input->type == kTfLiteFloat32) {
      memcpy(input->data, data, count * sizeof(float));
    } else if (input->type == kTfLiteInt8) {
      // Quantize
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
   * Set input tensor data (int8 - for quantized models)
   */
  bool setInputInt8(const int8_t* data, size_t count, int input_index = 0) {
    if (!initialized_ || input_index >= (int)num_inputs_) return false;
    
    TensorInfo* input = &tensors_[input_indices_[input_index]];
    memcpy(input->data, data, count);
    return true;
  }

  /**
   * Run inference
   */
  bool invoke() {
    if (!initialized_) return false;
    
    uint32_t start = micros();
    
    // Execute operators in order
    size_t numOps = modelReader_.getNumOperators(subgraph_);
    
    for (size_t i = 0; i < numOps; i++) {
      const uint8_t* op = modelReader_.getOperator(subgraph_, i);
      
      if (!executeOperator(op)) {
        Serial.print("ERROR: Failed to execute operator ");
        Serial.println(i);
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
      // Dequantize to static buffer
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

  /**
   * Get number of elements in input tensor
   */
  int getInputSize(int input_index = 0) {
    if (!initialized_ || input_index >= (int)num_inputs_) return 0;
    TensorInfo* t = &tensors_[input_indices_[input_index]];
    if (t->type == kTfLiteFloat32) return t->bytes / sizeof(float);
    return t->bytes;
  }

  /**
   * Get number of elements in output tensor
   */
  int getOutputSize(int output_index = 0) {
    if (!initialized_ || output_index >= (int)num_outputs_) return 0;
    TensorInfo* t = &tensors_[output_indices_[output_index]];
    if (t->type == kTfLiteFloat32) return t->bytes / sizeof(float);
    return t->bytes;
  }

  /**
   * Get inference time in microseconds
   */
  uint32_t getInferenceTimeUs() { return inference_time_us_; }

  /**
   * Get arena memory used
   */
  size_t getArenaUsed() { return arena_used_; }

  /**
   * Check if runtime is ready
   */
  bool isReady() { return initialized_; }

private:
  bool initialized_;
  
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

  // Allocate memory from arena
  void* allocateArena(size_t bytes) {
    // Align to 4 bytes
    bytes = (bytes + 3) & ~3;
    
    if (arena_used_ + bytes > arena_size_) {
      return nullptr;
    }
    
    void* ptr = arena_ + arena_used_;
    arena_used_ += bytes;
    return ptr;
  }

  // Convert TFLite FB type to runtime type
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

  // Allocate all tensors
  bool allocateTensors() {
    num_tensors_ = modelReader_.getNumTensors(subgraph_);
    
    if (num_tensors_ > TFLM_MAX_TENSORS) {
      Serial.println("ERROR: Too many tensors");
      return false;
    }
    
    for (int i = 0; i < num_tensors_; i++) {
      const uint8_t* tensor = modelReader_.getTensor(subgraph_, i);
      TensorInfo* info = &tensors_[i];
      
      // Get type
      info->type = convertType(modelReader_.getTensorType(tensor));
      
      // Get shape
      const uint8_t* shape = modelReader_.getTensorShape(tensor);
      info->num_dims = shape ? modelReader_.vectorLength(shape) : 0;
      
      size_t totalElements = 1;
      for (int d = 0; d < info->num_dims && d < TFLM_MAX_DIMS; d++) {
        info->dims[d] = modelReader_.vectorGet<int32_t>(shape, d);
        totalElements *= info->dims[d];
      }
      
      // Calculate bytes
      info->bytes = totalElements * TfLiteTypeGetSize(info->type);
      
      // Get quantization parameters
      const uint8_t* quant = modelReader_.getTensorQuantization(tensor);
      info->scale = modelReader_.getQuantizationScale(quant);
      info->zero_point = (int32_t)modelReader_.getQuantizationZeroPoint(quant);
      
      // Get buffer
      info->buffer_index = modelReader_.getTensorBuffer(tensor);
      
      // Allocate or point to buffer
      size_t bufferSize = 0;
      const uint8_t* bufferData = modelReader_.getBufferData(info->buffer_index, &bufferSize);
      
      if (bufferData && bufferSize > 0) {
        // This is a constant (weights/bias) - point directly to model data
        info->data = (void*)bufferData;
        info->is_variable = false;
      } else {
        // This needs runtime allocation (activations)
        info->data = allocateArena(info->bytes);
        if (!info->data) {
          Serial.print("ERROR: Cannot allocate tensor ");
          Serial.print(i);
          Serial.print(" (");
          Serial.print(info->bytes);
          Serial.println(" bytes)");
          return false;
        }
        memset(info->data, 0, info->bytes);
        info->is_variable = true;
      }
    }
    
    return true;
  }

  // Execute a single operator
  bool executeOperator(const uint8_t* op) {
    // Get operator code
    uint32_t opcodeIdx = modelReader_.getOperatorOpcodeIndex(op);
    int32_t builtinCode = modelReader_.getBuiltinCode(opcodeIdx);
    
    // Get inputs and outputs
    size_t numInputs = modelReader_.getNumOpInputs(op);
    size_t numOutputs = modelReader_.getNumOpOutputs(op);
    
    int32_t inputIdx[8], outputIdx[8];
    for (size_t i = 0; i < numInputs && i < 8; i++) {
      inputIdx[i] = modelReader_.getOpInput(op, i);
    }
    for (size_t i = 0; i < numOutputs && i < 8; i++) {
      outputIdx[i] = modelReader_.getOpOutput(op, i);
    }
    
    // Execute based on operator type
    switch (builtinCode) {
      case kTfLiteBuiltinFullyConnected:
        return executeFullyConnected(inputIdx, outputIdx, numInputs, op);
        
      case kTfLiteBuiltinConv2d:
        return executeConv2D(inputIdx, outputIdx, numInputs, op);
        
      case kTfLiteBuiltinDepthwiseConv2d:
        return executeDepthwiseConv2D(inputIdx, outputIdx, numInputs, op);
        
      case kTfLiteBuiltinMaxPool2d:
        return executeMaxPool2D(inputIdx, outputIdx, op);
        
      case kTfLiteBuiltinAveragePool2d:
        return executeAveragePool2D(inputIdx, outputIdx, op);
        
      case kTfLiteBuiltinRelu:
        return executeRelu(inputIdx[0], outputIdx[0]);
        
      case kTfLiteBuiltinRelu6:
        return executeRelu6(inputIdx[0], outputIdx[0]);
        
      case kTfLiteBuiltinLogistic:
        return executeLogistic(inputIdx[0], outputIdx[0]);
        
      case kTfLiteBuiltinTanh:
        return executeTanh(inputIdx[0], outputIdx[0]);
        
      case kTfLiteBuiltinSoftmax:
        return executeSoftmax(inputIdx[0], outputIdx[0]);
        
      case kTfLiteBuiltinReshape:
        return executeReshape(inputIdx[0], outputIdx[0]);
        
      case kTfLiteBuiltinAdd:
        return executeAdd(inputIdx[0], inputIdx[1], outputIdx[0]);
        
      case kTfLiteBuiltinMul:
        return executeMul(inputIdx[0], inputIdx[1], outputIdx[0]);
        
      case kTfLiteBuiltinSub:
        return executeSub(inputIdx[0], inputIdx[1], outputIdx[0]);
        
      case kTfLiteBuiltinQuantize:
        return executeQuantize(inputIdx[0], outputIdx[0]);
        
      case kTfLiteBuiltinDequantize:
        return executeDequantize(inputIdx[0], outputIdx[0]);
        
      case kTfLiteBuiltinMean:
        return executeMean(inputIdx[0], outputIdx[0]);
        
      default:
        Serial.print("WARNING: Unknown op ");
        Serial.println(builtinCode);
        // Try to just copy input to output as fallback
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

  // ========================================================================
  // Operator Implementations
  // ========================================================================
  
  bool executeFullyConnected(int32_t* inputs, int32_t* outputs, size_t numInputs, const uint8_t* op) {
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
        input_size, output_size, 0);  // TODO: get activation from options
    } else if (input->type == kTfLiteInt8) {
      kernels::FullyConnectedInt8(
        (int8_t*)input->data,
        (int8_t*)weights->data,
        bias ? (int32_t*)bias->data : nullptr,
        (int8_t*)output->data,
        input_size, output_size,
        -input->zero_point, -weights->zero_point, output->zero_point,
        (int32_t)(input->scale * weights->scale / output->scale * (1 << 30)),
        30, 0);
    }
    
    return true;
  }
  
  bool executeConv2D(int32_t* inputs, int32_t* outputs, size_t numInputs, const uint8_t* op) {
    TensorInfo* input = &tensors_[inputs[0]];
    TensorInfo* weights = &tensors_[inputs[1]];
    TensorInfo* bias = numInputs > 2 && inputs[2] >= 0 ? &tensors_[inputs[2]] : nullptr;
    TensorInfo* output = &tensors_[outputs[0]];
    
    if (input->type != kTfLiteFloat32) {
      // Simplified: only float for now
      return false;
    }
    
    // Get dimensions [batch, height, width, channels]
    int batch = input->dims[0];
    int in_h = input->dims[1];
    int in_w = input->dims[2];
    int in_c = input->dims[3];
    int out_c = weights->dims[0];
    int filter_h = weights->dims[1];
    int filter_w = weights->dims[2];
    
    // TODO: Parse stride/padding from op options
    int stride = 1;
    int pad = (filter_h - 1) / 2;  // SAME padding approximation
    
    kernels::Conv2D(
      (float*)input->data, (float*)weights->data,
      bias ? (float*)bias->data : nullptr,
      (float*)output->data,
      batch, in_h, in_w, in_c,
      out_c, filter_h, filter_w,
      stride, stride, pad, pad, 0);
    
    return true;
  }
  
  bool executeDepthwiseConv2D(int32_t* inputs, int32_t* outputs, size_t numInputs, const uint8_t* op) {
    TensorInfo* input = &tensors_[inputs[0]];
    TensorInfo* weights = &tensors_[inputs[1]];
    TensorInfo* bias = numInputs > 2 && inputs[2] >= 0 ? &tensors_[inputs[2]] : nullptr;
    TensorInfo* output = &tensors_[outputs[0]];
    
    if (input->type != kTfLiteFloat32) return false;
    
    int batch = input->dims[0];
    int in_h = input->dims[1];
    int in_w = input->dims[2];
    int in_c = input->dims[3];
    int filter_h = weights->dims[1];
    int filter_w = weights->dims[2];
    int depth_mult = weights->dims[3] / in_c;
    
    int stride = 1;
    int pad = (filter_h - 1) / 2;
    
    kernels::DepthwiseConv2D(
      (float*)input->data, (float*)weights->data,
      bias ? (float*)bias->data : nullptr,
      (float*)output->data,
      batch, in_h, in_w, in_c,
      filter_h, filter_w, depth_mult,
      stride, stride, pad, pad, 0);
    
    return true;
  }
  
  bool executeMaxPool2D(int32_t* inputs, int32_t* outputs, const uint8_t* op) {
    TensorInfo* input = &tensors_[inputs[0]];
    TensorInfo* output = &tensors_[outputs[0]];
    
    if (input->type != kTfLiteFloat32) return false;
    
    int batch = input->dims[0];
    int in_h = input->dims[1];
    int in_w = input->dims[2];
    int channels = input->dims[3];
    
    // Default 2x2 pool with stride 2
    kernels::MaxPool2D(
      (float*)input->data, (float*)output->data,
      batch, in_h, in_w, channels,
      2, 2, 2, 2, 0, 0);
    
    return true;
  }
  
  bool executeAveragePool2D(int32_t* inputs, int32_t* outputs, const uint8_t* op) {
    TensorInfo* input = &tensors_[inputs[0]];
    TensorInfo* output = &tensors_[outputs[0]];
    
    if (input->type != kTfLiteFloat32) return false;
    
    int batch = input->dims[0];
    int in_h = input->dims[1];
    int in_w = input->dims[2];
    int channels = input->dims[3];
    
    kernels::AveragePool2D(
      (float*)input->data, (float*)output->data,
      batch, in_h, in_w, channels,
      2, 2, 2, 2, 0, 0);
    
    return true;
  }
  
  bool executeRelu(int32_t inputIdx, int32_t outputIdx) {
    TensorInfo* input = &tensors_[inputIdx];
    TensorInfo* output = &tensors_[outputIdx];
    size_t count = input->bytes / TfLiteTypeGetSize(input->type);
    
    if (input->type == kTfLiteFloat32) {
      kernels::Relu((float*)input->data, (float*)output->data, count);
    } else if (input->type == kTfLiteInt8) {
      kernels::ReluInt8((int8_t*)input->data, (int8_t*)output->data, count, input->zero_point);
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
  
  bool executeSoftmax(int32_t inputIdx, int32_t outputIdx) {
    TensorInfo* input = &tensors_[inputIdx];
    TensorInfo* output = &tensors_[outputIdx];
    
    if (input->type == kTfLiteFloat32) {
      size_t count = input->bytes / sizeof(float);
      kernels::Softmax((float*)input->data, (float*)output->data, count);
    } else if (input->type == kTfLiteInt8) {
      size_t count = input->bytes;
      kernels::SoftmaxInt8(
        (int8_t*)input->data, (int8_t*)output->data, count,
        input->scale, input->zero_point,
        output->scale, output->zero_point);
    }
    return true;
  }
  
  bool executeReshape(int32_t inputIdx, int32_t outputIdx) {
    TensorInfo* input = &tensors_[inputIdx];
    TensorInfo* output = &tensors_[outputIdx];
    kernels::Reshape(input->data, output->data, input->bytes);
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
  
  bool executeQuantize(int32_t inputIdx, int32_t outputIdx) {
    TensorInfo* input = &tensors_[inputIdx];
    TensorInfo* output = &tensors_[outputIdx];
    size_t count = input->bytes / sizeof(float);
    
    kernels::Quantize((float*)input->data, (int8_t*)output->data, count,
                      output->scale, output->zero_point);
    return true;
  }
  
  bool executeDequantize(int32_t inputIdx, int32_t outputIdx) {
    TensorInfo* input = &tensors_[inputIdx];
    TensorInfo* output = &tensors_[outputIdx];
    size_t count = input->bytes;
    
    kernels::Dequantize((int8_t*)input->data, (float*)output->data, count,
                        input->scale, input->zero_point);
    return true;
  }
  
  bool executeMean(int32_t inputIdx, int32_t outputIdx) {
    TensorInfo* input = &tensors_[inputIdx];
    TensorInfo* output = &tensors_[outputIdx];
    
    if (input->type == kTfLiteFloat32) {
      kernels::Mean((float*)input->data, (float*)output->data,
                    input->dims, input->num_dims, nullptr, 0, false);
    }
    return true;
  }
};

#endif // AIDUINO_TFLM_H
