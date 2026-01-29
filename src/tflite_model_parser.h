/*
 * tflite_model_parser.h
 * 
 * Lightweight TFLite FlatBuffer model parser
 * Self-contained - no external dependencies
 */

#ifndef TFLITE_MODEL_PARSER_H
#define TFLITE_MODEL_PARSER_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

// ============================================================================
// FlatBuffer Basic Types
// ============================================================================

typedef uint32_t fb_uoffset_t;
typedef int32_t fb_soffset_t;
typedef uint16_t fb_voffset_t;

// ============================================================================
// TFLite Model Schema Constants
// ============================================================================

#define TFLITE_SCHEMA_VERSION 3
#define TFLITE_FILE_IDENTIFIER "TFL3"

// Built-in operator codes (most common ones)
enum TfLiteBuiltinOperator {
  kTfLiteBuiltinAdd = 0,
  kTfLiteBuiltinAveragePool2d = 1,
  kTfLiteBuiltinConcatenation = 2,
  kTfLiteBuiltinConv2d = 3,
  kTfLiteBuiltinDepthwiseConv2d = 4,
  kTfLiteBuiltinDepthToSpace = 5,
  kTfLiteBuiltinDequantize = 6,
  kTfLiteBuiltinEmbeddingLookup = 7,
  kTfLiteBuiltinFloor = 8,
  kTfLiteBuiltinFullyConnected = 9,
  kTfLiteBuiltinHashtableLookup = 10,
  kTfLiteBuiltinL2Normalization = 11,
  kTfLiteBuiltinL2Pool2d = 12,
  kTfLiteBuiltinLocalResponseNormalization = 13,
  kTfLiteBuiltinLogistic = 14,
  kTfLiteBuiltinLshProjection = 15,
  kTfLiteBuiltinLstm = 16,
  kTfLiteBuiltinMaxPool2d = 17,
  kTfLiteBuiltinMul = 18,
  kTfLiteBuiltinRelu = 19,
  kTfLiteBuiltinReluN1To1 = 20,
  kTfLiteBuiltinRelu6 = 21,
  kTfLiteBuiltinReshape = 22,
  kTfLiteBuiltinResizeBilinear = 23,
  kTfLiteBuiltinRnn = 24,
  kTfLiteBuiltinSoftmax = 25,
  kTfLiteBuiltinSpaceToDepth = 26,
  kTfLiteBuiltinSvdf = 27,
  kTfLiteBuiltinTanh = 28,
  kTfLiteBuiltinCustom = 32,
  kTfLiteBuiltinMean = 40,
  kTfLiteBuiltinSub = 41,
  kTfLiteBuiltinPad = 34,
  kTfLiteBuiltinQuantize = 114
};

// Tensor types in FlatBuffer
enum TfLiteFbTensorType {
  kTfLiteFbFloat32 = 0,
  kTfLiteFbFloat16 = 1,
  kTfLiteFbInt32 = 2,
  kTfLiteFbUInt8 = 3,
  kTfLiteFbInt64 = 4,
  kTfLiteFbString = 5,
  kTfLiteFbBool = 6,
  kTfLiteFbInt16 = 7,
  kTfLiteFbComplex64 = 8,
  kTfLiteFbInt8 = 9
};

// Activation types
enum TfLiteActivation {
  kTfLiteActNone = 0,
  kTfLiteActRelu = 1,
  kTfLiteActReluN1To1 = 2,
  kTfLiteActRelu6 = 3,
  kTfLiteActTanh = 4,
  kTfLiteActSignBit = 5
};

// Padding types
enum TfLitePadding {
  kTfLitePaddingSame = 0,
  kTfLitePaddingValid = 1
};

// ============================================================================
// FlatBuffer Reading Helpers
// ============================================================================

class FlatBufferReader {
public:
  FlatBufferReader(const uint8_t* buffer) : buffer_(buffer) {}
  
  // Read root table offset
  fb_uoffset_t readRootOffset() const {
    return *reinterpret_cast<const fb_uoffset_t*>(buffer_);
  }
  
  // Get pointer to root table
  const uint8_t* getRoot() const {
    return buffer_ + readRootOffset();
  }
  
  // Check file identifier
  bool hasIdentifier(const char* id) const {
    return memcmp(buffer_ + sizeof(fb_uoffset_t), id, 4) == 0;
  }
  
  // Read field from table
  template<typename T>
  T readField(const uint8_t* table, fb_voffset_t field, T defaultVal) const {
    fb_voffset_t offset = getFieldOffset(table, field);
    if (offset == 0) return defaultVal;
    return *reinterpret_cast<const T*>(table + offset);
  }
  
  // Get pointer to nested table/vector
  const uint8_t* readPointer(const uint8_t* table, fb_voffset_t field) const {
    fb_voffset_t offset = getFieldOffset(table, field);
    if (offset == 0) return nullptr;
    const uint8_t* p = table + offset;
    return p + *reinterpret_cast<const fb_uoffset_t*>(p);
  }
  
  // Get vector length
  fb_uoffset_t vectorLength(const uint8_t* vec) const {
    if (!vec) return 0;
    return *reinterpret_cast<const fb_uoffset_t*>(vec);
  }
  
  // Get vector element (for scalars)
  template<typename T>
  T vectorGet(const uint8_t* vec, size_t index) const {
    const T* data = reinterpret_cast<const T*>(vec + sizeof(fb_uoffset_t));
    return data[index];
  }
  
  // Get vector element (for tables/offsets)
  const uint8_t* vectorGetTable(const uint8_t* vec, size_t index) const {
    const uint8_t* elemPtr = vec + sizeof(fb_uoffset_t) + index * sizeof(fb_uoffset_t);
    fb_uoffset_t offset = *reinterpret_cast<const fb_uoffset_t*>(elemPtr);
    return elemPtr + offset;
  }

private:
  const uint8_t* buffer_;
  
  fb_voffset_t getFieldOffset(const uint8_t* table, fb_voffset_t field) const {
    fb_soffset_t vtableOffset = *reinterpret_cast<const fb_soffset_t*>(table);
    const uint8_t* vtable = table - vtableOffset;
    fb_voffset_t vtableSize = *reinterpret_cast<const fb_voffset_t*>(vtable);
    if (field >= vtableSize) return 0;
    return *reinterpret_cast<const fb_voffset_t*>(vtable + field);
  }
};

// ============================================================================
// TFLite Model Reader
// ============================================================================

class TfLiteModelReader {
public:
  TfLiteModelReader() : buffer_(nullptr), reader_(nullptr) {}
  
  bool load(const uint8_t* modelData) {
    buffer_ = modelData;
    reader_ = new FlatBufferReader(buffer_);
    
    // Verify identifier
    if (!reader_->hasIdentifier(TFLITE_FILE_IDENTIFIER)) {
      return false;
    }
    
    model_ = reader_->getRoot();
    
    // Get version
    version_ = reader_->readField<uint32_t>(model_, 4, 0);
    
    // Get subgraphs
    subgraphs_ = reader_->readPointer(model_, 8);
    numSubgraphs_ = reader_->vectorLength(subgraphs_);
    
    // Get buffers
    buffers_ = reader_->readPointer(model_, 12);
    numBuffers_ = reader_->vectorLength(buffers_);
    
    // Get operator codes
    operatorCodes_ = reader_->readPointer(model_, 6);
    numOperatorCodes_ = reader_->vectorLength(operatorCodes_);
    
    return true;
  }
  
  uint32_t getVersion() const { return version_; }
  size_t getNumSubgraphs() const { return numSubgraphs_; }
  size_t getNumBuffers() const { return numBuffers_; }
  
  // Get subgraph
  const uint8_t* getSubgraph(size_t index) const {
    if (index >= numSubgraphs_) return nullptr;
    return reader_->vectorGetTable(subgraphs_, index);
  }
  
  // Get tensors from subgraph
  const uint8_t* getTensors(const uint8_t* subgraph) const {
    return reader_->readPointer(subgraph, 4);
  }
  
  size_t getNumTensors(const uint8_t* subgraph) const {
    const uint8_t* tensors = getTensors(subgraph);
    return reader_->vectorLength(tensors);
  }
  
  // Get tensor at index
  const uint8_t* getTensor(const uint8_t* subgraph, size_t index) const {
    const uint8_t* tensors = getTensors(subgraph);
    return reader_->vectorGetTable(tensors, index);
  }
  
  // Get tensor shape
  const uint8_t* getTensorShape(const uint8_t* tensor) const {
    return reader_->readPointer(tensor, 4);
  }
  
  // Get tensor type
  int8_t getTensorType(const uint8_t* tensor) const {
    return reader_->readField<int8_t>(tensor, 6, 0);
  }
  
  // Get tensor buffer index
  uint32_t getTensorBuffer(const uint8_t* tensor) const {
    return reader_->readField<uint32_t>(tensor, 8, 0);
  }
  
  // Get tensor quantization
  const uint8_t* getTensorQuantization(const uint8_t* tensor) const {
    return reader_->readPointer(tensor, 12);
  }
  
  // Get quantization scale
  float getQuantizationScale(const uint8_t* quant) const {
    if (!quant) return 1.0f;
    const uint8_t* scales = reader_->readPointer(quant, 4);
    if (!scales || reader_->vectorLength(scales) == 0) return 1.0f;
    return reader_->vectorGet<float>(scales, 0);
  }
  
  // Get quantization zero point
  int64_t getQuantizationZeroPoint(const uint8_t* quant) const {
    if (!quant) return 0;
    const uint8_t* zps = reader_->readPointer(quant, 6);
    if (!zps || reader_->vectorLength(zps) == 0) return 0;
    return reader_->vectorGet<int64_t>(zps, 0);
  }
  
  // Get inputs from subgraph
  const uint8_t* getInputs(const uint8_t* subgraph) const {
    return reader_->readPointer(subgraph, 6);
  }
  
  size_t getNumInputs(const uint8_t* subgraph) const {
    const uint8_t* inputs = getInputs(subgraph);
    return reader_->vectorLength(inputs);
  }
  
  int32_t getInput(const uint8_t* subgraph, size_t index) const {
    const uint8_t* inputs = getInputs(subgraph);
    return reader_->vectorGet<int32_t>(inputs, index);
  }
  
  // Get outputs from subgraph
  const uint8_t* getOutputs(const uint8_t* subgraph) const {
    return reader_->readPointer(subgraph, 8);
  }
  
  size_t getNumOutputs(const uint8_t* subgraph) const {
    const uint8_t* outputs = getOutputs(subgraph);
    return reader_->vectorLength(outputs);
  }
  
  int32_t getOutput(const uint8_t* subgraph, size_t index) const {
    const uint8_t* outputs = getOutputs(subgraph);
    return reader_->vectorGet<int32_t>(outputs, index);
  }
  
  // Get operators from subgraph
  const uint8_t* getOperators(const uint8_t* subgraph) const {
    return reader_->readPointer(subgraph, 10);
  }
  
  size_t getNumOperators(const uint8_t* subgraph) const {
    const uint8_t* ops = getOperators(subgraph);
    return reader_->vectorLength(ops);
  }
  
  const uint8_t* getOperator(const uint8_t* subgraph, size_t index) const {
    const uint8_t* ops = getOperators(subgraph);
    return reader_->vectorGetTable(ops, index);
  }
  
  // Get operator code index
  uint32_t getOperatorOpcodeIndex(const uint8_t* op) const {
    return reader_->readField<uint32_t>(op, 4, 0);
  }
  
  // Get builtin code from operator code
  int32_t getBuiltinCode(size_t opcodeIndex) const {
    if (opcodeIndex >= numOperatorCodes_) return -1;
    const uint8_t* opcode = reader_->vectorGetTable(operatorCodes_, opcodeIndex);
    // Try deprecated_builtin_code first (field 4), then builtin_code (field 8)
    int8_t deprecated = reader_->readField<int8_t>(opcode, 4, 0);
    int32_t builtin = reader_->readField<int32_t>(opcode, 8, deprecated);
    return builtin;
  }
  
  // Get buffer data
  const uint8_t* getBufferData(size_t bufferIndex, size_t* outSize) const {
    if (bufferIndex >= numBuffers_) {
      *outSize = 0;
      return nullptr;
    }
    const uint8_t* buffer = reader_->vectorGetTable(buffers_, bufferIndex);
    const uint8_t* data = reader_->readPointer(buffer, 4);
    *outSize = reader_->vectorLength(data);
    if (*outSize == 0) return nullptr;
    return data + sizeof(fb_uoffset_t);  // Skip length prefix
  }
  
  // Get operator inputs
  const uint8_t* getOpInputs(const uint8_t* op) const {
    return reader_->readPointer(op, 6);
  }
  
  size_t getNumOpInputs(const uint8_t* op) const {
    return reader_->vectorLength(getOpInputs(op));
  }
  
  int32_t getOpInput(const uint8_t* op, size_t index) const {
    return reader_->vectorGet<int32_t>(getOpInputs(op), index);
  }
  
  // Get operator outputs
  const uint8_t* getOpOutputs(const uint8_t* op) const {
    return reader_->readPointer(op, 8);
  }
  
  size_t getNumOpOutputs(const uint8_t* op) const {
    return reader_->vectorLength(getOpOutputs(op));
  }
  
  int32_t getOpOutput(const uint8_t* op, size_t index) const {
    return reader_->vectorGet<int32_t>(getOpOutputs(op), index);
  }

private:
  const uint8_t* buffer_;
  FlatBufferReader* reader_;
  const uint8_t* model_;
  uint32_t version_;
  const uint8_t* subgraphs_;
  size_t numSubgraphs_;
  const uint8_t* buffers_;
  size_t numBuffers_;
  const uint8_t* operatorCodes_;
  size_t numOperatorCodes_;
};

#endif // TFLITE_MODEL_PARSER_H
