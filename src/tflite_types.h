/*
 * tflite_types.h
 * 
 * Self-contained TensorFlow Lite type definitions
 * No external dependencies
 */

#ifndef TFLITE_TYPES_H
#define TFLITE_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Status Codes
// ============================================================================

typedef enum TfLiteStatus {
  kTfLiteOk = 0,
  kTfLiteError = 1,
  kTfLiteDelegateError = 2,
  kTfLiteApplicationError = 3,
  kTfLiteDelegateDataNotFound = 4,
  kTfLiteDelegateDataWriteError = 5,
  kTfLiteDelegateDataReadError = 6,
  kTfLiteUnresolvedOps = 7,
  kTfLiteCancelled = 8
} TfLiteStatus;

// ============================================================================
// Data Types
// ============================================================================

typedef enum TfLiteType {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
  kTfLiteBool = 6,
  kTfLiteInt16 = 7,
  kTfLiteComplex64 = 8,
  kTfLiteInt8 = 9,
  kTfLiteFloat16 = 10,
  kTfLiteFloat64 = 11,
  kTfLiteComplex128 = 12,
  kTfLiteUInt64 = 13,
  kTfLiteResource = 14,
  kTfLiteVariant = 15,
  kTfLiteUInt32 = 16,
  kTfLiteUInt16 = 17,
  kTfLiteInt4 = 18
} TfLiteType;

// ============================================================================
// Allocation Types
// ============================================================================

typedef enum TfLiteAllocationType {
  kTfLiteMemNone = 0,
  kTfLiteMmapRo = 1,
  kTfLiteArenaRw = 2,
  kTfLiteArenaRwPersistent = 3,
  kTfLiteDynamic = 4,
  kTfLitePersistentRo = 5,
  kTfLiteCustom = 6
} TfLiteAllocationType;

// ============================================================================
// Quantization
// ============================================================================

typedef enum TfLiteQuantizationType {
  kTfLiteNoQuantization = 0,
  kTfLiteAffineQuantization = 1
} TfLiteQuantizationType;

typedef struct TfLiteQuantizationParams {
  float scale;
  int32_t zero_point;
} TfLiteQuantizationParams;

// ============================================================================
// Array Structures
// ============================================================================

typedef struct TfLiteIntArray {
  int size;
  int data[];
} TfLiteIntArray;

typedef struct TfLiteFloatArray {
  int size;
  float data[];
} TfLiteFloatArray;

// ============================================================================
// Tensor Data Union
// ============================================================================

typedef union TfLitePtrUnion {
  int32_t* i32;
  uint32_t* u32;
  int64_t* i64;
  uint64_t* u64;
  float* f;
  double* f64;
  int16_t* i16;
  uint16_t* ui16;
  int8_t* int8;
  uint8_t* uint8;
  bool* b;
  char* raw;
  const char* raw_const;
  void* data;
} TfLitePtrUnion;

// ============================================================================
// Tensor Structure
// ============================================================================

typedef struct TfLiteTensor {
  TfLiteType type;
  TfLitePtrUnion data;
  TfLiteIntArray* dims;
  TfLiteQuantizationParams params;
  TfLiteAllocationType allocation_type;
  size_t bytes;
  const void* allocation;
  const char* name;
  void* delegate;
  int buffer_handle;
  bool data_is_stale;
  bool is_variable;
} TfLiteTensor;

// ============================================================================
// Utility Functions
// ============================================================================

static inline size_t TfLiteTypeGetSize(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32: return sizeof(float);
    case kTfLiteInt32: return sizeof(int32_t);
    case kTfLiteUInt8: return sizeof(uint8_t);
    case kTfLiteInt64: return sizeof(int64_t);
    case kTfLiteBool: return sizeof(bool);
    case kTfLiteInt16: return sizeof(int16_t);
    case kTfLiteInt8: return sizeof(int8_t);
    case kTfLiteFloat16: return sizeof(uint16_t);
    case kTfLiteFloat64: return sizeof(double);
    case kTfLiteUInt64: return sizeof(uint64_t);
    case kTfLiteUInt32: return sizeof(uint32_t);
    case kTfLiteUInt16: return sizeof(uint16_t);
    default: return 0;
  }
}

#ifdef __cplusplus
}
#endif

#endif // TFLITE_TYPES_H
