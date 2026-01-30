/*
 * tflite_kernels.h
 * 
 * Self-contained kernel implementations for common TFLite operations
 * Optimized for ARM Cortex-M4
 */

#ifndef TFLITE_KERNELS_H
#define TFLITE_KERNELS_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include "tflite_types.h"

// ============================================================================
// Activation Functions
// ============================================================================

namespace kernels {

// ReLU: max(0, x)
inline void Relu(const float* input, float* output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = input[i] > 0.0f ? input[i] : 0.0f;
  }
}

inline void ReluInt8(const int8_t* input, int8_t* output, size_t size, 
                     int32_t zero_point) {
  for (size_t i = 0; i < size; i++) {
    output[i] = input[i] > zero_point ? input[i] : (int8_t)zero_point;
  }
}

// ReLU6: min(6, max(0, x))
inline void Relu6(const float* input, float* output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    float val = input[i];
    if (val < 0.0f) val = 0.0f;
    if (val > 6.0f) val = 6.0f;
    output[i] = val;
  }
}

// Logistic/Sigmoid: 1 / (1 + exp(-x))
inline void Logistic(const float* input, float* output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = 1.0f / (1.0f + expf(-input[i]));
  }
}

// Tanh
inline void Tanh(const float* input, float* output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = tanhf(input[i]);
  }
}

// Softmax
inline void Softmax(const float* input, float* output, size_t size) {
  // Find max for numerical stability
  float maxVal = input[0];
  for (size_t i = 1; i < size; i++) {
    if (input[i] > maxVal) maxVal = input[i];
  }
  
  // Compute exp and sum
  float sum = 0.0f;
  for (size_t i = 0; i < size; i++) {
    output[i] = expf(input[i] - maxVal);
    sum += output[i];
  }
  
  // Normalize
  float invSum = 1.0f / sum;
  for (size_t i = 0; i < size; i++) {
    output[i] *= invSum;
  }
}

inline void SoftmaxInt8(const int8_t* input, int8_t* output, size_t size,
                        float input_scale, int32_t input_zp,
                        float output_scale, int32_t output_zp) {
  // Dequantize, compute softmax, requantize
  float temp[256];  // Max softmax size
  if (size > 256) size = 256;
  
  // Dequantize
  for (size_t i = 0; i < size; i++) {
    temp[i] = input_scale * (input[i] - input_zp);
  }
  
  // Softmax
  float maxVal = temp[0];
  for (size_t i = 1; i < size; i++) {
    if (temp[i] > maxVal) maxVal = temp[i];
  }
  
  float sum = 0.0f;
  for (size_t i = 0; i < size; i++) {
    temp[i] = expf(temp[i] - maxVal);
    sum += temp[i];
  }
  
  // Normalize and requantize
  float invSum = 1.0f / sum;
  for (size_t i = 0; i < size; i++) {
    float val = temp[i] * invSum;
    int32_t q = (int32_t)roundf(val / output_scale) + output_zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    output[i] = (int8_t)q;
  }
}

// ============================================================================
// Fully Connected Layer
// ============================================================================

inline void FullyConnected(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int input_size,
    int output_size,
    int activation) {
  
  for (int o = 0; o < output_size; o++) {
    float sum = bias ? bias[o] : 0.0f;
    
    for (int i = 0; i < input_size; i++) {
      sum += input[i] * weights[o * input_size + i];
    }
    
    // Apply activation
    switch (activation) {
      case 1: // RELU
        sum = sum > 0.0f ? sum : 0.0f;
        break;
      case 3: // RELU6
        sum = sum < 0.0f ? 0.0f : (sum > 6.0f ? 6.0f : sum);
        break;
      case 4: // TANH
        sum = tanhf(sum);
        break;
    }
    
    output[o] = sum;
  }
}

inline void FullyConnectedInt8(
    const int8_t* input,
    const int8_t* weights,
    const int32_t* bias,
    int8_t* output,
    int input_size,
    int output_size,
    int32_t input_offset,
    int32_t weights_offset,
    int32_t output_offset,
    int32_t output_multiplier,
    int output_shift,
    int activation) {
  
  for (int o = 0; o < output_size; o++) {
    int32_t acc = bias ? bias[o] : 0;
    
    for (int i = 0; i < input_size; i++) {
      int32_t input_val = input[i] + input_offset;
      int32_t weight_val = weights[o * input_size + i] + weights_offset;
      acc += input_val * weight_val;
    }
    
    // Quantized multiply and shift
    acc = (int32_t)(((int64_t)acc * output_multiplier) >> (31 - output_shift));
    acc += output_offset;
    
    // Clamp to int8 range
    if (acc < -128) acc = -128;
    if (acc > 127) acc = 127;
    
    output[o] = (int8_t)acc;
  }
}

// ============================================================================
// Convolution 2D
// ============================================================================

inline void Conv2D(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch, int in_h, int in_w, int in_c,
    int out_c, int filter_h, int filter_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int activation) {
  
  int out_h = (in_h + 2 * pad_h - filter_h) / stride_h + 1;
  int out_w = (in_w + 2 * pad_w - filter_w) / stride_w + 1;
  
  for (int b = 0; b < batch; b++) {
    for (int oc = 0; oc < out_c; oc++) {
      for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
          float sum = bias ? bias[oc] : 0.0f;
          
          for (int fh = 0; fh < filter_h; fh++) {
            for (int fw = 0; fw < filter_w; fw++) {
              int ih = oh * stride_h - pad_h + fh;
              int iw = ow * stride_w - pad_w + fw;
              
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                for (int ic = 0; ic < in_c; ic++) {
                  int input_idx = ((b * in_h + ih) * in_w + iw) * in_c + ic;
                  int weight_idx = ((oc * filter_h + fh) * filter_w + fw) * in_c + ic;
                  sum += input[input_idx] * weights[weight_idx];
                }
              }
            }
          }
          
          // Apply activation
          switch (activation) {
            case 1: sum = sum > 0.0f ? sum : 0.0f; break;
            case 3: sum = sum < 0.0f ? 0.0f : (sum > 6.0f ? 6.0f : sum); break;
          }
          
          int output_idx = ((b * out_h + oh) * out_w + ow) * out_c + oc;
          output[output_idx] = sum;
        }
      }
    }
  }
}

// ============================================================================
// Depthwise Convolution 2D
// ============================================================================

inline void DepthwiseConv2D(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch, int in_h, int in_w, int in_c,
    int filter_h, int filter_w, int depth_multiplier,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int activation) {
  
  int out_h = (in_h + 2 * pad_h - filter_h) / stride_h + 1;
  int out_w = (in_w + 2 * pad_w - filter_w) / stride_w + 1;
  int out_c = in_c * depth_multiplier;
  
  for (int b = 0; b < batch; b++) {
    for (int ic = 0; ic < in_c; ic++) {
      for (int dm = 0; dm < depth_multiplier; dm++) {
        int oc = ic * depth_multiplier + dm;
        
        for (int oh = 0; oh < out_h; oh++) {
          for (int ow = 0; ow < out_w; ow++) {
            float sum = bias ? bias[oc] : 0.0f;
            
            for (int fh = 0; fh < filter_h; fh++) {
              for (int fw = 0; fw < filter_w; fw++) {
                int ih = oh * stride_h - pad_h + fh;
                int iw = ow * stride_w - pad_w + fw;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                  int input_idx = ((b * in_h + ih) * in_w + iw) * in_c + ic;
                  int weight_idx = (ic * depth_multiplier + dm) * filter_h * filter_w 
                                   + fh * filter_w + fw;
                  sum += input[input_idx] * weights[weight_idx];
                }
              }
            }
            
            // Activation
            switch (activation) {
              case 1: sum = sum > 0.0f ? sum : 0.0f; break;
              case 3: sum = sum < 0.0f ? 0.0f : (sum > 6.0f ? 6.0f : sum); break;
            }
            
            int output_idx = ((b * out_h + oh) * out_w + ow) * out_c + oc;
            output[output_idx] = sum;
          }
        }
      }
    }
  }
}

// ============================================================================
// Pooling Operations
// ============================================================================

inline void MaxPool2D(
    const float* input, float* output,
    int batch, int in_h, int in_w, int channels,
    int filter_h, int filter_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {
  
  int out_h = (in_h + 2 * pad_h - filter_h) / stride_h + 1;
  int out_w = (in_w + 2 * pad_w - filter_w) / stride_w + 1;
  
  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channels; c++) {
      for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
          float maxVal = -1e30f;
          
          for (int fh = 0; fh < filter_h; fh++) {
            for (int fw = 0; fw < filter_w; fw++) {
              int ih = oh * stride_h - pad_h + fh;
              int iw = ow * stride_w - pad_w + fw;
              
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int idx = ((b * in_h + ih) * in_w + iw) * channels + c;
                if (input[idx] > maxVal) maxVal = input[idx];
              }
            }
          }
          
          int out_idx = ((b * out_h + oh) * out_w + ow) * channels + c;
          output[out_idx] = maxVal;
        }
      }
    }
  }
}

inline void AveragePool2D(
    const float* input, float* output,
    int batch, int in_h, int in_w, int channels,
    int filter_h, int filter_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {
  
  int out_h = (in_h + 2 * pad_h - filter_h) / stride_h + 1;
  int out_w = (in_w + 2 * pad_w - filter_w) / stride_w + 1;
  
  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channels; c++) {
      for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
          float sum = 0.0f;
          int count = 0;
          
          for (int fh = 0; fh < filter_h; fh++) {
            for (int fw = 0; fw < filter_w; fw++) {
              int ih = oh * stride_h - pad_h + fh;
              int iw = ow * stride_w - pad_w + fw;
              
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int idx = ((b * in_h + ih) * in_w + iw) * channels + c;
                sum += input[idx];
                count++;
              }
            }
          }
          
          int out_idx = ((b * out_h + oh) * out_w + ow) * channels + c;
          output[out_idx] = count > 0 ? sum / count : 0.0f;
        }
      }
    }
  }
}

// ============================================================================
// Element-wise Operations
// ============================================================================

inline void Add(const float* a, const float* b, float* output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = a[i] + b[i];
  }
}

inline void Mul(const float* a, const float* b, float* output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = a[i] * b[i];
  }
}

inline void Sub(const float* a, const float* b, float* output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = a[i] - b[i];
  }
}

// ============================================================================
// Reshape (just copy, shape is metadata only)
// ============================================================================

inline void Reshape(const void* input, void* output, size_t bytes) {
  memcpy(output, input, bytes);
}

// ============================================================================
// Quantization Operations
// ============================================================================

inline void Quantize(const float* input, int8_t* output, size_t size,
                     float scale, int32_t zero_point) {
  for (size_t i = 0; i < size; i++) {
    int32_t q = (int32_t)roundf(input[i] / scale) + zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    output[i] = (int8_t)q;
  }
}

inline void Dequantize(const int8_t* input, float* output, size_t size,
                       float scale, int32_t zero_point) {
  for (size_t i = 0; i < size; i++) {
    output[i] = scale * (input[i] - zero_point);
  }
}

// ============================================================================
// Mean / Reduce Operations
// ============================================================================

inline void Mean(const float* input, float* output,
                 const int* input_dims, int num_dims,
                 const int* axis, int num_axis, bool keep_dims) {
  // Simplified: assumes reducing all spatial dimensions for typical use
  // Full implementation would handle arbitrary axes
  size_t total_size = 1;
  for (int i = 0; i < num_dims; i++) {
    total_size *= input_dims[i];
  }
  
  float sum = 0.0f;
  for (size_t i = 0; i < total_size; i++) {
    sum += input[i];
  }
  
  output[0] = sum / total_size;
}

}  // namespace kernels

#endif // TFLITE_KERNELS_H
