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
