# AIduino TensorFlow Lite Micro Library

A TensorFlow Lite Micro library specifically designed for the AIduino board (STM32F407VGT6 ARM Cortex-M4). This library enables running machine learning models directly on your AIduino microcontroller for edge AI applications.

## Features

- **Optimized for STM32F4**: Takes advantage of the ARM Cortex-M4 FPU and DSP instructions
- **Arduino IDE Compatible**: Easy installation and use within the Arduino ecosystem
- **Memory Efficient**: Designed for constrained environments (192KB SRAM + 64KB CCM)
- **Complete Examples**: Includes ready-to-use examples for common TinyML applications
- **Sensor Integration**: Built-in support for INMP441 microphone and MPU6050/LIS3DH accelerometer

## Supported Applications

| Application | Description | Sensors Used |
|-------------|-------------|--------------|
| Keyword Spotting | Detect spoken commands | INMP441 I2S Microphone |
| Gesture Recognition | Detect motion patterns | MPU6050/LIS3DH Accelerometer |
| Anomaly Detection | Detect unusual patterns | Any sensor data |
| Sine Wave Prediction | Basic ML demo | None (synthetic data) |

## Hardware 

- **AIduino Board** ( STM32F407VGT6-based board )
  - ARM Cortex-M4 @ 168MHz with FPU
  - 1MB Flash
  - 192KB SRAM + 64KB CCM
- **Sensors** 
  - INMP441 Digital MEMS Microphone (I2S)
  - MPU6050 or LIS3DH Accelerometer (I2C)
  - RGB LED (for visual feedback)

## Installation

### Method 1: Arduino Library Manager (Recommended)
1. Open Arduino IDE
2. Go to **Sketch > Include Library > Manage Libraries**
3. Search for "AIduino_TFLiteMicro"
4. Click Install

### Method 2: Manual Installation
1. Download the latest release from GitHub
2. Extract to your Arduino libraries folder:
   - Windows: `Documents\Arduino\libraries\`
   - macOS: `~/Documents/Arduino/libraries/`
   - Linux: `~/Arduino/libraries/`
3. Restart Arduino IDE

### Prerequisites
- **Arduino IDE** 1.8.x or 2.x
- **STM32duino** board support package:
  1. Open **File > Preferences**
  2. Add to Additional Board Manager URLs:
     ```
     https://github.com/stm32duino/BoardManagerFiles/raw/main/package_stmicroelectronics_index.json
     ```
  3. Go to **Tools > Board > Board Manager**
  4. Search for "STM32" and install "STM32 MCU based boards"

## Quick Start

### 1. Basic Example (Hello World)

```cpp
#include <AIduino_TFLiteMicro.h>
#include "your_model.h"  // Your converted TFLite model

// Create model instance with tensor arena size
AIduinoModel model(10 * 1024);  // 10KB arena

void setup() {
  Serial.begin(115200);
  
  // Initialize model
  AIduino_Error_t err = model.begin(your_model);
  if (err != AIDUINO_OK) {
    Serial.println(AIduinoModel::getErrorString(err));
    while (1);
  }
  
  model.printModelInfo();
}

void loop() {
  // Prepare input data
  float input_data[] = {/* your input values */};
  
  // Set input
  model.setInputFloat(input_data, sizeof(input_data)/sizeof(float));
  
  // Run inference
  model.predict();
  
  // Get results
  int predicted_class = model.getTopClass();
  float confidence = model.getTopConfidence();
  
  Serial.print("Class: ");
  Serial.print(predicted_class);
  Serial.print(" Confidence: ");
  Serial.println(confidence);
  
  delay(100);
}
```

### 2. Converting Your Model

#### Python Script to Train and Convert:

```python
import tensorflow as tf
import numpy as np

# 1. Create/train your model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(input_size,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)

# 2. Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]  # Full integer quantization
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Representative dataset for quantization
def representative_dataset():
    for i in range(100):
        yield [x_train[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

# 3. Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# 4. Convert to C array
# Run in terminal: xxd -i model.tflite > model.h
```

### 3. Memory Considerations

| Model Type | Typical Arena Size | Flash Usage |
|------------|-------------------|-------------|
| Simple (sine wave) | 8-10 KB | 10-20 KB |
| Keyword Spotting | 40-60 KB | 50-100 KB |
| Gesture Recognition | 20-40 KB | 30-50 KB |
| Image Classification | 60-100 KB | 100-200 KB |

**Tips for reducing memory:**
- Use int8 quantization instead of float32
- Reduce model complexity (fewer layers/neurons)
- Use smaller input dimensions
- Apply pruning during training

## API Reference

### AIduinoModel Class

#### Constructor
```cpp
AIduinoModel(size_t tensor_arena_size = 50*1024)
```

#### Methods

| Method | Description |
|--------|-------------|
| `begin(const uint8_t* model_data)` | Initialize with TFLite model |
| `predict()` | Run inference |
| `setInputFloat(const float* data, size_t n)` | Set float input data |
| `setInputInt8(const int8_t* data, size_t n)` | Set quantized input data |
| `getOutputFloat(float* data, size_t n)` | Get float output data |
| `getTopClass()` | Get index of highest output |
| `getTopConfidence()` | Get value of highest output |
| `getInferenceTimeUs()` | Get last inference time in microseconds |
| `printModelInfo()` | Print model details to Serial |

### Error Codes

| Code | Meaning |
|------|---------|
| `AIDUINO_OK` | Success |
| `AIDUINO_ERROR_MODEL_INVALID` | Invalid model format |
| `AIDUINO_ERROR_MEMORY_ALLOC` | Memory allocation failed |
| `AIDUINO_ERROR_TENSOR_ALLOC` | Tensor allocation failed |
| `AIDUINO_ERROR_INVOKE_FAILED` | Inference failed |

## Examples

### Included Examples

1. **hello_world**: Simple sine wave prediction demo
2. **keyword_spotting**: Voice command detection using INMP441 microphone
3. **gesture_recognition**: Motion gesture detection using accelerometer

### Running Examples

1. Open Arduino IDE
2. Go to **File > Examples > AIduino_TFLiteMicro**
3. Select an example
4. Select your board: **Tools > Board > STM32 boards > Generic STM32F4 series**
5. Select the part number: **Tools > Board part number > Generic F407VGTx**
6. Upload and open Serial Monitor

## Pin Configuration (AIduino Board)

| Function | Pin | Description |
|----------|-----|-------------|
| I2S WS | PB12 | INMP441 Word Select |
| I2S SCK | PB13 | INMP441 Serial Clock |
| I2S SD | PB15 | INMP441 Serial Data |
| I2C SDA | PB9 | MPU6050/LIS3DH Data |
| I2C SCL | PB8 | MPU6050/LIS3DH Clock |
| LED Red | PD12 | RGB LED Red |
| LED Green | PD13 | RGB LED Green |
| LED Blue | PD14 | RGB LED Blue |

### Performance Tips

1. **Use quantized models** - 4x smaller and faster
2. **Pre-allocate buffers** - Avoid dynamic allocation in loop()
3. **Batch predictions** - Process multiple samples if possible
4. **Enable FPU** - Already enabled by default for Cortex-M4

## Building from Source

If you need to modify the library:

```bash
git clone https://github.com/aiduino/AIduino_TFLiteMicro.git
cd AIduino_TFLiteMicro

# The library can be used directly - no compilation needed
# Just copy to your Arduino libraries folder
```

## References

- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [TinyML Book](https://tinymlbook.com/)
- [STM32F407 Reference Manual](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)
- [CMSIS-NN Documentation](https://arm-software.github.io/CMSIS_5/NN/html/index.html)
