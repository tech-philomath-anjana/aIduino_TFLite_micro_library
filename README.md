# AIduino TFLM Standalone

## SELF-CONTAINED TensorFlow Lite Micro Runtime

**No external downloads needed** 
Everything is included

## Installation

1. Extract zip to `Documents/Arduino/libraries/`
2. Restart Arduino IDE
3. Done

## Usage

### Step 1: Convert your model

```bash
xxd -i your_model.tflite > model.h
```

### Step 2: Use in Arduino sketch

```cpp
#include <AIduino_TFLM.h>
#include "model.h"

TFLiteRuntime runtime;

void setup() {
  Serial.begin(115200);
  
  // Load model (model data, arena size in bytes)
  if (!runtime.begin(model_tflite, 30*1024)) {
    Serial.println("Failed!");
    while(1);
  }
}

void loop() {
  // Your input data
  float input[] = {1.0, 2.0, 3.0};
  
  // Run inference
  runtime.setInput(input, 3);
  runtime.invoke();
  
  // Get result
  int classIndex = runtime.getOutputMaxIndex();
  float confidence = runtime.getOutputMaxValue();
  
  Serial.print("Class: ");
  Serial.print(classIndex);
  Serial.print(" Conf: ");
  Serial.print(confidence);
  Serial.print(" Time: ");
  Serial.print(runtime.getInferenceTimeUs());
  Serial.println(" us");
  
  delay(1000);
}
```

## API

| Function | Description |
|----------|-------------|
| `begin(model, arena_size)` | Load model. Start with 30*1024 bytes. |
| `setInput(float*, count)` | Set float input data |
| `setInputInt8(int8_t*, count)` | Set quantized input |
| `invoke()` | Run inference |
| `getOutput()` | Get output array |
| `getOutputMaxIndex()` | Get predicted class |
| `getOutputMaxValue()` | Get confidence |
| `getInferenceTimeUs()` | Get timing |

## Supported Operations

- FullyConnected (Dense)
- Conv2D
- DepthwiseConv2D
- MaxPool2D
- AveragePool2D
- ReLU, ReLU6
- Softmax
- Logistic (Sigmoid)
- Tanh
- Reshape
- Add, Mul, Sub
- Quantize, Dequantize
- Mean

## Files

```
AIduino_TFLM_Standalone/
├── library.properties      # Arduino metadata
├── README.md              # This file
└── src/
    ├── AIduino_TFLM.h     # Main library (include this)
    ├── tflite_types.h     # Type definitions
    ├── tflite_model_parser.h  # FlatBuffer parser
    └── tflite_kernels.h   # Operation implementations
```



# AIduino TFLM SD Card

## TensorFlow Lite Micro with SD Card Model Loading

Load `.tflite` models directly from SD card instead of flash memory.

## Installation

1. Extract `AIduino_TFLM_SDCard.zip` to `Documents/Arduino/libraries/`
2. Restart Arduino IDE
3. Done!

## SD Card Setup

Copy your `.tflite` model files to SD card root:

```
SD Card/
├── gesture.tflite
├── temp.tflite
└── (other models...)
```

**Note:** Use short filenames (8.3 format recommended)

## Usage

```cpp
#include <AIduino_TFLM_SDCard.h>
#include <STM32SD.h>

TFLiteRuntime_SD runtime;

void setup() {
  Serial.begin(115200);
  
  // Initialize SD card
  if (!SD.begin()) {
    Serial.println("SD Card failed!");
    while(1);
  }
  
  // Load model from SD card
  if (!runtime.beginFromSD("gesture.tflite", 15*1024)) {
    Serial.println("Model load failed!");
    while(1);
  }
  
  Serial.println("Ready!");
}

void loop() {
  // Your input data (e.g., accelerometer X, Y, Z)
  float input[] = {0.5, -0.2, 9.8};
  
  // Run inference
  runtime.setInput(input, 3);
  runtime.invoke();
  
  // Get result
  int gesture = runtime.getOutputMaxIndex();
  float confidence = runtime.getOutputMaxValue();
  
  Serial.print("Gesture: ");
  Serial.print(gesture);
  Serial.print(" Confidence: ");
  Serial.println(confidence);
  
  delay(100);
}
```

## API Reference

| Function | Description |
|----------|-------------|
| `beginFromSD(filename, arena_size)` | Load model from SD card |
| `begin(data, arena_size)` | Load model from memory (flash) |
| `setInput(float*, count)` | Set input tensor data |
| `invoke()` | Run inference |
| `getOutput()` | Get output tensor as float array |
| `getOutputMaxIndex()` | Get index of highest value (classification) |
| `getOutputMaxValue()` | Get highest output value (confidence) |
| `getInferenceTimeUs()` | Get last inference time in microseconds |
| `getModelSize()` | Get loaded model size in bytes |
| `getArenaUsed()` | Get arena memory used |
| `isReady()` | Check if model is loaded |

## How It Works

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   SD Card    │ ---> │     RAM      │ ---> │  Inference   │
│  .tflite     │      │   (model)    │      │   (output)   │
└──────────────┘      └──────────────┘      └──────────────┘
     File              Memory Copy           Run ML Model
```

1. **SD Card** stores the `.tflite` model file
2. **beginFromSD()** reads file into RAM
3. **invoke()** runs the neural network
4. **getOutput()** returns predictions

## Memory Usage

```
RAM (192 KB total):
┌─────────────────────────────────┐
│ Model buffer    (~4 KB)         │  ← .tflite loaded here
├─────────────────────────────────┤
│ Arena           (~15 KB)        │  ← Tensor computations
├─────────────────────────────────┤
│ Free            (~170 KB)       │  ← Available for your code
└─────────────────────────────────┘
```

## Supported Operations

- FullyConnected (Dense)
- Conv2D
- DepthwiseConv2D
- MaxPool2D, AveragePool2D
- ReLU, ReLU6
- Softmax
- Logistic (Sigmoid)
- Tanh
- Reshape
- Add, Mul, Sub
- Quantize, Dequantize

## File Structure

```
AIduino_TFLM_SDCard/
├── library.properties
├── README.md
└── src/
    ├── AIduino_TFLM_SDCard.h    ← Main library (include this)
    ├── tflite_types.h           ← Data type definitions
    ├── tflite_model_parser.h    ← FlatBuffer parser
    └── tflite_kernels.h         ← ML operations
```

## Converting Models

If you have a `.h` file and need `.tflite`:

**Python script:**
```python
import re

def h_to_tflite(h_file, tflite_file):
    with open(h_file, 'r') as f:
        content = f.read()
    hex_values = re.findall(r'0x([0-9a-fA-F]{2})', content)
    data = bytes([int(h, 16) for h in hex_values])
    with open(tflite_file, 'wb') as f:
        f.write(data)

h_to_tflite('gesture_model.h', 'gesture.tflite')
```

If you have a `.tflite` and need `.h` (for flash version):
```bash
xxd -i model.tflite > model.h
```

---

## Example: Multiple Models

```cpp
#include <AIduino_TFLM_SDCard.h>
#include <STM32SD.h>

TFLiteRuntime_SD gestureModel;
TFLiteRuntime_SD tempModel;

void setup() {
  Serial.begin(115200);
  SD.begin();
  
  gestureModel.beginFromSD("gesture.tflite", 15*1024);
  tempModel.beginFromSD("temp.tflite", 10*1024);
}

void loop() {
  // Gesture inference
  float accel[] = {ax, ay, az};
  gestureModel.setInput(accel, 3);
  gestureModel.invoke();
  int gesture = gestureModel.getOutputMaxIndex();
  
  // Temperature inference
  float temp[] = {temperature};
  tempModel.setInput(temp, 1);
  tempModel.invoke();
  float predicted = tempModel.getOutput()[0];
}
```
