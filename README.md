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

