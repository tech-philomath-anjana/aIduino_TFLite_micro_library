/*
 * AIduino_HelloWorld.ino
 * 
 * Hello World example for AIduino TensorFlow Lite Micro Library
 * This example demonstrates running a simple sine wave prediction model
 * 
 * Hardware: AIduino Board (STM32F407VGT6)
 * 
 * The model predicts sin(x) for input values x
 * LED brightness varies with the predicted sine wave
 */

#include <AIduino_TFLiteMicro.h>

// Include the model (this should be your trained model converted to a C array)
// Use xxd -i model.tflite > model.h to convert
// For this example, we'll include a placeholder
#include "sine_model.h"

// LED pin (adjust for your board)
#define LED_PIN PD12  // AIduino RGB LED Red channel

// Model configuration
#define TENSOR_ARENA_SIZE (8 * 1024)  // 8KB for simple sine model

// Create the model instance
AIduinoModel model(TENSOR_ARENA_SIZE);

// Inference variables
float x_val = 0.0f;
const float x_increment = 0.05f;

void setup() {
  // Initialize serial
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("=================================");
  Serial.println("AIduino TFLite Micro - Hello World");
  Serial.println("=================================");
  
  // Initialize LED
  pinMode(LED_PIN, OUTPUT);
  
  // Initialize the model
  Serial.println("Loading model...");
  
  AIduino_Error_t err = model.begin(sine_model);
  
  if (err != AIDUINO_OK) {
    Serial.print("Failed to initialize model: ");
    Serial.println(AIduinoModel::getErrorString(err));
    while (1) {
      digitalWrite(LED_PIN, HIGH);
      delay(100);
      digitalWrite(LED_PIN, LOW);
      delay(100);
    }
  }
  
  Serial.println("Model loaded successfully!");
  model.printModelInfo();
  
  // Print platform info
  AIduinoUtils::printMemoryInfo();
}

void loop() {
  // Update x value (wrap around 2*PI)
  x_val += x_increment;
  if (x_val > 2.0f * PI) {
    x_val = 0.0f;
  }
  
  // Set input
  AIduino_Error_t err = model.setInputFloat(&x_val, 1);
  if (err != AIDUINO_OK) {
    Serial.print("Input error: ");
    Serial.println(AIduinoModel::getErrorString(err));
    return;
  }
  
  // Run inference
  err = model.predict();
  if (err != AIDUINO_OK) {
    Serial.print("Inference error: ");
    Serial.println(AIduinoModel::getErrorString(err));
    return;
  }
  
  // Get output
  float y_predicted;
  err = model.getOutputFloat(&y_predicted, 1);
  if (err != AIDUINO_OK) {
    Serial.print("Output error: ");
    Serial.println(AIduinoModel::getErrorString(err));
    return;
  }
  
  // Calculate actual sine for comparison
  float y_actual = sin(x_val);
  
  // Print results
  Serial.print("x: ");
  Serial.print(x_val, 4);
  Serial.print(" | Predicted: ");
  Serial.print(y_predicted, 4);
  Serial.print(" | Actual: ");
  Serial.print(y_actual, 4);
  Serial.print(" | Error: ");
  Serial.print(abs(y_predicted - y_actual), 6);
  Serial.print(" | Time: ");
  Serial.print(model.getInferenceTimeUs());
  Serial.println(" us");
  
  // Update LED brightness based on prediction
  // Map -1 to 1 range to 0-255 for PWM
  int brightness = (int)((y_predicted + 1.0f) * 127.5f);
  brightness = constrain(brightness, 0, 255);
  analogWrite(LED_PIN, brightness);
  
  delay(50);  // ~20 Hz update rate
}
