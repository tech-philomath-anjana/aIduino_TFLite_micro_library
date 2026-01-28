/*
 * AIduino_GestureRecognition.ino
 * 
 * Gesture Recognition example for AIduino
 * Detects motion gestures using MPU6050/LIS3DH accelerometer
 * 
 * Hardware: AIduino Board (STM32F407VGT6)
 * - MPU6050 or LIS3DH (SDA=PB9, SCL=PB8)
 * - RGB LED for visual feedback
 * 
 * Supported Gestures:
 * - Shake (rapid back-and-forth motion)
 * - Swipe Left/Right
 * - Swipe Up/Down
 * - Circle (circular motion)
 * - Tap (quick impact)
 */

#include <AIduino_TFLiteMicro.h>
#include <Wire.h>
// #include "gesture_model.h"  // Include your trained model

// I2C Configuration
#define I2C_SDA_PIN  PB9
#define I2C_SCL_PIN  PB8

// MPU6050 Configuration
#define MPU6050_ADDR 0x68

// LED Pins
#define LED_RED_PIN    PD12
#define LED_GREEN_PIN  PD13
#define LED_BLUE_PIN   PD14

// Accelerometer Configuration
#define SAMPLE_RATE_HZ   100   // 100 Hz sampling
#define WINDOW_SIZE      128   // 128 samples (~1.28 seconds)
#define NUM_AXES         3     // X, Y, Z
#define INPUT_SIZE       (WINDOW_SIZE * NUM_AXES)

// Model Configuration
#define TENSOR_ARENA_SIZE (30 * 1024)  // 30KB

// Gesture labels
const char* GESTURES[] = {
  "idle",
  "shake",
  "swipe_left",
  "swipe_right",
  "swipe_up",
  "swipe_down",
  "circle",
  "tap"
};
const int NUM_GESTURES = sizeof(GESTURES) / sizeof(GESTURES[0]);

// Global objects
AIduinoModel model(TENSOR_ARENA_SIZE);
AIduinoAccelFeatures accelFeatures;

// Sample buffer
float accelBuffer[INPUT_SIZE];

// State
bool modelLoaded = false;
unsigned long lastSampleTime = 0;
const unsigned long sampleInterval = 1000 / SAMPLE_RATE_HZ;  // ms per sample

// Statistics
int gestureHistory[NUM_GESTURES] = {0};
int totalDetections = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  
  Serial.println("=====================================");
  Serial.println("AIduino Gesture Recognition Demo");
  Serial.println("=====================================");
  
  // Initialize LEDs
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_BLUE_PIN, OUTPUT);
  
  // Initial LED state (blue = initializing)
  setLED(0, 0, 255);
  
  // Initialize I2C
  Wire.setSDA(I2C_SDA_PIN);
  Wire.setSCL(I2C_SCL_PIN);
  Wire.begin();
  
  // Initialize accelerometer
  if (!initMPU6050()) {
    Serial.println("ERROR: Failed to initialize MPU6050");
    Serial.println("Check wiring: SDA=PB9, SCL=PB8");
    setLED(255, 0, 0);  // Red = error
    while (1);
  }
  Serial.println("MPU6050 accelerometer initialized");
  
  // Initialize feature collector
  if (!accelFeatures.begin(SAMPLE_RATE_HZ, WINDOW_SIZE)) {
    Serial.println("ERROR: Failed to initialize feature collector");
    setLED(255, 0, 0);
    while (1);
  }
  Serial.println("Feature collector initialized");
  
  // Load the model
  Serial.println("Loading gesture model...");
  
  // Uncomment when you have your model:
  // AIduino_Error_t err = model.begin(gesture_model);
  // if (err != AIDUINO_OK) {
  //   Serial.print("ERROR: ");
  //   Serial.println(AIduinoModel::getErrorString(err));
  //   setLED(255, 0, 0);
  //   while (1);
  // }
  // modelLoaded = true;
  // model.printModelInfo();
  
  Serial.println("NOTE: No model loaded - running in demo mode");
  Serial.println("Include your trained gesture_model.h to enable detection");
  
  // Ready state (green = ready)
  setLED(0, 255, 0);
  Serial.println("\nReady! Perform gestures:");
  Serial.println("- Shake the board");
  Serial.println("- Swipe left/right/up/down");
  Serial.println("- Draw a circle");
  Serial.println("- Tap the board");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Sample at fixed rate
  if (currentTime - lastSampleTime >= sampleInterval) {
    lastSampleTime = currentTime;
    
    // Read accelerometer
    float ax, ay, az;
    if (!readMPU6050(&ax, &ay, &az)) {
      return;
    }
    
    // Add sample to feature buffer
    bool windowReady = accelFeatures.addSample(ax, ay, az);
    
    // Visual feedback during sampling (pulse blue)
    int brightness = (millis() / 50) % 64 + 32;
    setLED(0, 0, brightness);
    
    // When window is full, run inference
    if (windowReady) {
      // Get features
      accelFeatures.getFeatures(accelBuffer);
      
      // Run inference
      if (modelLoaded) {
        performInference();
      } else {
        // Demo mode - detect simple patterns
        detectSimplePatterns(accelBuffer);
      }
      
      // Reset for next window
      accelFeatures.reset();
    }
  }
}

// Initialize MPU6050
bool initMPU6050() {
  // Wake up MPU6050
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0x00);  // Wake up
  if (Wire.endTransmission() != 0) {
    return false;
  }
  
  // Configure accelerometer range (+/- 2g)
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x1C);  // ACCEL_CONFIG register
  Wire.write(0x00);  // +/- 2g range
  Wire.endTransmission();
  
  // Verify connection
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x75);  // WHO_AM_I register
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 1);
  
  if (Wire.available()) {
    uint8_t whoami = Wire.read();
    return (whoami == 0x68 || whoami == 0x98);  // MPU6050 or MPU6500
  }
  
  return false;
}

// Read accelerometer data
bool readMPU6050(float* ax, float* ay, float* az) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B);  // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) {
    return false;
  }
  
  Wire.requestFrom(MPU6050_ADDR, 6);
  if (Wire.available() < 6) {
    return false;
  }
  
  int16_t rawX = (Wire.read() << 8) | Wire.read();
  int16_t rawY = (Wire.read() << 8) | Wire.read();
  int16_t rawZ = (Wire.read() << 8) | Wire.read();
  
  // Convert to g (for +/- 2g range, sensitivity is 16384 LSB/g)
  *ax = rawX / 16384.0f;
  *ay = rawY / 16384.0f;
  *az = rawZ / 16384.0f;
  
  return true;
}

// Run inference with TFLite model
void performInference() {
  AIduino_Error_t err = model.setInputFloat(accelBuffer, INPUT_SIZE);
  if (err != AIDUINO_OK) {
    Serial.print("Input error: ");
    Serial.println(AIduinoModel::getErrorString(err));
    return;
  }
  
  err = model.predict();
  if (err != AIDUINO_OK) {
    Serial.print("Inference error: ");
    Serial.println(AIduinoModel::getErrorString(err));
    return;
  }
  
  int gestureIdx = model.getTopClass();
  float confidence = model.getTopConfidence();
  
  // Print results
  Serial.print("Gesture: ");
  if (gestureIdx >= 0 && gestureIdx < NUM_GESTURES) {
    Serial.print(GESTURES[gestureIdx]);
  } else {
    Serial.print("unknown");
  }
  Serial.print(" (");
  Serial.print(confidence * 100, 1);
  Serial.print("%) - ");
  Serial.print(model.getInferenceTimeUs());
  Serial.println(" us");
  
  // Visual feedback
  if (confidence > 0.7) {
    handleGesture(gestureIdx);
  } else {
    setLED(0, 255, 0);  // Green = ready
  }
}

// Simple pattern detection for demo mode
void detectSimplePatterns(float* buffer) {
  // Calculate statistics
  float maxMag = 0;
  float totalMag = 0;
  float variance = 0;
  
  for (int i = 0; i < WINDOW_SIZE; i++) {
    float x = buffer[i * 3];
    float y = buffer[i * 3 + 1];
    float z = buffer[i * 3 + 2];
    
    float mag = sqrt(x*x + y*y + z*z);
    totalMag += mag;
    if (mag > maxMag) maxMag = mag;
  }
  
  float avgMag = totalMag / WINDOW_SIZE;
  
  // Calculate variance
  for (int i = 0; i < WINDOW_SIZE; i++) {
    float x = buffer[i * 3];
    float y = buffer[i * 3 + 1];
    float z = buffer[i * 3 + 2];
    
    float mag = sqrt(x*x + y*y + z*z);
    variance += (mag - avgMag) * (mag - avgMag);
  }
  variance /= WINDOW_SIZE;
  
  // Simple classification based on statistics
  int detectedGesture = 0;  // idle
  
  if (variance > 0.5) {
    detectedGesture = 1;  // shake
    Serial.println(">>> SHAKE detected!");
    setLED(255, 165, 0);  // Orange
  } else if (maxMag > 2.5) {
    detectedGesture = 7;  // tap
    Serial.println(">>> TAP detected!");
    setLED(255, 0, 255);  // Magenta
  } else if (avgMag < 0.95) {
    // Check for swipe direction
    float firstX = buffer[0];
    float lastX = buffer[(WINDOW_SIZE-1) * 3];
    float firstY = buffer[1];
    float lastY = buffer[(WINDOW_SIZE-1) * 3 + 1];
    
    float deltaX = lastX - firstX;
    float deltaY = lastY - firstY;
    
    if (abs(deltaX) > 0.5) {
      if (deltaX > 0) {
        detectedGesture = 3;  // swipe_right
        Serial.println(">>> SWIPE RIGHT detected!");
        setLED(0, 255, 0);  // Green
      } else {
        detectedGesture = 2;  // swipe_left
        Serial.println(">>> SWIPE LEFT detected!");
        setLED(255, 0, 0);  // Red
      }
    } else if (abs(deltaY) > 0.5) {
      if (deltaY > 0) {
        detectedGesture = 4;  // swipe_up
        Serial.println(">>> SWIPE UP detected!");
        setLED(0, 0, 255);  // Blue
      } else {
        detectedGesture = 5;  // swipe_down
        Serial.println(">>> SWIPE DOWN detected!");
        setLED(255, 255, 0);  // Yellow
      }
    }
  }
  
  // Update statistics
  if (detectedGesture > 0) {
    gestureHistory[detectedGesture]++;
    totalDetections++;
    
    // Print debug info
    Serial.print("Stats - Max: ");
    Serial.print(maxMag, 2);
    Serial.print(" Avg: ");
    Serial.print(avgMag, 2);
    Serial.print(" Var: ");
    Serial.println(variance, 4);
    
    delay(500);  // Debounce
  }
  
  // Return to ready state
  setLED(0, 128, 0);  // Dim green
}

// Handle gesture with LED feedback
void handleGesture(int gestureIdx) {
  gestureHistory[gestureIdx]++;
  totalDetections++;
  
  switch (gestureIdx) {
    case 0:  // idle
      setLED(0, 64, 0);  // Dim green
      break;
      
    case 1:  // shake
      setLED(255, 165, 0);  // Orange
      break;
      
    case 2:  // swipe_left
      setLED(255, 0, 0);  // Red
      break;
      
    case 3:  // swipe_right
      setLED(0, 255, 0);  // Green
      break;
      
    case 4:  // swipe_up
      setLED(0, 0, 255);  // Blue
      break;
      
    case 5:  // swipe_down
      setLED(255, 255, 0);  // Yellow
      break;
      
    case 6:  // circle
      setLED(0, 255, 255);  // Cyan
      break;
      
    case 7:  // tap
      setLED(255, 0, 255);  // Magenta
      break;
      
    default:
      setLED(255, 255, 255);  // White
      break;
  }
  
  delay(300);  // Hold LED state
  setLED(0, 128, 0);  // Return to ready
}

// Set RGB LED color
void setLED(int r, int g, int b) {
  analogWrite(LED_RED_PIN, r);
  analogWrite(LED_GREEN_PIN, g);
  analogWrite(LED_BLUE_PIN, b);
}

// Print statistics
void printStats() {
  Serial.println("\n=== Gesture Statistics ===");
  for (int i = 0; i < NUM_GESTURES; i++) {
    Serial.print(GESTURES[i]);
    Serial.print(": ");
    Serial.println(gestureHistory[i]);
  }
  Serial.print("Total: ");
  Serial.println(totalDetections);
  Serial.println("========================\n");
}
