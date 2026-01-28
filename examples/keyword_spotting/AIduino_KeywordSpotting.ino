/*
 * AIduino_KeywordSpotting.ino
 * 
 * Keyword Spotting example for AIduino
 * Detects spoken commands like "Hello AIduino" using the INMP441 microphone
 * 
 * Hardware: AIduino Board (STM32F407VGT6)
 * - INMP441 I2S Microphone (WS=PB12, SCK=PB13, SD=PB15)
 * - RGB LED for visual feedback
 * 
 * Workflow:
 * 1. Capture audio from I2S microphone
 * 2. Convert to spectrogram/MFCC features
 * 3. Run TFLite model for classification
 * 4. Display result on RGB LED
 */

#include <AIduino_TFLiteMicro.h>
// #include "keyword_model.h"  // Include your trained model

// I2S Configuration for INMP441
#define I2S_WS_PIN   PB12
#define I2S_SCK_PIN  PB13
#define I2S_SD_PIN   PB15

// LED Pins
#define LED_RED_PIN    PD12
#define LED_GREEN_PIN  PD13
#define LED_BLUE_PIN   PD14

// Audio Configuration
#define SAMPLE_RATE     16000   // 16 kHz
#define AUDIO_LENGTH_MS 1000    // 1 second of audio
#define AUDIO_SAMPLES   (SAMPLE_RATE * AUDIO_LENGTH_MS / 1000)

// Feature Configuration (for MFCC)
#define NUM_MFCC        13
#define WINDOW_SIZE_MS  30
#define WINDOW_STRIDE_MS 20
#define NUM_FRAMES      49      // Based on audio length and stride

// Model Configuration
#define TENSOR_ARENA_SIZE (50 * 1024)  // 50KB

// Keyword labels
const char* KEYWORDS[] = {
  "silence",
  "unknown",
  "yes",
  "no",
  "hello",
  "stop"
};
const int NUM_KEYWORDS = sizeof(KEYWORDS) / sizeof(KEYWORDS[0]);

// Global objects
AIduinoModel model(TENSOR_ARENA_SIZE);
AIduinoAudioFeatures audioFeatures;

// Buffers
int16_t audioBuffer[AUDIO_SAMPLES];
float featureBuffer[NUM_FRAMES * NUM_MFCC];

// State
bool modelLoaded = false;
int detectionCount = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  
  Serial.println("=====================================");
  Serial.println("AIduino Keyword Spotting Demo");
  Serial.println("=====================================");
  
  // Initialize LEDs
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_BLUE_PIN, OUTPUT);
  
  // Initial LED state (blue = initializing)
  setLED(0, 0, 255);
  
  // Initialize I2S for microphone
  if (!initI2S()) {
    Serial.println("ERROR: Failed to initialize I2S");
    setLED(255, 0, 0);  // Red = error
    while (1);
  }
  Serial.println("I2S microphone initialized");
  
  // Initialize audio feature extractor
  if (!audioFeatures.begin(SAMPLE_RATE, WINDOW_SIZE_MS, WINDOW_STRIDE_MS, NUM_MFCC)) {
    Serial.println("ERROR: Failed to initialize audio features");
    setLED(255, 0, 0);
    while (1);
  }
  Serial.println("Audio feature extractor initialized");
  
  // Load the model
  Serial.println("Loading keyword model...");
  
  // Uncomment when you have your model:
  // AIduino_Error_t err = model.begin(keyword_model);
  // if (err != AIDUINO_OK) {
  //   Serial.print("ERROR: ");
  //   Serial.println(AIduinoModel::getErrorString(err));
  //   setLED(255, 0, 0);
  //   while (1);
  // }
  // modelLoaded = true;
  
  Serial.println("NOTE: No model loaded - running in demo mode");
  Serial.println("Include your trained keyword_model.h to enable detection");
  
  // Ready state (green = ready)
  setLED(0, 255, 0);
  Serial.println("\nReady! Listening for keywords...");
  Serial.println("Say: yes, no, hello, stop");
}

void loop() {
  // Capture audio
  setLED(0, 0, 255);  // Blue = listening
  
  if (!captureAudio()) {
    Serial.println("Audio capture failed");
    delay(100);
    return;
  }
  
  // Extract features
  int numFrames = audioFeatures.extractFeatures(audioBuffer, AUDIO_SAMPLES, 
                                                 featureBuffer, NUM_FRAMES);
  
  if (numFrames < NUM_FRAMES) {
    Serial.println("Feature extraction incomplete");
    setLED(0, 255, 0);  // Back to ready
    return;
  }
  
  // Run inference (if model is loaded)
  if (modelLoaded) {
    AIduino_Error_t err = model.setInputFloat(featureBuffer, NUM_FRAMES * NUM_MFCC);
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
    
    // Get results
    int detectedClass = model.getTopClass();
    float confidence = model.getTopConfidence();
    
    // Print result
    Serial.print("Detected: ");
    if (detectedClass >= 0 && detectedClass < NUM_KEYWORDS) {
      Serial.print(KEYWORDS[detectedClass]);
    } else {
      Serial.print("unknown");
    }
    Serial.print(" (confidence: ");
    Serial.print(confidence * 100, 1);
    Serial.print("%) - ");
    Serial.print(model.getInferenceTimeUs());
    Serial.println(" us");
    
    // Visual feedback based on detection
    handleDetection(detectedClass, confidence);
    
  } else {
    // Demo mode - show audio level
    int32_t audioLevel = 0;
    for (int i = 0; i < AUDIO_SAMPLES; i++) {
      audioLevel += abs(audioBuffer[i]);
    }
    audioLevel /= AUDIO_SAMPLES;
    
    Serial.print("Audio level: ");
    Serial.println(audioLevel);
    
    // Pulse LED based on audio level
    int brightness = map(audioLevel, 0, 5000, 0, 255);
    brightness = constrain(brightness, 0, 255);
    setLED(0, brightness, 0);
  }
  
  delay(100);  // Brief pause between detections
}

// I2S initialization for INMP441 microphone
bool initI2S() {
  // Note: This is STM32-specific initialization
  // Using STM32duino I2S library or HAL
  
  // Configure GPIO pins for I2S
  // WS (Word Select/LRCLK) - PB12
  // SCK (Bit Clock) - PB13  
  // SD (Serial Data) - PB15
  
  // For full implementation, use:
  // - STM32 I2S HAL functions
  // - Configure DMA for efficient audio capture
  // - Set up circular buffer
  
  // Placeholder - return true for demo
  return true;
}

// Capture audio samples
bool captureAudio() {
  // Placeholder implementation
  // In actual use, read from I2S DMA buffer
  
  // For demo: generate synthetic audio (sine wave with noise)
  static float phase = 0;
  for (int i = 0; i < AUDIO_SAMPLES; i++) {
    float signal = sin(phase) * 10000;
    float noise = (random(-1000, 1000));
    audioBuffer[i] = (int16_t)(signal + noise);
    phase += 2 * PI * 440 / SAMPLE_RATE;  // 440 Hz tone
  }
  
  return true;
}

// Handle keyword detection
void handleDetection(int classIdx, float confidence) {
  // Only act on confident detections
  if (confidence < 0.8) {
    setLED(0, 255, 0);  // Green = ready, no action
    return;
  }
  
  detectionCount++;
  
  switch (classIdx) {
    case 0:  // silence
      setLED(0, 64, 0);  // Dim green
      break;
      
    case 1:  // unknown
      setLED(255, 255, 0);  // Yellow
      break;
      
    case 2:  // yes
      setLED(0, 255, 0);  // Bright green
      Serial.println(">>> Command: YES detected!");
      break;
      
    case 3:  // no
      setLED(255, 0, 0);  // Red
      Serial.println(">>> Command: NO detected!");
      break;
      
    case 4:  // hello
      setLED(0, 255, 255);  // Cyan
      Serial.println(">>> Command: HELLO detected!");
      break;
      
    case 5:  // stop
      setLED(255, 0, 255);  // Magenta
      Serial.println(">>> Command: STOP detected!");
      break;
      
    default:
      setLED(255, 255, 255);  // White
      break;
  }
  
  delay(500);  // Hold LED state briefly
}

// Set RGB LED color
void setLED(int r, int g, int b) {
  analogWrite(LED_RED_PIN, r);
  analogWrite(LED_GREEN_PIN, g);
  analogWrite(LED_BLUE_PIN, b);
}
