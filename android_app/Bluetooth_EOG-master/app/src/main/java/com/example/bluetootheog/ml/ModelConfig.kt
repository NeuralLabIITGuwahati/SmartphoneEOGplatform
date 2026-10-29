package com.example.bluetootheog.ml

object ModelConfig {
    const val SAMPLE_RATE = 128
    const val WINDOW_SAMPLES = 256       // 2 seconds × 128 Hz
    const val N_CHANNELS = 2         // H and V
    const val NUM_CLASSES = 6
    const val BUFFER_WINDOWS = 3         // rolling buffer = 3 × 256 = 768 samples
    const val BUFFER_SAMPLES = WINDOW_SAMPLES * BUFFER_WINDOWS  // 768
    const val EPSILON = 1e-8f     // avoid division by zero in z-score
    const val MODEL_FILE = "eegnet_final.tflite"

    val CLASS_NAMES = arrayOf("Left", "Right", "Up", "Down", "Blink", "No Movement")
}