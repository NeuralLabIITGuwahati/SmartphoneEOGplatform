// ml/EyeMovementClassifier.kt
package com.example.bluetootheog.ml

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class EyeMovementClassifier(context: Context) {

    private val interpreter: Interpreter
    private val inputBuffer: ByteBuffer

    init {
        // Load model from assets
        val modelBuffer = loadModelFile(context.assets, ModelConfig.MODEL_FILE)
        val options = Interpreter.Options().apply {
            setNumThreads(2)
        }
        interpreter = Interpreter(modelBuffer, options)

        // Prepare input buffer
        inputBuffer = ByteBuffer.allocateDirect(
            1 * ModelConfig.WINDOW_SAMPLES * ModelConfig.N_CHANNELS * 4  // 4 bytes per float
        ).apply {
            order(ByteOrder.nativeOrder())
        }

        Log.d("EyeMovementClassifier", "Model loaded successfully")
    }

    // ─────────────────────────────────────────────────────────────────────
    // Run inference on preprocessed window
    // Input: FloatArray of size 512 (interleaved H,V: [h0,v0,h1,v1,...,h255,v255])
    // Output: predicted class index (0-5) and confidence
    // ─────────────────────────────────────────────────────────────────────
    fun predict(windowData: FloatArray): Pair<Int, Float> {

        // Validate input size
        if (windowData.size != ModelConfig.WINDOW_SAMPLES * ModelConfig.N_CHANNELS) {
            Log.e(
                "EyeMovementClassifier",
                "Invalid input size: ${windowData.size}, expected ${ModelConfig.WINDOW_SAMPLES * ModelConfig.N_CHANNELS}"
            )
            return Pair(-1, 0f)
        }

        // Fill input buffer
        inputBuffer.rewind()
        inputBuffer.asFloatBuffer().put(windowData)

        // Prepare output buffer — shape [1, 6]
        val outputBuffer = ByteBuffer.allocateDirect(1 * ModelConfig.NUM_CLASSES * 4)
            .apply { order(ByteOrder.nativeOrder()) }

        // Run inference
        interpreter.run(inputBuffer, outputBuffer)

        // Extract probabilities
        outputBuffer.rewind()
        val probabilities = FloatArray(ModelConfig.NUM_CLASSES)
        outputBuffer.asFloatBuffer().get(probabilities)

        // Get argmax and confidence
        val predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1
        val confidence = if (predictedClass >= 0) probabilities[predictedClass] else 0f

        Log.d(
            "EyeMovementClassifier",
            "Prediction: ${ModelConfig.CLASS_NAMES[predictedClass]} " + "(confidence: ${
                "%.1f".format(confidence * 100)
            }%)"
        )

        return Pair(predictedClass, confidence)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Load .tflite file from assets into ByteBuffer
    // ─────────────────────────────────────────────────────────────────────
    private fun loadModelFile(assetManager: AssetManager, modelPath: String): ByteBuffer {
        val assetFileDescriptor = assetManager.openFd(modelPath)
        val inputStream = assetManager.open(modelPath)
        val modelBytes = inputStream.readBytes()
        inputStream.close()

        return ByteBuffer.allocateDirect(modelBytes.size).apply {
            order(ByteOrder.nativeOrder())
            put(modelBytes)
            rewind()
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────────────
    fun close() {
        interpreter.close()
    }
}