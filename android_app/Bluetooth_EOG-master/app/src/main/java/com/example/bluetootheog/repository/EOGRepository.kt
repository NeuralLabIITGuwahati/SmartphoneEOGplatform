package com.example.bluetootheog.repository

import com.example.bluetootheog.bluetooth.BluetoothManager
import com.example.bluetootheog.ml.EOGPreprocessor
import com.example.bluetootheog.ml.EyeMovementClassifier
import com.example.bluetootheog.ml.ModelConfig
import com.example.bluetootheog.model.EOGReading

class EOGRepository(
    private val bluetoothManager: BluetoothManager, private val classifier: EyeMovementClassifier
) {

    // Recording state
    var isRecording = false
        private set

    // Inference state
    var isInferring = false
        private set

    private val recordingBuffer = mutableListOf<EOGReading>()
    private var firstSampleIndex: Long = -1
    val preprocessor = EOGPreprocessor()
    private var samplesSinceLastInference = 0
    var onPredictionReady: ((String) -> Unit)? = null

    // Recording start/stop handling
    fun startRecording() {
        recordingBuffer.clear()
        firstSampleIndex = -1
        isRecording = true
    }
    fun stopRecording(): List<EOGReading> {
        isRecording = false
        return recordingBuffer.toList()  // Return a copy
    }

    // Inference start/stop handling
    fun startInferring() {
        samplesSinceLastInference = 0
        preprocessor.reset()
        isInferring = true
    }
    fun stopInferring() {
        isInferring = false
        preprocessor.reset()
    }

    companion object {
        const val SAMPLE_RATE = 128
    }

    fun onNewData(sampleIndex: Long, h: Float, v: Float) {

        // ── Inference mode ─────────────────────────────────────────────────
        if (isInferring) {
            preprocessor.addSample(h, v)
            samplesSinceLastInference++

            if (preprocessor.isReady() && samplesSinceLastInference >= 128) {
                samplesSinceLastInference = 0

                // Get preprocessed window
                val windowData = preprocessor.getProcessedWindow(
                    EOGPreprocessor.NormalizationMode.PER_BUFFER
                )

                // Run inference
                val (predictedClass, confidence) = classifier.predict(windowData)

                // Notify UI only if confidence is above threshold
                if (confidence > 0.7f && predictedClass >= 0) {
                    onPredictionReady?.invoke(ModelConfig.CLASS_NAMES[predictedClass])
                }
            }
        }

        // ── Recording mode ─────────────────────────────────────────────────
        if (isRecording) {
            // Capturing first sample index when recording starts
            if (firstSampleIndex == -1L) {
                firstSampleIndex = sampleIndex
            }

            // Relative indexing from recording start
            val relativeSampleIndex = sampleIndex - firstSampleIndex
            val timeMs = relativeSampleIndex * (1000.0 / SAMPLE_RATE)

            recordingBuffer.add(
                EOGReading(
                    sampleIndex = relativeSampleIndex, timeMs = timeMs, h = h, v = v
                )
            )
        }
    }
}