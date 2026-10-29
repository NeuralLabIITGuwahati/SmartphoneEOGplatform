package com.example.bluetootheog.ml

class EOGPreprocessor {

    // ── Rolling buffer — holds last 768 samples per channel (3 windows per channel) ──
    private val hBuffer = ArrayDeque<Float>(ModelConfig.BUFFER_SAMPLES)
    private val vBuffer = ArrayDeque<Float>(ModelConfig.BUFFER_SAMPLES)

    // ── Normalization mode ────────────────────────────────────────────────`
    enum class NormalizationMode {
        PER_WINDOW,   // normalize using stats from the 256-sample window only
        PER_BUFFER    // normalize using stats from all 768 samples in buffer
    }

    // ─────────────────────────────────────────────────────────────────────
    // Add new sample to rolling buffer
    // Called from EOGRepository on every new BT sample
    // ─────────────────────────────────────────────────────────────────────
    fun addSample(h: Float, v: Float) {
        hBuffer.addLast(h)
        vBuffer.addLast(v)

        // Keep buffer size bounded at BUFFER_SAMPLES (768)
        if (hBuffer.size > ModelConfig.BUFFER_SAMPLES) {
            hBuffer.removeFirst()
            vBuffer.removeFirst()
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Check if we have enough samples to run inference
    // ─────────────────────────────────────────────────────────────────────
    fun isReady(): Boolean = hBuffer.size >= ModelConfig.WINDOW_SAMPLES

    // ─────────────────────────────────────────────────────────────────────
    // Get preprocessed window ready for TFLite input
    // Returns FloatArray of shape (256 × 2) = 512 floats
    // Interleaved: [h0, v0, h1, v1, ..., h255, v255]
    // ─────────────────────────────────────────────────────────────────────
    fun getProcessedWindow(mode: NormalizationMode): FloatArray {

        // Take last 256 samples from buffer
        val hWindow = hBuffer.takeLast(ModelConfig.WINDOW_SAMPLES).toFloatArray()
        val vWindow = vBuffer.takeLast(ModelConfig.WINDOW_SAMPLES).toFloatArray()

        // Normalize based on selected mode
        val (hNorm, vNorm) = when (mode) {
            NormalizationMode.PER_WINDOW -> normalizePerWindow(hWindow, vWindow)
            NormalizationMode.PER_BUFFER -> normalizePerBuffer(hWindow, vWindow)
        }

        // Interleave H and V into flat array: [h0,v0, h1,v1, ..., h255,v255]
        // This matches EEGNet input shape (256, 2)
        return FloatArray(ModelConfig.WINDOW_SAMPLES * ModelConfig.N_CHANNELS) { i ->
            if (i % 2 == 0) hNorm[i / 2] else vNorm[i / 2]
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Option A — Per-window normalization
    // Stats computed from the 256-sample window only
    // ─────────────────────────────────────────────────────────────────────
    private fun normalizePerWindow(
        hWindow: FloatArray,
        vWindow: FloatArray
    ): Pair<FloatArray, FloatArray> {
        return Pair(
            zScore(hWindow, hWindow.mean(), hWindow.std()),
            zScore(vWindow, vWindow.mean(), vWindow.std())
        )
    }

    // ─────────────────────────────────────────────────────────────────────
    // Option B — Per-buffer normalization
    // Stats computed from all 768 samples in rolling buffer
    // Closer to how training normalization worked (per trial)
    // ─────────────────────────────────────────────────────────────────────
    private fun normalizePerBuffer(
        hWindow: FloatArray,
        vWindow: FloatArray
    ): Pair<FloatArray, FloatArray> {
        val hAll = hBuffer.toFloatArray()
        val vAll = vBuffer.toFloatArray()

        // Compute stats from full buffer
        val hMean = hAll.mean();
        val hStd = hAll.std()
        val vMean = vAll.mean();
        val vStd = vAll.std()

        // Normalize only the window using buffer stats
        return Pair(
            zScore(hWindow, hMean, hStd),
            zScore(vWindow, vMean, vStd)
        )
    }

    // ─────────────────────────────────────────────────────────────────────
    // Z-score formula — matches Python preprocessing exactly:
    // normalized = (signal - mean) / (std + 1e-8)
    // ─────────────────────────────────────────────────────────────────────
    private fun zScore(data: FloatArray, mean: Float, std: Float): FloatArray {
        return FloatArray(data.size) { i ->
            (data[i] - mean) / (std + ModelConfig.EPSILON)
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Reset buffer — call when BT disconnects
    // ─────────────────────────────────────────────────────────────────────
    fun reset() {
        hBuffer.clear()
        vBuffer.clear()
    }

    // ─────────────────────────────────────────────────────────────────────
    // FloatArray extension helpers — mean and std
    // ─────────────────────────────────────────────────────────────────────
    private fun FloatArray.mean(): Float = sum() / size

    private fun FloatArray.std(): Float {
        val m = mean()
        return kotlin.math.sqrt(map { (it - m) * (it - m) }.sum() / size)
    }
}