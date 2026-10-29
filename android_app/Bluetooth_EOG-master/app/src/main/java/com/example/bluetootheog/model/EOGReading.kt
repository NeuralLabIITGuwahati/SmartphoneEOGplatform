package com.example.bluetootheog.model

data class EOGReading(
    val sampleIndex: Long,
    val timeMs: Double,    // Time in milliseconds
    val h: Float,
    val v: Float
)
