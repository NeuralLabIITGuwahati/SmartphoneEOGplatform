package com.example.bluetootheog.model

class CircularBuffer(private val size: Int) {

    private val buffer = FloatArray(size)
    private var index = 0
    private var filled = false

    fun add(value: Float) {
        buffer[index] = value
        index = (index + 1) % size
        if (index == 0) filled = true
    }

    fun toList(): List<Float> {
        return if (!filled) {
            buffer.take(index)
        } else {
            buffer.slice(index until size) + buffer.slice(0 until index)
        }
    }
}