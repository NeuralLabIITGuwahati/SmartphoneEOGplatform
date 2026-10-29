package com.example.bluetootheog.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.roundToInt

@Composable
fun LabeledGraph(
    values: List<Float>,
    tick: Int,
    modifier: Modifier = Modifier,
    yLabel: String = "EOG (a.u.)",
    isConnected: Boolean = false
) {
    Row(
        modifier = modifier
    ) {
        if (isConnected) {
            // Y-Axis Label
            Box(
                modifier = Modifier
                    .width(60.dp)
                    .fillMaxHeight(),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = yLabel,
                    modifier = Modifier.graphicsLayer(rotationZ = -90f),
                    fontSize = 12.sp,
                    maxLines = 1
                )
            }
        }

        // Graph
        EOGGraph(
            values = values, tick = tick, modifier = Modifier.weight(1f)
        )
    }
}


@Composable
fun EOGGraph(values: List<Float>, tick: Int, modifier: Modifier = Modifier) {

    val safeValues = values.toList()
    val path = remember { Path() }

    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(250.dp)
    ) {

        if (safeValues.size < 2) return@Canvas

        val w = size.width
        val h = size.height

        // Scaling
        val maxVal = safeValues.maxOrNull() ?: 1f
        val minVal = safeValues.minOrNull() ?: -1f
        val range = (maxVal - minVal).takeIf { it != 0f } ?: 1f

        val xStep = w / (safeValues.size - 1)

        // Grid lines
        val gridColor = Color(0xFFCCCCCC)

        val labelPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 32f
        }

        // Horizontal grid lines: 5 partitions
        val rows = 5
        for (i in 0..rows) {
            val y = h * i / rows

            drawLine(
                start = Offset(0f, y), end = Offset(w, y), color = gridColor, strokeWidth = 1f
            )

            val value = maxVal - (range / rows) * i
            drawContext.canvas.nativeCanvas.drawText(
                String.format("%.0f", (value / 10).roundToInt() * 10f), 10f, y - 5f, labelPaint
            )
        }

        // Vertical grid lines: 8 partitions
        val cols = 8
        for (i in 0..cols) {
            val x = w * i / cols
            drawLine(
                start = Offset(x, 0f), end = Offset(x, h), color = gridColor, strokeWidth = 1f
            )
        }

        // Build waveform path
        path.reset()

        val firstY = h - ((safeValues[0] - minVal) / range) * h
        path.moveTo(0f, firstY)

        for (i in 1 until safeValues.size) {
            val x = i * xStep
            val y = h - ((safeValues[i] - minVal) / range) * h
            path.lineTo(x, y)
        }

        drawPath(
            path = path, color = Color.Green, style = Stroke(
                width = 3f, cap = StrokeCap.Round
            )
        )
    }
}