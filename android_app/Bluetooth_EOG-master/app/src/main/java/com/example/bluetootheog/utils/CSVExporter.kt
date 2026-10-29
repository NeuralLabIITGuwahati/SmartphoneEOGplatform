package com.example.bluetootheog.utils

import android.content.Context
import android.os.Environment
import com.example.bluetootheog.model.EOGReading
import com.example.bluetootheog.model.EyeMovements
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object CSVExporter {
    fun saveToCSV(context: Context, readings: List<EOGReading>, label: String) {
        // Creates a file like: EOG_2024_01_15_10_30_00_label.csv
        val fileName = "EOG_${
            SimpleDateFormat(
                "yyyy_MM_dd_HH_mm_ss", Locale.getDefault()
            ).format(Date())
        }_${label}.csv"

        val file = File(
            Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOCUMENTS
            ), fileName
        )

        file.bufferedWriter().use { writer ->
            // Headers
            writer.write("sample_index,time_ms,h_signal,v_signal,label\n")
            // Data rows
            readings.forEachIndexed { index, reading ->
                val rowLabel = when {
                    index > 0 -> ""                              // Empty for all other rows
                    label == EyeMovements.NONE.label -> "UNLABELED"  // First row, when no label selected
                    else -> label                                // First row, when label selected
                }
                writer.write("${reading.sampleIndex},${reading.timeMs},${reading.h},${reading.v},$rowLabel\n")
            }
        }
    }
}