package com.example.bluetootheog.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.FiberManualRecord
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults.topAppBarColors
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.bluetootheog.bluetooth.BluetoothManager
import com.example.bluetootheog.model.CircularBuffer
import com.example.bluetootheog.model.EyeMovements
import com.example.bluetootheog.repository.EOGRepository
import com.example.bluetootheog.utils.CSVExporter
import com.example.bluetootheog.utils.formatTime
import kotlinx.coroutines.delay

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EOGApp(bluetoothManager: BluetoothManager, repository: EOGRepository) {

    val hBuffer = remember { CircularBuffer(400) }
    val vBuffer = remember { CircularBuffer(400) }
    val hValues = remember { mutableStateOf(listOf<Float>()) }
    val vValues = remember { mutableStateOf(listOf<Float>()) }
    val isConnected = remember { mutableStateOf(false) }
    val isRecording = remember { mutableStateOf(false) }
    val isInferring = remember { mutableStateOf(false) }
    val elapsedSeconds = remember { mutableStateOf(0) }
    val selectedMovement = remember { mutableStateOf(EyeMovements.NONE) }
    val dropdownExpanded = remember { mutableStateOf(false) }
    val context = androidx.compose.ui.platform.LocalContext.current
    val predictionLabel = remember { mutableStateOf("") }

    val graphTick = remember { mutableStateOf(0) }

    // Refresh compose every 16ms
    LaunchedEffect(Unit) {
        while (true) {
            delay(16) // 60 FPS refresh
            graphTick.value++ // Force redraw of compose
        }
    }


    // Assigning callback to collect data
    LaunchedEffect(bluetoothManager) {
        bluetoothManager.onDataReceived = { sampleIndex, h, v ->
            hBuffer.add(h)
            vBuffer.add(v)
            hValues.value = hBuffer.toList()
            vValues.value = vBuffer.toList()
            repository.onNewData(sampleIndex, h, v)
        }

        // Observe connection status
        bluetoothManager.onConnectionChanged = { connected ->
            isConnected.value = connected
        }

        // Observe prediction status
        repository.onPredictionReady = { classLabel ->
            predictionLabel.value = classLabel
        }
    }

    // LaunchedEffect that ticks every second while recording
    LaunchedEffect(isRecording.value) {
        if (isRecording.value) {
            elapsedSeconds.value = 0  // Resets on every new recording
            while (isRecording.value) {
                delay(1000)
                elapsedSeconds.value++
            }
        }
    }

    Scaffold(topBar = {
        TopAppBar(
            colors = topAppBarColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer,
                titleContentColor = MaterialTheme.colorScheme.primary,
            ), title = {
                Text("Bluetooth EOG")
            })
    }, bottomBar = {
        BottomAppBar(
            containerColor = MaterialTheme.colorScheme.primaryContainer,
            contentColor = MaterialTheme.colorScheme.primary,
        ) {
            Text(
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.Center,
                text = "Neural Engineering Lab | IIT Guwahati",
            )
        }
    }, floatingActionButton = {
        Column(
            horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {

                // Prediction label shown, only when inferring
                if (isInferring.value && predictionLabel.value.isNotEmpty()) {
                    Text(
                        text = predictionLabel.value,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.tertiary
                    )
                }

                // Inference FAB
                ExtendedFloatingActionButton(
                    onClick = {
                        if (!isConnected.value) return@ExtendedFloatingActionButton
                        if (isRecording.value) return@ExtendedFloatingActionButton
                        if (isInferring.value) {
                            repository.stopInferring()
                            predictionLabel.value = ""
                            isInferring.value = false
                        } else {
                            repository.startInferring()
                            isInferring.value = true
                        }
                    }, containerColor = when {
                        !isConnected.value -> MaterialTheme.colorScheme.surfaceVariant
                        isRecording.value -> MaterialTheme.colorScheme.surfaceVariant
                        isInferring.value -> MaterialTheme.colorScheme.errorContainer
                        else -> MaterialTheme.colorScheme.primaryContainer
                    }, icon = {
                        Icon(
                            imageVector = Icons.Filled.Visibility,
                            contentDescription = "Inference",
                            tint = if (isInferring.value) MaterialTheme.colorScheme.error
                            else MaterialTheme.colorScheme.primary
                        )
                    }, text = { Text(if (isInferring.value) "Stop" else "Infer") })
            }

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {

                // Dropdown to show eye movement labels
                Box {
                    OutlinedButton(
                        onClick = {
                            if (isConnected.value && !isRecording.value && !isInferring.value) dropdownExpanded.value =
                                true
                        },
                        enabled = isConnected.value && !isRecording.value && !isInferring.value,
                        colors = ButtonDefaults.outlinedButtonColors(
                            containerColor = MaterialTheme.colorScheme.surface
                        )
                    ) {
                        Text(
                            text = selectedMovement.value.label,
                            style = MaterialTheme.typography.bodySmall
                        )
                        Icon(
                            Icons.Filled.ArrowDropDown, contentDescription = "Select movement"
                        )
                    }

                    DropdownMenu(
                        expanded = dropdownExpanded.value,
                        onDismissRequest = { dropdownExpanded.value = false }) {
                        EyeMovements.entries.forEach { movement ->
                            DropdownMenuItem(text = { Text(movement.label) }, onClick = {
                                selectedMovement.value = movement
                                dropdownExpanded.value = false
                            })
                        }
                    }
                }

                // Stopwatch shown in middle when recording
                if (isRecording.value) {
                    Text(
                        text = formatTime(elapsedSeconds.value),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.error
                    )
                }

                // Record button
                ExtendedFloatingActionButton(
                    onClick = {
                        if (!isConnected.value) return@ExtendedFloatingActionButton
                        if (isInferring.value) return@ExtendedFloatingActionButton
                    if (isRecording.value) {
                        val data = repository.stopRecording()
                        CSVExporter.saveToCSV(context, data, selectedMovement.value.label)
                        isRecording.value = false
                    } else {
                        repository.startRecording()
                        isRecording.value = true
                    }
                    }, containerColor = when {
                        !isConnected.value -> MaterialTheme.colorScheme.surfaceVariant  // disconnected
                        isInferring.value -> MaterialTheme.colorScheme.surfaceVariant
                        isRecording.value -> MaterialTheme.colorScheme.errorContainer  // recording
                        else -> MaterialTheme.colorScheme.primaryContainer // connected, ready
                    }, icon = {
                        Icon(
                            imageVector = if (isRecording.value) Icons.Filled.Stop
                            else Icons.Filled.FiberManualRecord,
                            contentDescription = "Record",
                            tint = if (isRecording.value) MaterialTheme.colorScheme.error // recording
                            else MaterialTheme.colorScheme.primary // idle
                        )
                    }, text = { Text(if (isRecording.value) "Stop" else "Record") })
            }
            // Connect/ Disconnect FAB
            ExtendedFloatingActionButton(
                onClick = {
                    if (isConnected.value) bluetoothManager.disconnectFromHC05()
                    else bluetoothManager.connectToHC05()
                },
                icon = {
                    Icon(
                        if (isConnected.value) Icons.Filled.Close else Icons.Filled.Add,
                        "Connect Bluetooth Button"
                    )
                },
                text = { Text(if (isConnected.value) "Disconnect" else "Connect") },
            )
        }
    }) { innerPadding ->
        Column(
            modifier = Modifier
                .padding(innerPadding)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {

            // Live EOG text
            Text(
                text = if (hValues.value.isNotEmpty() && vValues.value.isNotEmpty()) {
                    "H: ${hValues.value.last().toInt()}   V: ${vValues.value.last().toInt()}"
                } else {
                    "Waiting for connection..."
                }, modifier = Modifier.padding(8.dp)
            )

            // Live Graph
            Text("Horizontal Channel")
            LabeledGraph(
                values = hValues.value,
                tick = graphTick.value,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(200.dp),
                yLabel = "H (a.u.)",
                isConnected = isConnected.value
            )

            Text("Vertical Channel")
            LabeledGraph(
                values = vValues.value,
                tick = graphTick.value,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(200.dp),
                yLabel = "V (a.u.)",
                isConnected = isConnected.value
            )
        }
    }
}