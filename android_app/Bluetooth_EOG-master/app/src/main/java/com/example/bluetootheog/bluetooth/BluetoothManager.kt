package com.example.bluetootheog.bluetooth

import android.Manifest
import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothSocket
import android.content.pm.PackageManager
import android.util.Log
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.InputStream
import kotlin.concurrent.thread

class BluetoothManager(private val activity: Activity) {

    // Variable declaration
    var onDataReceived: ((Long, Float, Float) -> Unit)? = null
    var onConnectionChanged: ((Boolean) -> Unit)? = null
    private val REQUEST_BLUETOOTH_PERMISSIONS = 1
    private var bluetoothAdapter: BluetoothAdapter? = null
    private var bluetoothSocket: BluetoothSocket? = null
    private var inputStream: InputStream? = null
    private var isReading = false
    var isConnected = false
        private set

    // Bluetooth permission object
    private val bluetoothPermissions = arrayOf(
        Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.BLUETOOTH_SCAN
    )

    // Function to check whether bluetooth permission is granted
    fun checkBluetoothPermissions() {
        val missingPermissions = bluetoothPermissions.filter {
            ContextCompat.checkSelfPermission(activity, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missingPermissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                activity, missingPermissions.toTypedArray(), REQUEST_BLUETOOTH_PERMISSIONS
            )
        } else {
            Log.d("BluetoothEOG", "All Bluetooth permissions granted.")
            Toast.makeText(activity, "All Bluetooth permissions granted.", Toast.LENGTH_SHORT)
                .show()
        }
    }

    // Function to initialize Bluetooth Adapter object
    // Checks whether bluetooth is supported and enabled
    private fun initializeBluetooth() {
        bluetoothAdapter = BluetoothAdapter.getDefaultAdapter()
        if (bluetoothAdapter == null) {
            Log.e("BluetoothEOG", "Bluetooth not supported on this device.")
            Toast.makeText(activity, "Bluetooth not supported on this device.", Toast.LENGTH_SHORT)
                .show()
            return
        }

        if (!bluetoothAdapter!!.isEnabled) {
            Log.e("BluetoothEOG", "Bluetooth is disabled. Please enable it.")
            Toast.makeText(activity, "Bluetooth is disabled. Please enable it.", Toast.LENGTH_SHORT)
                .show()
        } else {
            Log.d("BluetoothEOG", "Bluetooth is ON.")
        }
    }

    fun connectToHC05() {

        if (isConnected) {
            disconnectFromHC05()
            return
        }

        initializeBluetooth()

        if (ActivityCompat.checkSelfPermission(
                activity, Manifest.permission.BLUETOOTH_CONNECT
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            Toast.makeText(
                activity, "Permission not granted for Bluetooth connect.", Toast.LENGTH_SHORT
            ).show()
            return
        }

        val deviceName = "HC-05"
        val pairedDevices: Set<BluetoothDevice>? = bluetoothAdapter?.bondedDevices
        if (pairedDevices.isNullOrEmpty()) {
            Toast.makeText(activity, "No paired devices found", Toast.LENGTH_LONG).show()
            return
        }

        val hc05Device = pairedDevices.find { it.name.equals(deviceName, ignoreCase = true) }
        if (hc05Device == null) {
            Toast.makeText(activity, "HC-05 not found. Pair it first.", Toast.LENGTH_SHORT).show()
            return
        }

        thread {
            try {
                activity.runOnUiThread {
                    Toast.makeText(activity, "Connecting to HC-05...", Toast.LENGTH_SHORT).show()
                }

                // Trying normal SPP UUID first
                val uuid = java.util.UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
                var socket = hc05Device.createInsecureRfcommSocketToServiceRecord(uuid)

                try {
                    bluetoothAdapter?.cancelDiscovery()
                    socket.connect()
                } catch (e: Exception) {
                    // Fallback using reflection- Manually connecting socket
                    Log.e("BluetoothEOG", "Standard connect failed, retrying via reflection...")
                    val method = hc05Device.javaClass.getMethod(
                        "createRfcommSocket", Int::class.javaPrimitiveType
                    )
                    socket = method.invoke(hc05Device, 1) as BluetoothSocket
                    socket.connect()
                }

                bluetoothSocket = socket
                isConnected = true
                activity.runOnUiThread {
                    Toast.makeText(activity, "Connected to HC-05!", Toast.LENGTH_SHORT).show()
                    onConnectionChanged?.invoke(true)
                }
                startReading()
            } catch (e: Exception) {
                Log.e("BluetoothEOG", "Connection failed: ${e.message}")
                activity.runOnUiThread {
                    Toast.makeText(activity, "Connection failed: ${e.message}", Toast.LENGTH_LONG)
                        .show()
                    onConnectionChanged?.invoke(false)
                }
            }
        }
    }

    // Bluetooth disconnect function
    fun disconnectFromHC05() {
        thread {
            try {
                isReading = false
                inputStream?.close()
                bluetoothSocket?.close()
                bluetoothSocket = null
                isConnected = false

                activity.runOnUiThread {
                    Toast.makeText(activity, "Disconnected from HC-05", Toast.LENGTH_SHORT).show()
                    onConnectionChanged?.invoke(false)
                }
            } catch (e: Exception) {
                Log.e("BluetoothEOG", "Disconnection error: ${e.message}")
            }
        }
    }

    private fun startReading() {
        val socket = bluetoothSocket ?: return
        inputStream = socket.inputStream
        isReading = true

        thread {
            val reader = inputStream?.bufferedReader()

            while (isReading) {
                try {
                    val line = reader?.readLine() ?: break

                    val parts = line.trim().split(",")

                    if (parts.size == 3) {
                        val sampleIndex = parts[0].toLongOrNull()
                        val hVal = parts[1].toFloatOrNull()
                        val vVal = parts[2].toFloatOrNull()

                        if (sampleIndex != null && hVal != null && vVal != null) {
                            activity.runOnUiThread {
                                onDataReceived?.invoke(sampleIndex, hVal, vVal)
                            }
                        }
                    }

                } catch (e: Exception) {
                    Log.e("BluetoothEOG", "Read error: ${e.message}")
                    isReading = false
                }
            }
        }
    }
}