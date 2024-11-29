package com.csed404.sensor_classifier_app

import android.hardware.Sensor
import android.hardware.SensorEvent

class SensorDataBuffer(private val windowSize: Int)  {
    private val accelData = mutableListOf<FloatArray>()
    private val gyroData = mutableListOf<FloatArray>()
    private val gravityData = mutableListOf<FloatArray>()

    fun addSensorEvent(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION -> addData(accelData, event.values)
            Sensor.TYPE_GYROSCOPE -> addData(gyroData, event.values)
            Sensor.TYPE_GRAVITY -> addData(gravityData, event.values)
        }
    }

    private fun addData(buffer: MutableList<FloatArray>, values: FloatArray) {
        val newValues = floatArrayOf(values[0], values[1], values[2])
        buffer.add(newValues)
        if (buffer.size > windowSize) buffer.removeAt(0)
    }

    fun isReady(): Boolean {
        return accelData.size >= windowSize && gyroData.size >= windowSize && gravityData.size >= windowSize
    }

    fun getBufferedData(): Triple<MutableList<FloatArray>, MutableList<FloatArray>, MutableList<FloatArray>> {
        return Triple(accelData, gyroData, gravityData)
    }

    fun advanceWindow() {
        accelData.clear()
        gyroData.clear()
        gravityData.clear()
    }
}