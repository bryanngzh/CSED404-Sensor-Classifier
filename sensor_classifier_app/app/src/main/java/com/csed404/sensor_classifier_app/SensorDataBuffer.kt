package com.csed404.sensor_classifier_app

import android.hardware.Sensor
import android.hardware.SensorEvent
import androidx.collection.CircularArray

class SensorDataBuffer(private val windowSize: Int, private val stride: Int) {
    private val accelData = CircularArray<FloatArray>()
    private val gyroData = CircularArray<FloatArray>()
    private val gravityData = CircularArray<FloatArray>()

    fun addSensorEvent(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION -> addData(accelData, event.values)
            Sensor.TYPE_GYROSCOPE -> addData(gyroData, event.values)
            Sensor.TYPE_GRAVITY -> addData(gravityData, event.values)
        }
    }

    private fun addData(buffer: CircularArray<FloatArray>, values: FloatArray) {
        val newValues = floatArrayOf(values[0], values[1], values[2])
        buffer.addLast(newValues)
        if (buffer.size() > windowSize + stride) {
            repeat(stride) {
                buffer.popFirst()
            }
        }
    }

    fun isReady(): Boolean {
        return accelData.size() >= windowSize && gyroData.size() >= windowSize && gravityData.size() >= windowSize
    }

    fun getBufferedData(): Triple<MutableList<FloatArray>, MutableList<FloatArray>, MutableList<FloatArray>> {
        val accelList = mutableListOf<FloatArray>()
        val gyroList = mutableListOf<FloatArray>()
        val gravityList = mutableListOf<FloatArray>()

        for (i in 0 until windowSize) {
            accelList.add(accelData[i])
        }

        for (i in 0 until windowSize) {
            gravityList.add(gravityData[i])
        }

        for (i in 0 until windowSize) {
            gyroList.add(gyroData[i])
        }

        return Triple(accelList, gravityList, gyroList)
    }

}