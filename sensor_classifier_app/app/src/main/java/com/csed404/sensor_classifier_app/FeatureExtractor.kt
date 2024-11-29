package com.csed404.sensor_classifier_app

import android.util.Log

class FeatureExtractor {
    fun extract(buffer: SensorDataBuffer): List<Double> {
        try {
            Log.d("FeatureExtractor", "Starting feature extraction")

            val (accel, gyro, gravity) = try {
                Log.d("FeatureExtractor", "Retrieving buffered data")
                buffer.getBufferedData()
            } catch (e: Exception) {
                Log.e("FeatureExtractor", "Error retrieving buffered data", e)
                throw e
            }

            Log.d("FeatureExtractor", "Accelerometer data size: ${accel.size}")
            Log.d("FeatureExtractor", "Gyroscope data size: ${gyro.size}")
            Log.d("FeatureExtractor", "Gravity data size: ${gravity.size}")

            // Extract features for each sensor data type
            val accelFeatures = try {
                Log.d("FeatureExtractor", "Calculating accelerometer features")
                calculateFeatures(accel)
            } catch (e: Exception) {
                Log.e("FeatureExtractor", "Error calculating accelerometer features", e)
                throw e
            }

            val gyroFeatures = try {
                Log.d("FeatureExtractor", "Calculating gyroscope features")
                calculateFeatures(gyro)
            } catch (e: Exception) {
                Log.e("FeatureExtractor", "Error calculating gyroscope features", e)
                throw e
            }

            val gravityFeatures = try {
                Log.d("FeatureExtractor", "Calculating gravity features")
                calculateFeatures(gravity)
            } catch (e: Exception) {
                Log.e("FeatureExtractor", "Error calculating gravity features", e)
                throw e
            }

            // Combine all features into a single vector
            val combinedFeatures = accelFeatures + gravityFeatures + gyroFeatures

            Log.d("FeatureExtractor", "Combined features: $combinedFeatures")
            Log.d("FeatureExtractor", "Combined features size: ${combinedFeatures.size}")

            return combinedFeatures
        } catch (e: Exception) {
            Log.e("FeatureExtractor", "Unexpected error during feature extraction", e)
            throw e
        }
    }

    private fun calculateFeatures(data: MutableList<FloatArray>): List<Double> {
        try {
            Log.d("FeatureExtractor", "Calculating features for data of size ${data.size}")

            if (data.isEmpty()) {
                Log.w("FeatureExtractor", "Empty data list, returning default features")
                return listOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }

            val n = data.size.toDouble()

            // Calculate the sum of x, y, and z values
            var sumX = 0f
            var sumY = 0f
            var sumZ = 0f

            // Calculate the sum of squares of x, y, and z values
            var sumSqX = 0f
            var sumSqY = 0f
            var sumSqZ = 0f

            for (values in data) {
                sumX += values[0]
                sumY += values[1]
                sumZ += values[2]

                sumSqX += values[0] * values[0]
                sumSqY += values[1] * values[1]
                sumSqZ += values[2] * values[2]
            }

            // Calculate mean and variance for x, y, and z
            val meanX = sumX / n
            val meanY = sumY / n
            val meanZ = sumZ / n

            val varX = (sumSqX / n) - (meanX * meanX)
            val varY = (sumSqY / n) - (meanY * meanY)
            val varZ = (sumSqZ / n) - (meanZ * meanZ)

            return scaleFeatures(listOf(meanX, varX, meanY, varY, meanZ, varZ))
        } catch (e: Exception) {
            Log.e("FeatureExtractor", "Error calculating features", e)
            throw e
        }
    }

    private fun scaleFeatures(features: List<Double>): List<Double> {
        val min = features.minOrNull() ?: 0.0 // Minimum value in the feature set
        val max = features.maxOrNull() ?: 1.0 // Maximum value in the feature set

        return features.map { feature ->
            if (max - min == 0.0) 0.0 else 2 * (feature - min) / (max - min) - 1
        }
    }
}
