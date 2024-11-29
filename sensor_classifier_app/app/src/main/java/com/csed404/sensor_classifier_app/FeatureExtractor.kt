package com.csed404.sensor_classifier_app

import android.util.Log

class FeatureExtractor {
    fun extract(buffer: SensorDataBuffer): List<Double> {
        try {
            Log.d("FeatureExtractor", "Starting feature extraction")

            val (accel, gravity, gyro) = try {
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

            return scaleFeatures(combinedFeatures)
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

            return listOf(meanX, varX, meanY, varY, meanZ, varZ)
        } catch (e: Exception) {
            Log.e("FeatureExtractor", "Error calculating features", e)
            throw e
        }
    }

    private fun scaleFeatures(features: List<Double>): List<Double> {
        val rangeFile = mapOf(
            1 to Pair(-6.9469874256799979, 1.0444658363165003),
            2 to Pair(9.825174800575169e-05, 115.30016398551612),
            3 to Pair(-5.7359475886799984, 3.059779641644349),
            4 to Pair(8.1018296696998434e-05, 135.15430181123241),
            5 to Pair(-6.4505823278824996, 0.79364102275639981),
            6 to Pair(8.4828034519179976e-05, 151.62829775990298),
            7 to Pair(-9.470988697990002, 8.048963317880002),
            8 to Pair(3.1791779520062846e-07, 42.812955460133168),
            9 to Pair(-9.7623577165749964, 8.4149724245050006),
            10 to Pair(7.1640755550106405e-07, 11.352167317776766),
            11 to Pair(-4.5776806712200013, 9.4278232145149978),
            12 to Pair(8.5195131660498191e-07, 18.466017917173559),
            13 to Pair(-0.56770945230945002, 0.70243211312293508),
            14 to Pair(5.3845529637667709e-07, 14.486241177061888),
            15 to Pair(-0.62620162442690719, 0.82894594433394009),
            16 to Pair(1.7858889730798361e-06, 25.811976868207161),
            17 to Pair(-0.49441600293095, 0.84737761478579532),
            18 to Pair(1.0901294404476607e-06, 15.316157433549423)
        )

        val scaledFeatures = mutableListOf<Double>()

        val l = -1.0
        val u = 1.0

        for ((index, featureValue) in features.withIndex()) {
            val (min, max) = rangeFile[index + 1] ?: continue
            val scaledValue = ((featureValue - min) / (max - min)) * (u - l) + l
            scaledFeatures.add(scaledValue)
        }

        return scaledFeatures
    }
}
