package com.csed404.sensor_classifier_app

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity(), SensorEventListener {
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var gravity: Sensor? = null

    private var isClassifying = false
    private val sensorDataBuffer = SensorDataBuffer(200, 100) // 2s @ 100Hz
    private val featureExtractor = FeatureExtractor()
    private lateinit var classifier: SVMClassifier

    private lateinit var classifiedActivityTextView: TextView
    private lateinit var startStopButton: Button

    private val coroutineScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI components
        classifiedActivityTextView = findViewById(R.id.classifiedActivity)
        startStopButton = findViewById(R.id.startStopButton)

        // Initialize sensors
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        gravity = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)

        // Initialize Model
        classifier = SVMClassifier.loadModel(this, "activity_model.model")

        // Start/Stop classification
        startStopButton.setOnClickListener {
            if (isClassifying) {
                stopClassification()
            } else {
                startClassification()
            }
        }
    }

    private fun startClassification() {
        isClassifying = true
        startStopButton.text = "Stop Classification"

        // Register sensor listeners
        accelerometer?.let { sensorManager.registerListener(this, it, 10000) }
        gyroscope?.let { sensorManager.registerListener(this, it, 10000) }
        gravity?.let { sensorManager.registerListener(this, it, 10000) }

        coroutineScope.launch {
            try {
                while (isClassifying) {
                    try {
                        Log.d("ClassificationLoop", "Starting iteration")

                        if (sensorDataBuffer.isReady()) {
                            Log.d("ClassificationLoop", "Sensor buffer is ready")

                            val features = try {
                                Log.d("FeatureExtraction", "Extracting features")
                                featureExtractor.extract(sensorDataBuffer)
                            } catch (e: Exception) {
                                Log.e("FeatureExtraction", "Error extracting features", e)
                                throw e
                            }

                            val predictedActivity = try {
                                Log.d("ActivityPrediction", "Predicting activity with features: $features")
                                classifier.predict(features)
                            } catch (e: Exception) {
                                Log.e("ActivityPrediction", "Error predicting activity", e)
                                throw e
                            }

                            withContext(Dispatchers.Main) {
                                Log.d("UIUpdate", "Updating UI with activity: $predictedActivity")
                                classifiedActivityTextView.text = "Current Activity: $predictedActivity"
                            }

                        } else {
                            Log.d("ClassificationLoop", "Sensor buffer not ready")
                        }

                        delay(1000) // Stride Δ = 1.0s
                    } catch (iterationE: Exception) {
                        Log.e("ClassificationLoop", "Error in classification iteration", iterationE)
                        delay(1000)
                    }
                }
            } catch (e: Exception) {
                Log.e("ClassificationLoop", "Unhandled exception in classification loop", e)
            }
        }
    }

    private fun stopClassification() {
        isClassifying = false
        startStopButton.text = "Start Classification"
        sensorManager.unregisterListener(this)
        coroutineScope.cancel()
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (isClassifying) {
            sensorDataBuffer.addSensorEvent(event)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}