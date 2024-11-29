package com.csed404.sensor_classifier_app

import android.content.Context
import android.util.Log
import libsvm.svm
import libsvm.svm_model
import libsvm.svm_node
import java.io.File
import java.io.IOException

class SVMClassifier(private val model: svm_model) {

    companion object {
        fun loadModel(context: Context, modelFileName: String): SVMClassifier {
            // Create a file in the app's internal storage directory
            val modelFile = File(context.filesDir, modelFileName)

            // If the file doesn't already exist in internal storage, copy it from assets
            if (!modelFile.exists()) {
                try {
                    // Open the file from assets
                    context.assets.open(modelFileName).use { input ->
                        // Create an output stream to the internal storage file
                        modelFile.outputStream().use { output ->
                            // Copy the file from assets to internal storage
                            input.copyTo(output)
                        }
                    }
                } catch (e: IOException) {
                    Log.e("ModelLoader", "Error copying model from assets", e)
                    throw RuntimeException("Could not copy model file from assets", e)
                }
            }

            // Load the model from the file in internal storage
            return SVMClassifier(svm.svm_load_model(modelFile.absolutePath))
        }
    }

    fun predict(features: List<Double>): String {
        // Map features to svm_node array
        val nodes = features.mapIndexed { index, value ->
            svm_node().apply {
                this.index = index + 1
                this.value = value
            }
        }.toTypedArray()

        val nodeStrings = nodes.joinToString(", ") { "index: ${it.index}, value: ${it.value}" }
        Log.d("SVMClassifier", "Nodes: $nodeStrings")

        // Perform prediction using the SVM model
        val label = svm.svm_predict(model, nodes)
        Log.d("SVMClassifier", "Label: $label")
        return mapLabelToActivity(label)
    }

    private fun mapLabelToActivity(label: Double): String {
        return when (label) {
            0.0 -> "Others"
            1.0 -> "Walking"
            2.0 -> "Running"
            3.0 -> "Standing"
            4.0 -> "Sitting"
            5.0 -> "Upstairs"
            6.0 -> "Downstairs"
            else -> "Unknown"
        }
    }
}
