package com.github.pohhnii

import com.github.pohhnii.layer.Layer
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.joinToString
import java.io.File


class Model {

    private val layers = mutableListOf<Layer>()

    fun add(layer: Layer) {
        layers.add(layer)
    }

    fun forward(input: D2Array<Double>): List<D2Array<Double>> {
        var result = listOf(input)

        for (layer in layers) {
            result = layer.forward(result)
        }

        return result
    }

    fun copy(): Model {
        val model = Model()

        for (layer in layers) {
            model.add(layer.copy())
        }

        return model
    }

    fun save(directory: File): Boolean {
        if (!directory.isDirectory) return false

        for (i in layers.indices) {
            val layer = layers[i]
            val layerDir = File(directory, "layer-$i")
            layerDir.mkdirs()
            for (j in layer.data.indices) {
                val dataFile = File(layerDir, "data-$j.csv")
                dataFile.writeText(layer.data[j].joinToString(separator = ", "))
            }
        }

        return true
    }

    fun load(directory: File): Boolean {
        if (!directory.isDirectory) return false

        for (i in layers.indices) {
            val layer = layers[i]
            val layerDir = File(directory, "layer-$i")

            val newData = mutableListOf<D2Array<Double>>()
            for (j in layer.data.indices) {
                val dataFile = File(layerDir, "data-$j.csv")
                if (!dataFile.isFile) return false
                val fileContent = dataFile.readText()
                newData.add(mk.ndarray(fileContent.split(", ").map { it.toDouble() }, layer.data[j].shape))
            }

            layer.data = newData
        }

        return true
    }

    fun mutate(mutationRate: Double = 0.1, sigma: Double = 0.1) {
        for (layer in layers) {
            layer.mutate(mutationRate, sigma)
        }
    }

    fun crossover(other: Model) {
        for (i in layers.indices) layers[i].crossover(other.layers[i])
    }

}