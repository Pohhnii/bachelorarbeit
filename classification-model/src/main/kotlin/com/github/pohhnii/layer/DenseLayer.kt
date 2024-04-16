package com.github.pohhnii.layer

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import kotlin.random.Random

class DenseLayer(private val inputs: Int, private val nodes: Int) : Layer() {
    override var data: List<D2Array<Double>> = listOf(
        mk.ndarray(DoubleArray(inputs * nodes) { Random.nextDouble(-1.0, 1.0) }, inputs, nodes),
        mk.zeros(1, nodes)
    )

    private val weights: D2Array<Double>
        get() = data[0]

    private val bias: D2Array<Double>
        get() = data[1]

    override fun forward(inputs: List<D2Array<Double>>): List<D2Array<Double>> {
        val outputs = mutableListOf<D2Array<Double>>()

        for (input in inputs) {
            outputs.add(input.dot(weights) + bias)
        }

        return outputs
    }

    override fun copy(): Layer {
        val layer = DenseLayer(inputs, nodes)
        layer.data = data.map { it.copy() }
        return layer
    }
}