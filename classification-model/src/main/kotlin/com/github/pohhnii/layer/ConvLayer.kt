package com.github.pohhnii.layer

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import kotlin.random.Random

class ConvLayer(private val kernelSize: Int, private val featureMaps: Int, private val padding: Int = 0) : Layer() {

    override var data: List<D2Array<Double>> = buildList {
        for (i in 0 until featureMaps) {
            val kernel = mk.ndarray(
                DoubleArray(kernelSize * kernelSize) { Random.nextDouble(-1.0, 1.0) },
                kernelSize,
                kernelSize
            )
            val bias = mk.zeros<Double>(1, 1)

            add(kernel)
            add(bias)
        }
    }

    private fun getKernel(index: Int): D2Array<Double> = data[index * 2]
    private fun getBias(index: Int): D2Array<Double> = data[index * 2 + 1]

    override fun forward(inputs: List<D2Array<Double>>): List<D2Array<Double>> {
        val padded = if (padding > 0) inputs.map { addPadding(it) } else inputs
        val featureMaps = mutableListOf<D2Array<Double>>()

        for (i in 0 until this.featureMaps) {
            val convolvedInputs = padded.map { convolution(it, getKernel(i), getBias(i)) }
            featureMaps.add(convolvedInputs.reduce { feature, convolved -> feature + convolved })
        }

        return featureMaps
    }

    override fun copy(): Layer {
        val layer = ConvLayer(kernelSize, featureMaps, padding)
        layer.data = data.map { it.copy() }
        return layer
    }

    private fun convolution(input: D2Array<Double>, kernel: D2Array<Double>, bias: D2Array<Double>): D2Array<Double> {
        val resultCols = input.shape[0] - kernel.shape[0] + 1
        val resultRows = input.shape[1] - kernel.shape[1] + 1
        val result = mk.zeros<Double>(resultCols, resultRows)

        for (i in 0 until resultCols) {
            for (j in 0 until resultRows) {
                val subMatrix = input[i until i + kernel.shape[0], j until j + kernel.shape[1]]
                result[i, j] = (subMatrix * kernel).sum() + bias[0, 0]
            }
        }

        return result
    }

    private fun addPadding(input: D2Array<Double>): D2Array<Double> {
        val padded = mk.d2array(
            input.shape[0] + padding * 2,
            input.shape[1] + padding * 2
        ) { 0.0 }

        for (i in 0 until input.shape[0]) {
            for (j in 0 until input.shape[1]) {
                padded[i + padding, j + padding] = input[i, j]
            }
        }

        return padded
    }

}