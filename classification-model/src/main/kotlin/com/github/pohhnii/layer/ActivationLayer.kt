package com.github.pohhnii.layer

import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import kotlin.math.exp
import kotlin.math.tanh


typealias ActivationFunction = (D2Array<Double>) -> D2Array<Double>

enum class ActivationFunctionType(val activate: ActivationFunction) {
    SIGMOID({ arr -> arr.map { 1 / (1 + exp(-it)) } }),
    TANH({ arr -> arr.map { tanh(it) } }),
    RELU({ arr -> arr.map { if (it > 0) it else 0.0 } }),
    LEAKY_RELU({ arr -> arr.map { if (it > 0) it else 0.01 * it } }),
    SOFTMAX({ arr ->
        val expSum = arr.map { exp(it) }.sum()
        arr.map { exp(it) / expSum }
    })
}

class ActivationLayer(private val activationFunction: ActivationFunctionType) : Layer() {

    override var data: List<D2Array<Double>> = emptyList()

    override fun forward(inputs: List<D2Array<Double>>): List<D2Array<Double>> {
        return inputs.map { activationFunction.activate(it) }
    }

    override fun copy(): Layer {
        return ActivationLayer(activationFunction)
    }

}