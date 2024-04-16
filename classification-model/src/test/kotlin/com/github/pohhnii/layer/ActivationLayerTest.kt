package com.github.pohhnii.layer

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.junit.jupiter.api.Test

class ActivationLayerTest {


    @Test
    fun forwardReLU() {
        val layer = ActivationLayer(ActivationFunctionType.RELU)
        val input: List<D2Array<Double>> = listOf(mk.ndarray(doubleArrayOf(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0), 2, 3))

        val result = layer.forward(input)

        val expected = mk.ndarray(doubleArrayOf(0.0, 0.0, 1.0, 0.0, 0.0, 2.0), 2, 3)
        assert(result[0] == expected)
    }

}