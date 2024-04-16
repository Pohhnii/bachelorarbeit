package com.github.pohhnii.layer

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.junit.jupiter.api.Test

class DenseLayerTest {


    @Test
    fun feedForward() {
        val layer = DenseLayer(2, 8)
        val inputs = listOf(
            mk.ndarray(doubleArrayOf(1.0, 2.0), 1, 2),
            mk.ndarray(doubleArrayOf(3.0, 4.0), 1, 2),
            mk.ndarray(doubleArrayOf(5.0, 6.0), 1, 2)
        )

        val outputs = layer.forward(inputs)

        assert(outputs.size == 3)
        assert(outputs[0].shape[0] == 1)
        assert(outputs[0].shape[1] == 8)
    }

}