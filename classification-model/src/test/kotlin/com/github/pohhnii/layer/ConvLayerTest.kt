package com.github.pohhnii.layer

import layer.ConvLayer
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import kotlin.test.Test
import kotlin.test.assertEquals

class ConvLayerTest {

    @Test
    fun forward() {
        val layer = ConvLayer(kernelSize = 3, featureMaps = 3, padding = 1)
        val input = listOf(
            mk.ndarray(
                doubleArrayOf(
                    1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0
                ), 2, 3
            ),
            mk.ndarray(
                doubleArrayOf(
                    7.0, 8.0, 9.0,
                    10.0, 11.0, 12.0
                ), 2, 3
            )
        )

        val result = layer.forward(input)

        assertEquals(3, result.size)
        assertEquals(2, result[0].shape[0])
        assertEquals(3, result[0].shape[1])
        assertEquals(2, result[1].shape[0])
        assertEquals(3, result[1].shape[1])
        assertEquals(2, result[2].shape[0])
        assertEquals(3, result[2].shape[1])
    }

}