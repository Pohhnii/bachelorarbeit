package com.github.pohhnii.layer

import layer.FlattenLayer
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class FlattenLayerTest {

    @Test
    fun feedForward() {
        val flattenLayer = FlattenLayer()
        val inputs = listOf(
            mk.ndarray(
                arrayOf(
                    doubleArrayOf(1.0, 2.0, 3.0),
                    doubleArrayOf(4.0, 5.0, 6.0),
                    doubleArrayOf(7.0, 8.0, 9.0)
                )
            ),
            mk.ndarray(
                arrayOf(
                    doubleArrayOf(10.0, 11.0, 12.0),
                    doubleArrayOf(13.0, 14.0, 15.0),
                    doubleArrayOf(16.0, 17.0, 18.0)
                )
            )
        )

        val outputs = flattenLayer.forward(inputs)
        assertEquals(1, outputs.size)
        assertEquals(1, outputs[0].shape[0])
        assertEquals(18, outputs[0].shape[1])

        val expected = mk.ndarray(
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                10.0, 11.0, 12.0,
                13.0, 14.0, 15.0,
                16.0, 17.0, 18.0
            ), 1, 18
        )

        assertEquals(expected, outputs[0])
    }

}