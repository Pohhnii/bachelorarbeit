package com.github.pohhnii.layer

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import kotlin.test.Test
import kotlin.test.assertEquals

class PoolLayerTest {

    @Test
    fun forwardMax() {
        val layer = PoolLayer(PoolType.MAX, 2, 2)
        val input = listOf(
            mk.ndarray(
                doubleArrayOf(
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                    19.0, 20.0, 21.0, 22.0, 23.0, 24.0
                ), 4, 6
            ),
            mk.ndarray(
                doubleArrayOf(
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                    19.0, 20.0, 21.0, 22.0, 23.0, 24.0
                ), 4, 6
            )
        )

        val result = layer.forward(input)

        assertEquals(2, result.size)
        assertEquals(2, result[0].shape[0])
        assertEquals(3, result[0].shape[1])

        assertEquals(mk.ndarray(doubleArrayOf(8.0, 10.0, 12.0, 20.0, 22.0, 24.0), 2, 3), result[0])
    }

    @Test
    fun forwardAvg() {
        val layer = PoolLayer(PoolType.AVG, 2, 2)
        val input = listOf(
            mk.ndarray(
                doubleArrayOf(
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                    19.0, 20.0, 21.0, 22.0, 23.0, 24.0
                ), 4, 6
            ),
            mk.ndarray(
                doubleArrayOf(
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                    19.0, 20.0, 21.0, 22.0, 23.0, 24.0
                ), 4, 6
            )
        )

        val result = layer.forward(input)

        assertEquals(2, result.size)
        assertEquals(2, result[0].shape[0])
        assertEquals(3, result[0].shape[1])

        assertEquals(mk.ndarray(doubleArrayOf(4.5, 6.5, 8.5, 16.5, 18.5, 20.5), 2, 3), result[0])
    }

}