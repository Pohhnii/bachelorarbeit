package com.github.pohhnii.layer

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.average
import org.jetbrains.kotlinx.multik.ndarray.operations.max


typealias PoolFunction = (MultiArray<Double, D2>) -> Double

enum class PoolType(val pool: PoolFunction) {
    MAX({ it.max()!! }), AVG({ it.average() })
}


class PoolLayer(private val poolType: PoolType, private val kernelSize: Int, private val stride: Int = 1) : Layer() {
    override var data: List<D2Array<Double>> = emptyList()

    override fun forward(inputs: List<D2Array<Double>>): List<D2Array<Double>> {
        return inputs.map { pool(it) }
    }

    override fun copy(): Layer {
        return PoolLayer(poolType, kernelSize, stride)
    }

    private fun pool(input: D2Array<Double>): D2Array<Double> {
        val resultCols = (input.shape[0] - kernelSize) / stride + 1
        val resultRows = (input.shape[1] - kernelSize) / stride + 1
        val result = mk.zeros<Double>(resultCols, resultRows)

        for (i in 0 until resultCols) {
            for (j in 0 until resultRows) {
                val subMatrix = input[
                    i * stride until i * stride + kernelSize,
                    j * stride until j * stride + kernelSize
                ]

                result[i, j] = poolType.pool(subMatrix)
            }
        }

        return result
    }
}