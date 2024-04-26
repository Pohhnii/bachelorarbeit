package layer

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.average
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.max
import org.jetbrains.kotlinx.multik.ndarray.operations.times


typealias PoolFunction = (MultiArray<Double, D2>) -> Double

enum class PoolType(val pool: PoolFunction, val backwards: (MultiArray<Double, D2>, Double) -> MultiArray<Double, D2>) {
    MAX(
        { it.max()!! },
        { input, output -> input.map { if (it == output) 1.0 else 0.0 } }
    ),
    AVG(
        { it.average() },
        { input, _ -> input.map { 1.0 / input.size } }
    )
}


class PoolLayer(private val poolType: PoolType, private val kernelSize: Int, private val stride: Int = 1) : Layer() {
    override var data: List<D2Array<Double>> = emptyList()

    override fun forward(inputs: List<D2Array<Double>>): List<D2Array<Double>> {
        return inputs.map { pool(it) }
    }

    override fun copy(): Layer {
        return PoolLayer(poolType, kernelSize, stride)
    }

    override fun backpropagation(
        errors: List<D2Array<Double>>,
        inputs: List<D2Array<Double>>,
        outputs: List<D2Array<Double>>,
        learnRate: Double
    ): List<D2Array<Double>> {
        val dxs = mutableListOf<D2Array<Double>>()
        for (i in inputs.indices) {
            val resultRows = outputs[i].shape[0]
            val resultCols = outputs[i].shape[1]
            val dx = mk.ndarray(DoubleArray(inputs[i].size), inputs[i].shape[0], inputs[i].shape[1])

            for (j in 0 until resultRows) {
                for (k in 0 until resultCols) {
                    val inputRow = j * stride
                    val inputCol = k * stride

                    val dxSlice = poolType.backwards(
                        inputs[i][inputRow until inputRow + kernelSize, inputCol until inputCol + kernelSize],
                        outputs[i][j, k]
                    ) * errors[i][j, k]

                    for (l in 0 until kernelSize) {
                        for (m in 0 until kernelSize) {
                            dx[inputRow + l, inputCol + m] += dxSlice[l, m]
                        }
                    }
                }
            }

            dxs.add(dx)
        }

        return dxs
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