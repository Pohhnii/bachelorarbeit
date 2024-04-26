package layer

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.set

class FlattenLayer : Layer() {
    override var data: List<D2Array<Double>> = emptyList()

    override fun forward(inputs: List<D2Array<Double>>): List<D2Array<Double>> {
        val featureSize = inputs[0].size
        val outputSize = inputs.size * featureSize
        val output = mk.ndarray(DoubleArray(outputSize), 1, outputSize)

        for (i in inputs.indices) {
            for (j in 0 until featureSize) {
                output[0, i * featureSize + j] = inputs[i].data[j]
            }
        }

        return listOf(output)
    }

    override fun copy(): Layer {
        return FlattenLayer()
    }

    override fun backpropagation(
        errors: List<D2Array<Double>>,
        inputs: List<D2Array<Double>>,
        outputs: List<D2Array<Double>>,
        learnRate: Double
    ): List<D2Array<Double>> {
        val dxs = mutableListOf<D2Array<Double>>()

        for (i in inputs.indices) {
            val inputRows = inputs[i].shape[0]
            val inputCols = inputs[i].shape[1]
            val dx = mk.ndarray(DoubleArray(inputRows * inputCols), inputRows, inputCols)

            for (j in 0 until inputRows) {
                for (k in 0 until inputCols) {
                    dx[j, k] = errors[0].data[j * inputCols + k]
                }
            }

            dxs.add(dx)
        }

        return dxs
    }

}