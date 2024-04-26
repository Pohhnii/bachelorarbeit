package layer

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.jetbrains.kotlinx.multik.ndarray.operations.times

class ConvLayerJs(kernelSize: Int, featureMaps: Int, padding: Int = 0) : ConvLayer(kernelSize, featureMaps, padding) {

    override fun addPadding(input: D2Array<Double>, padding: Int): D2Array<Double> {
        val padded = mk.d2array(
            input.shape[0] + padding * 2,
            input.shape[1] + padding * 2
        ) { 0.0 }

        for (i in 0 until input.shape[0]) {
            for (j in 0 until input.shape[1]) {
                padded[i + padding, j + padding] = input.data[i * input.shape[1] + j]
            }
        }

        return padded
    }

    override fun convolution(input: D2Array<Double>, kernel: D2Array<Double>, bias: D2Array<Double>): D2Array<Double> {
        val resultCols = input.shape[0] - kernel.shape[0] + 1
        val resultRows = input.shape[1] - kernel.shape[1] + 1
        val result = mk.zeros<Double>(resultCols, resultRows)

        for (i in 0 until resultCols) {
            for (j in 0 until resultRows) {
                val subMatrix = input[i until i + kernel.shape[0], j until j + kernel.shape[1]]
                result[i, j] = (subMatrix * kernel).sum() + bias.data[0]
            }
        }

        return result
    }

}