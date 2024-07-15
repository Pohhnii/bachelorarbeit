package layer

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import kotlin.random.Random

open class ConvLayer(private val kernelSize: Int, private val featureMaps: Int, private val padding: Int = 0) : Layer() {

    override var data: List<D2Array<Double>> = buildList {
        for (i in 0 until featureMaps) {
            val kernel = mk.ndarray(
                DoubleArray(kernelSize * kernelSize) { Random.nextDouble(-1.0, 1.0) },
                kernelSize,
                kernelSize
            )
            val bias = mk.zeros<Double>(1, 1)

            add(kernel)
            add(bias)
        }
    }

    private fun getKernel(index: Int): D2Array<Double> = data[index * 2]
    private fun getBias(index: Int): D2Array<Double> = data[index * 2 + 1]

    override fun forward(inputs: List<D2Array<Double>>): List<D2Array<Double>> {
        val padded = if (padding > 0) inputs.map { addPadding(it, padding) } else inputs
        val featureMaps = mutableListOf<D2Array<Double>>()

        for (i in 0 until this.featureMaps) {
            val convolvedInputs = padded.map { convolution(it, getKernel(i), getBias(i)) }
            featureMaps.add(convolvedInputs.reduce { feature, convolved -> feature + convolved })
        }

        return featureMaps
    }

    override fun copy(): Layer {
        val layer = ConvLayer(kernelSize, featureMaps, padding)
        layer.data = data.map { it.copy() }
        return layer
    }

    override fun backpropagation(
        errors: List<D2Array<Double>>,
        inputs: List<D2Array<Double>>,
        outputs: List<D2Array<Double>>,
        learnRate: Double
    ): List<D2Array<Double>> {
        val padded = if (padding > 0) inputs.map { addPadding(it, padding) } else inputs

        val dx = padded.map { mk.ndarray(DoubleArray(it.size) { 0.0 }, it.shape[0], it.shape[1]) }.toMutableList()
        val df = Array(featureMaps) { mk.ndarray(DoubleArray(kernelSize * kernelSize), kernelSize, kernelSize) }
        val db = Array(featureMaps) { mk.ndarray(DoubleArray(1) { 0.0 }, 1, 1) }

        for (i in 0 until featureMaps) {
            for (j in padded.indices) {
                df[i] += convolution(padded[j], errors[i])
            }

            db[i].data[0] += errors[i].sum()
        }

        val kernelPadding = errors[0].shape[0] - 1
        for (i in 0 until featureMaps) {
            val kernel = getKernel(i)
            val paddedKernel = addPadding(kernel, kernelPadding)
            for (j in dx.indices) {
                dx[j] += convolution(paddedKernel, errors[i])
            }
        }

        val newData = mutableListOf<D2Array<Double>>()
        for (i in 0 until featureMaps) {
            val bias = getBias(i)
            val kernel = getKernel(i)

            newData.add(kernel + (df[i] * learnRate))
            newData.add(bias + (db[i] * learnRate))
        }

        this.data = newData

        return dx.map { removePadding(it, padding) }
    }

    protected open fun convolution(input: D2Array<Double>, kernel: D2Array<Double>, bias: D2Array<Double>): D2Array<Double> {
        val resultCols = input.shape[0] - kernel.shape[0] + 1
        val resultRows = input.shape[1] - kernel.shape[1] + 1
        val result = mk.zeros<Double>(resultCols, resultRows)

        for (i in 0 until resultCols) {
            for (j in 0 until resultRows) {
                val subMatrix = input[i until i + kernel.shape[0], j until j + kernel.shape[1]]
                result[i, j] = (subMatrix * kernel).sum() + bias[0, 0]
            }
        }

        return result
    }

    private fun convolution(input: D2Array<Double>, kernel: D2Array<Double>): D2Array<Double> {
        val resultCols = input.shape[0] - kernel.shape[0] + 1
        val resultRows = input.shape[1] - kernel.shape[1] + 1
        val result = mk.zeros<Double>(resultCols, resultRows)

        for (i in 0 until resultCols) {
            for (j in 0 until resultRows) {
                val subMatrix = input[i until i + kernel.shape[0], j until j + kernel.shape[1]]
                result[i, j] = (subMatrix * kernel).sum()
            }
        }

        return result
    }

    protected open fun addPadding(input: D2Array<Double>, padding: Int): D2Array<Double> {
        val padded = mk.d2array(
            input.shape[0] + padding * 2,
            input.shape[1] + padding * 2
        ) { 0.0 }

        for (i in 0 until input.shape[0]) {
            for (j in 0 until input.shape[1]) {
                padded[i + padding, j + padding] = input[i, j]
            }
        }

        return padded
    }

    private fun removePadding(padded: D2Array<Double>, padding: Int): D2Array<Double> {
        val original = mk.d2array(
            padded.shape[0] - padding * 2,
            padded.shape[1] - padding * 2
        ) { 0.0 }

        for (i in 0 until original.shape[0]) {
            for (j in 0 until original.shape[1]) {
                original[i, j] = padded[i + padding, j + padding]
            }
        }

        return original
    }

}