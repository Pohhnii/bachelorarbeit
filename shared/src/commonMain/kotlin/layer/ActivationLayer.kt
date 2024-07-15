package layer

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import kotlin.math.exp
import kotlin.math.tanh


typealias ActivationFunction = (D2Array<Double>) -> D2Array<Double>

enum class ActivationFunctionType(
    val activate: ActivationFunction,
    val backwards: (input: D2Array<Double>, error: D2Array<Double>, output: D2Array<Double>) -> D2Array<Double>
) {
    SIGMOID(
        { arr -> arr.map { 1 / (1 + exp(-it)) } },
        { _, error, output -> output.map { it * (1 - it) } * error }
    ),

    TANH(
        { arr -> arr.map { tanh(it) } },
        { _, error, output -> output.map { (1 + it) * (1 - it) } * error }
    ),

    RELU(
        { arr -> arr.map { if (it > 0) it else 0.0 } },
        { input, error, _ -> input.map { if (it > 0) 1.0 else 0.0 } * error }
    ),

    LEAKY_RELU(
        { arr -> arr.map { if (it > 0) it else 0.01 * it } },
        { input, error, _ -> input.map { if (it > 0) 1.0 * it else 0.01 } * error }
    ),

    SOFTMAX({ arr ->
        val expSum = arr.map { exp(it) }.sum()
        arr.map { exp(it) / expSum }
    },
        { _, error, output ->
            val result = mk.ndarray(DoubleArray(error.size) { 0.0 }, error.shape[0], error.shape[1])
            for (i in result.data.indices) {
                val si = output.data[i]
                result.data[i] = si * (1 - si)
            }
            result * error
        }
    )
}

class ActivationLayer(private val activationFunction: ActivationFunctionType) : Layer() {

    override var data: List<D2Array<Double>> = emptyList()

    override fun forward(inputs: List<D2Array<Double>>): List<D2Array<Double>> {
        return inputs.map { activationFunction.activate(it) }
    }

    override fun copy(): Layer {
        return ActivationLayer(activationFunction)
    }

    override fun backpropagation(
        errors: List<D2Array<Double>>,
        inputs: List<D2Array<Double>>,
        outputs: List<D2Array<Double>>,
        learnRate: Double
    ): List<D2Array<Double>> {
        return errors.mapIndexed { index, error -> activationFunction.backwards(inputs[index], error, outputs[index]) }
    }
}